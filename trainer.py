"""難易度予測モデルの学習。サブコマンド cv / full で 2 つの手順を切り替える。

cv（交差検証）: 手順の良し悪しを比べて採用候補を決める
  1. 最終確認用に全体の 2 割を先に取り分ける（ラベルの帯で層化・種固定）。候補選びの間は一切使わない
  2. 残り 8 割で 5 分割交差検証（層化）。指標は MAE と MSE。分割ごとの値と平均・標準偏差を出す
  3. 入力（データファイル）× ハイパーパラメータ候補 のすべてを交差検証し、1 組を選ぶ:
       交差検証の平均 MAE が最小の組。ただし最小の組がゲームの入力（列が x1..x24 のデータ）でないとき、
       ゲームの入力の最良との差が「分割ごとの MAE の標準偏差（両者の大きい方）」以内ならゲームの入力を採る
  4. 選んだ 1 組を残り 8 割で学習（種を 3 つ変えて 3 本。早期終了のために 8 割の中から 1 割を検証用に取る）し、
     最終確認用 2 割で 1 回だけ測る。3 本のうち検証 MAE が中央のものを採用候補にする

full（全データ）: 決まった手順で最終版のモデルを作る
  cv が書いた chosen.json のデータとハイパーパラメータで、全行を使って種 0 / 1 / 2 の 3 本を学習する（最終確認用の取り分けはしない）。
  早期終了のために種ごとに全体から 1 割を検証用に取る（層化・種ごとに違う分け方）。
  3 本は export_model.py で平均モデル（Ensemble）1 つの TorchScript に書き出す

入力ファイルは行順が同じ前提（同じ譜面が同じ行）。分割はラベルと種だけで決まるので、どのファイルでも同じ行が同じ分割になる。

使い方:
  uv run python trainer.py cv --data data.csv data_4mode.csv --out retrain_2026-09-10/cv --baseline model.pt.bak
  uv run python trainer.py full --chosen retrain_2026-09-10/cv/chosen.json --out retrain_2026-09-10/full
出力（--out の下）:
  cv:
    split.csv          行ごとの分割（holdout / fold）
    cv_results.csv     交差検証の全結果（データ × 候補 × 分割）
    cv_summary.csv     データ × 候補ごとの平均・標準偏差
    chosen.json        選んだ組（data / hp）と採用候補。full が読む
    final_results.csv  採用候補 3 本（検証 MAE・最終確認用 2 割の MAE/MSE）
    final_seed<N>.pt   3 本の重み（state_dict）。export_model.py で TorchScript にする
    baseline.csv       --baseline の現行モデルを最終確認用 2 割で測った値
    summary.md         人が読む要約
  full:
    full_split.csv     行ごとの検証用の印（種ごと）
    full_results.csv   3 本の検証 MAE / MSE・ベストエポック
    full_seed<N>.pt    3 本の重み（state_dict）。export_model.py で平均モデルにする
    full_summary.md    人が読む要約
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from make_dataset import GAME_INPUT_COLS
from model import Net

HP_GRID = [{"lr": lr, "weight_decay": wd} for lr in (1e-3, 3e-4) for wd in (0.0, 1e-4, 1e-3)]
BATCH_SIZE = 16
MAX_EPOCHS = 3000
PATIENCE = 100
N_FOLDS = 5
HOLDOUT_RATIO = 0.2
INNER_VAL_RATIO = 0.1
FINAL_SEEDS = (0, 1, 2)


def label_band(y):
    """層化に使うラベルの帯。100 以上は 1 つの帯（本数が少ないので細かく分けない）。"""
    return np.where(y >= 100, 100, (y // 20) * 20)


def stratified_split(y, ratio, rng):
    """各帯から ratio の割合を取り分ける。返り値は取り分けた行の真偽配列。"""
    pick = np.zeros(len(y), dtype=bool)
    band = label_band(y)
    for b in np.unique(band):
        idx = np.flatnonzero(band == b)
        rng.shuffle(idx)
        n = int(round(len(idx) * ratio))
        pick[idx[:n]] = True
    return pick


def stratified_folds(y, n_folds, rng):
    """各帯の中で行を混ぜてから順番に分割へ割り当てる。"""
    fold = np.full(len(y), -1, dtype=int)
    band = label_band(y)
    for b in np.unique(band):
        idx = np.flatnonzero(band == b)
        rng.shuffle(idx)
        for i, row in enumerate(idx):
            fold[row] = i % n_folds
    return fold


def fit(x_tr, y_tr, x_val, y_val, hp, seed):
    """1 回の学習。検証 MSE が最小だった時点の重みを返す。"""
    torch.manual_seed(seed)
    net = Net(input_size=x_tr.shape[1])
    net.set_standardization(x_tr)
    opt = torch.optim.Adam(net.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    gen = torch.Generator().manual_seed(seed)

    best = {"val_mse": float("inf"), "epoch": -1, "state": None}
    train_losses = []
    since_best = 0
    n = len(x_tr)
    for epoch in range(MAX_EPOCHS):
        net.train()
        perm = torch.randperm(n, generator=gen)
        total = 0.0
        for start in range(0, n, BATCH_SIZE):
            b = perm[start:start + BATCH_SIZE]
            opt.zero_grad()
            loss = F.mse_loss(net(x_tr[b]).squeeze(1), y_tr[b])
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        train_losses.append(total / n)

        net.eval()
        with torch.no_grad():
            val_mse = F.mse_loss(net(x_val).squeeze(1), y_val).item()
        if val_mse < best["val_mse"]:
            best = {"val_mse": val_mse, "epoch": epoch, "state": {k: v.clone() for k, v in net.state_dict().items()}}
            since_best = 0
        else:
            since_best += 1
            if since_best >= PATIENCE:
                break

    net.load_state_dict(best["state"])
    net.eval()
    with torch.no_grad():
        pred = net(x_val).squeeze(1)
    tl = np.array(train_losses)
    diverged = bool(np.isnan(tl).any()) or bool(tl[-1] > 10 * tl.min())
    return {"net": net, "best_epoch": best["epoch"], "epochs_run": len(tl), "val_mse": best["val_mse"],
            "val_mae": F.l1_loss(pred, y_val).item(), "diverged": diverged}


def run_fits(jobs):
    """fit の引数の並び jobs を並列に学習し、同じ並びで結果を返す。

    各学習は独立（fit が自分で種を固定する）なのでプロセスで並列にする。各ワーカーは 1 スレッド
    （スレッド数が変わると浮動小数の足す順が変わり得るので、結果は 1 スレッドの学習として固定する）。
    """
    with ProcessPoolExecutor(max_workers=min(len(jobs), os.cpu_count()), initializer=torch.set_num_threads, initargs=(1,)) as ex:
        yield from ex.map(fit, *zip(*jobs))


def load_data(path: Path):
    df = pd.read_csv(path)
    x = torch.tensor(df.drop(columns="y").values, dtype=torch.float32)
    y = torch.tensor(df["y"].values, dtype=torch.float32)
    return x, y, [c for c in df.columns if c != "y"]


def hp_name(hp):
    return f"lr={hp['lr']:g},wd={hp['weight_decay']:g}"


def progress(tag, r, t0):
    print(f"[{tag}] MAE={r['val_mae']:.3f} MSE={r['val_mse']:.2f} best_epoch={r['best_epoch']} run={r['epochs_run']} "
          f"diverged={r['diverged']}  ({time.time() - t0:.0f}s)", flush=True)


def save_weights(path: Path, net: Net, cols, data_name: str, hp, seed: int, best_epoch: int):
    """export_model.py が読む形式で重みを保存する。"""
    torch.save({"state_dict": net.state_dict(), "input_size": net.input_size, "columns": cols, "data": data_name,
                "hp": hp, "seed": seed, "best_epoch": best_epoch}, path)


def fit_seeds(x, y, rows, hp, out: Path, prefix: str, data_name: str, cols, t0):
    """rows の中から種ごとに 1 割を検証用に取り分け（層化・種 1000+seed）、残りで学習した 3 本を <prefix>_seed<N>.pt に保存する。

    返り値: 種ごとの (fit の結果, 検証用の印（rows と同じ長さ）, 結果行)
    """
    picks = [stratified_split(y.numpy()[rows], INNER_VAL_RATIO, np.random.default_rng(1000 + seed)) for seed in FINAL_SEEDS]
    jobs = [(x[rows[~val]], y[rows[~val]], x[rows[val]], y[rows[val]], hp, seed) for seed, val in zip(FINAL_SEEDS, picks)]
    fits = []
    for seed, val, r in zip(FINAL_SEEDS, picks, run_fits(jobs)):
        save_weights(out / f"{prefix}_seed{seed}.pt", r["net"], cols, data_name, hp, seed, r["best_epoch"])
        row = {"model": f"{prefix}_seed{seed}", "data": data_name, "lr": hp["lr"], "weight_decay": hp["weight_decay"], "seed": seed,
               "train_rows": int((~val).sum()), "val_rows": int(val.sum()), "best_epoch": r["best_epoch"], "epochs_run": r["epochs_run"],
               "diverged": int(r["diverged"]), "val_mae": r["val_mae"], "val_mse": r["val_mse"]}
        progress(f"{prefix} seed{seed}", r, t0)
        fits.append((r, val, row))
    return fits


def train_cv(args, datasets, t0):
    """交差検証: 2 割を取り分け → 残りで データ × 候補 を 5 分割交差検証 → 1 組選ぶ → 3 本学習して 2 割で測る。"""
    ys = [y for _, y, _ in datasets.values()]
    for y in ys[1:]:
        if not torch.equal(y, ys[0]):
            raise SystemExit("データファイル間で y（行順）が違う")
    y_np = ys[0].numpy()
    game_data = [name for name, (_, _, cols) in datasets.items() if cols == GAME_INPUT_COLS]
    if len(game_data) > 1:
        raise SystemExit(f"ゲームの入力と同じ列のデータが 2 つ以上ある: {game_data}")
    game_data = game_data[0] if game_data else None

    # 1. 最終確認用 2 割
    rng = np.random.default_rng(args.seed)
    holdout = stratified_split(y_np, HOLDOUT_RATIO, rng)
    dev_rows = np.flatnonzero(~holdout)
    fold = np.full(len(y_np), -1, dtype=int)
    fold[dev_rows] = stratified_folds(y_np[dev_rows], N_FOLDS, rng)
    pd.DataFrame({"row": np.arange(1, len(y_np) + 1), "y": y_np.astype(int), "holdout": holdout.astype(int), "fold": fold}).to_csv(
        args.out / "split.csv", index=False, lineterminator="\n")
    print(f"rows={len(y_np)} holdout={int(holdout.sum())} dev={len(dev_rows)} folds={np.bincount(fold[dev_rows]).tolist()}")

    # 2-3. 交差検証
    keys = []
    jobs = []
    for name, (x, y, cols) in datasets.items():
        for hp in HP_GRID:
            for k in range(N_FOLDS):
                tr = dev_rows[fold[dev_rows] != k]
                va = dev_rows[fold[dev_rows] == k]
                keys.append((name, hp, k))
                jobs.append((x[tr], y[tr], x[va], y[va], hp, args.seed))
    cv_rows = []
    for (name, hp, k), r in zip(keys, run_fits(jobs)):
        cv_rows.append({"data": name, "lr": hp["lr"], "weight_decay": hp["weight_decay"], "fold": k,
                        "val_mae": r["val_mae"], "val_mse": r["val_mse"], "best_epoch": r["best_epoch"],
                        "epochs_run": r["epochs_run"], "diverged": int(r["diverged"])})
        progress(f"cv {name} {hp_name(hp)} fold{k}", r, t0)
    cv = pd.DataFrame(cv_rows)
    cv.to_csv(args.out / "cv_results.csv", index=False, lineterminator="\n")
    summary = cv.groupby(["data", "lr", "weight_decay"], sort=False).agg(
        mae_mean=("val_mae", "mean"), mae_std=("val_mae", "std"), mse_mean=("val_mse", "mean"), mse_std=("val_mse", "std"),
        best_epoch_mean=("best_epoch", "mean"), diverged=("diverged", "sum")).reset_index()
    summary.to_csv(args.out / "cv_summary.csv", index=False, lineterminator="\n")

    # 選ぶ
    best_all = summary.loc[summary["mae_mean"].idxmin()]
    chosen = best_all
    reason = "交差検証の平均 MAE が最小"
    if game_data is not None and best_all["data"] != game_data:
        s_game = summary[summary["data"] == game_data]
        best_game = s_game.loc[s_game["mae_mean"].idxmin()]
        tol = max(best_all["mae_std"], best_game["mae_std"])
        diff = best_game["mae_mean"] - best_all["mae_mean"]
        if diff <= tol:
            chosen = best_game
            reason = (f"最小は {best_all['data']} {hp_name(best_all)}（{best_all['mae_mean']:.3f}）だが、ゲームの入力 {game_data} の最良"
                      f"（{best_game['mae_mean']:.3f}）との差 {diff:.3f} が分割ごとの標準偏差 {tol:.3f} 以内なのでゲームの入力を採る")
        else:
            reason = f"ゲームの入力 {game_data} の最良（{best_game['mae_mean']:.3f}）との差 {diff:.3f} が分割ごとの標準偏差 {tol:.3f} を超える"
    chosen_hp = {"lr": float(chosen["lr"]), "weight_decay": float(chosen["weight_decay"])}
    print(f"[select] {chosen['data']} {hp_name(chosen_hp)}: {reason}")

    # 4. 最終学習（3 本）と最終確認用 2 割
    x, y, cols = datasets[chosen["data"]]
    ho = np.flatnonzero(holdout)
    finals = []
    for r, _, row in fit_seeds(x, y, dev_rows, chosen_hp, args.out, "final", chosen["data"], cols, t0):
        with torch.no_grad():
            p = r["net"](x[ho]).squeeze(1)
        row.update(holdout_mae=F.l1_loss(p, y[ho]).item(), holdout_mse=F.mse_loss(p, y[ho]).item())
        print(f"[final seed{row['seed']}] holdout MAE={row['holdout_mae']:.3f} MSE={row['holdout_mse']:.2f}", flush=True)
        finals.append(row)
    candidate = sorted(finals, key=lambda f: f["val_mae"])[len(finals) // 2]["model"]
    for f in finals:
        f["candidate"] = int(f["model"] == candidate)
    pd.DataFrame(finals).to_csv(args.out / "final_results.csv", index=False, lineterminator="\n")

    baseline = None
    if args.baseline is not None:
        if game_data is None:
            raise SystemExit("--baseline はゲームの入力と同じ列（x1..x24）のデータがあるときだけ使える")
        x_game, y_game, _ = datasets[game_data]
        base = torch.jit.load(str(args.baseline), map_location="cpu").eval()
        with torch.no_grad():
            p = base(x_game[ho]).squeeze(1)
        baseline = {"model": args.baseline.name, "data": game_data, "holdout_rows": len(ho),
                    "holdout_mae": F.l1_loss(p, y_game[ho]).item(), "holdout_mse": F.mse_loss(p, y_game[ho]).item()}
        pd.DataFrame([baseline]).to_csv(args.out / "baseline.csv", index=False, lineterminator="\n")
        print(f"[baseline] holdout MAE={baseline['holdout_mae']:.3f} MSE={baseline['holdout_mse']:.2f}")

    with (args.out / "summary.md").open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"# 学習結果（種 {args.seed}・バッチ {BATCH_SIZE}・最大 {MAX_EPOCHS} エポック・早期終了 patience {PATIENCE}）\n\n")
        f.write(f"行数 {len(y_np)}・最終確認用 {len(ho)}・候補選び用 {len(dev_rows)}（{N_FOLDS} 分割）\n\n")
        f.write("## 交差検証（MAE 平均 ± 標準偏差 / MSE 平均）\n\n| データ | 候補 | MAE | MSE | ベストエポック平均 | 発散 |\n|---|---|---|---|---|---|\n")
        for r in summary.itertuples():
            f.write(f"| {r.data} | {hp_name(r._asdict())} | {r.mae_mean:.3f} ± {r.mae_std:.3f} | {r.mse_mean:.1f} | {r.best_epoch_mean:.0f} | {int(r.diverged)} |\n")
        f.write(f"\n**選んだ組: {chosen['data']} {hp_name(chosen_hp)}** — {reason}\n\n")
        f.write("## 最終学習 3 本（最終確認用 2 割）\n\n| モデル | 検証 MAE | 最終確認 MAE | 最終確認 MSE | ベストエポック | 採用候補 |\n|---|---|---|---|---|---|\n")
        for r in finals:
            f.write(f"| {r['model']} | {r['val_mae']:.3f} | {r['holdout_mae']:.3f} | {r['holdout_mse']:.1f} | {r['best_epoch']} | {'*' if r['candidate'] else ''} |\n")
        if baseline is not None:
            f.write(f"\n現行モデル {baseline['model']}（{baseline['data']}）の最終確認 MAE {baseline['holdout_mae']:.3f} / MSE {baseline['holdout_mse']:.1f}\n")
        f.write(f"\n所要 {time.time() - t0:.0f} 秒\n")
    (args.out / "chosen.json").write_text(json.dumps({"data": chosen["data"], "hp": chosen_hp, "candidate": candidate, "reason": reason}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] candidate={candidate}  ({time.time() - t0:.0f}s)")


def train_full(args, data_name, dataset, hp, t0):
    """全データ: 全行で種 0 / 1 / 2 の 3 本を学習する。早期終了のため種ごとに 1 割を検証用に取る。"""
    x, y, cols = dataset
    rows = np.arange(len(y))
    split = pd.DataFrame({"row": rows + 1, "y": y.numpy().astype(int)})
    results = []
    for _, val, row in fit_seeds(x, y, rows, hp, args.out, "full", data_name, cols, t0):
        split[f"val_seed{row['seed']}"] = val.astype(int)
        results.append(row)
    split.to_csv(args.out / "full_split.csv", index=False, lineterminator="\n")
    pd.DataFrame(results).to_csv(args.out / "full_results.csv", index=False, lineterminator="\n")
    with (args.out / "full_summary.md").open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"# 全データモードの学習結果（{data_name}・{hp_name(hp)}・バッチ {BATCH_SIZE}・最大 {MAX_EPOCHS} エポック・早期終了 patience {PATIENCE}）\n\n")
        f.write(f"行数 {len(rows)}・検証用は種ごとに {INNER_VAL_RATIO:.0%}（層化）\n\n")
        f.write("| モデル | 学習行 | 検証行 | 検証 MAE | 検証 MSE | ベストエポック | 発散 |\n|---|---|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['model']} | {r['train_rows']} | {r['val_rows']} | {r['val_mae']:.3f} | {r['val_mse']:.1f} | {r['best_epoch']} | {r['diverged']} |\n")
        f.write(f"\n所要 {time.time() - t0:.0f} 秒\n")
    print(f"[done] full_seed0..{FINAL_SEEDS[-1]}.pt  ({time.time() - t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser(description="難易度予測モデルの学習（cv: 交差検証で候補選び / full: 全データで 3 本学習）")
    sub = ap.add_subparsers(dest="mode", required=True)
    cv = sub.add_parser("cv", help="交差検証で候補選び")
    cv.add_argument("--data", nargs="+", type=Path, default=[Path("data.csv")], help="比べる入力ファイル（行順が同じもの）")
    cv.add_argument("--out", type=Path, required=True)
    cv.add_argument("--seed", type=int, default=0, help="分割と候補選びの学習に使う種（最終学習の種は 0,1,2）")
    cv.add_argument("--baseline", type=Path, default=None, help="比較に出す現行モデル（TorchScript）。ゲームの入力と同じ列のデータで測る")
    full = sub.add_parser("full", help="全データで 3 本学習")
    full.add_argument("--chosen", type=Path, required=True, help="cv が書いた chosen.json（データとハイパーパラメータをここから読む）")
    full.add_argument("--out", type=Path, required=True)
    full.add_argument("--data", type=Path, default=None, help="chosen.json のデータの代わりに使う入力ファイル")
    full.add_argument("--lr", type=float, default=None, help="chosen.json の学習率を上書き")
    full.add_argument("--weight-decay", type=float, default=None, help="chosen.json の重み減衰を上書き")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if args.mode == "cv":
        train_cv(args, {p.name: load_data(p) for p in args.data}, t0)
    else:
        chosen = json.loads(args.chosen.read_text(encoding="utf-8"))
        data = args.data if args.data is not None else Path(chosen["data"])
        hp = {"lr": chosen["hp"]["lr"] if args.lr is None else args.lr,
              "weight_decay": chosen["hp"]["weight_decay"] if args.weight_decay is None else args.weight_decay}
        train_full(args, data.name, load_data(data), hp, t0)


if __name__ == "__main__":
    main()
