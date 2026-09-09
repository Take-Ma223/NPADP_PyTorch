"""難易度予測モデルの学習。--mode で 2 つの手順を切り替える。

交差検証モード（--mode cv・既定）: 手順の良し悪しを比べて採用候補を決める
  1. 最終確認用に全体の 2 割を先に取り分ける（ラベルの帯で層化・種固定）。候補選びの間は一切使わない
  2. 残り 8 割で 5 分割交差検証（層化）。指標は MAE と MSE。分割ごとの値と平均・標準偏差を出す
  3. 入力（データファイル）× ハイパーパラメータ候補 のすべてを交差検証し、1 組を選ぶ:
       交差検証の平均 MAE が最小の組。ただし最小の組が 24 入力（data.csv）でないとき、
       24 入力の最良との差が「分割ごとの MAE の標準偏差（両者の大きい方）」以内なら 24 入力を採る
  4. 選んだ 1 組を残り 8 割で学習（種を 3 つ変えて 3 本。早期終了のために 8 割の中から 1 割を検証用に取る）し、
     最終確認用 2 割で 1 回だけ測る。3 本のうち検証 MAE が中央のものを採用候補にする

全データモード（--mode full）: 決まった手順で最終版のモデルを作る
  全行を使い、--lr / --weight-decay の 1 組で種 0 / 1 / 2 の 3 本を学習する（最終確認用の取り分けはしない）。
  早期終了のために種ごとに全体から 1 割を検証用に取る（層化・種ごとに違う分け方）。
  3 本は pred.py で平均モデル（Ensemble）1 つの TorchScript に書き出す

入力ファイルは行順が同じ前提（同じ譜面が同じ行）。分割はラベルと種だけで決まるので、どのファイルでも同じ行が同じ分割になる。

使い方:
  uv run python trainer.py --data data.csv data_4mode.csv --out retrain_2026-09-10 --baseline model.pt.bak
  uv run python trainer.py --mode full --data data.csv --out retrain_2026-09-10
出力（--out の下）:
  交差検証モード:
    split.csv          行ごとの分割（holdout / fold）
    cv_results.csv     交差検証の全結果（データ × 候補 × 分割）
    cv_summary.csv     データ × 候補ごとの平均・標準偏差
    final_results.csv  採用候補 3 本（検証 MAE・最終確認用 2 割の MAE/MSE）と現行モデル
    final_seed<N>.pt   3 本の重み（state_dict）。pred.py で TorchScript にする
    summary.md         人が読む要約
  全データモード:
    full_split.csv     行ごとの検証用の印（種ごと）
    full_results.csv   3 本の検証 MAE / MSE・ベストエポック
    full_seed<N>.pt    3 本の重み（state_dict）。pred.py で平均モデルにする
    full_summary.md    人が読む要約
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

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


def fit(x_tr, y_tr, x_val, y_val, hp, seed, input_size):
    """1 回の学習。検証 MSE が最小だった時点の重みを返す。"""
    torch.manual_seed(seed)
    net = Net(input_size=input_size)
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
    return {
        "net": net, "best_epoch": best["epoch"], "epochs_run": len(tl),
        "val_mse": best["val_mse"], "val_mae": (pred - y_val).abs().mean().item(),
        "train_loss_min": float(tl.min()), "train_loss_last": float(tl[-1]), "diverged": diverged,
    }


def load_data(path: Path):
    df = pd.read_csv(path)
    x = torch.tensor(df.drop(columns="y").values, dtype=torch.float32)
    y = torch.tensor(df["y"].values, dtype=torch.float32)
    return x, y, [c for c in df.columns if c != "y"]


def hp_name(hp):
    return f"lr={hp['lr']:g},wd={hp['weight_decay']:g}"


def save_weights(path: Path, net: Net, cols, data_name: str, hp, seed: int, best_epoch: int):
    """pred.py が読む形式で重みを保存する。"""
    torch.save({"state_dict": net.state_dict(), "input_size": net.input_size, "columns": cols, "data": data_name,
                "hp": hp, "seed": seed, "best_epoch": best_epoch}, path)


def train_full(args, datasets, t0):
    """全データモード: 全行で種 0 / 1 / 2 の 3 本を学習する。早期終了のため種ごとに 1 割を検証用に取る。"""
    if len(datasets) != 1:
        raise SystemExit("全データモードは --data を 1 つだけ指定する")
    name, (x, y, cols) = next(iter(datasets.items()))
    y_np = y.numpy()
    hp = {"lr": args.lr, "weight_decay": args.weight_decay}
    split = pd.DataFrame({"row": np.arange(1, len(y_np) + 1), "y": y_np.astype(int)})
    results = []
    for seed in FINAL_SEEDS:
        val = stratified_split(y_np, INNER_VAL_RATIO, np.random.default_rng(1000 + seed))
        split[f"val_seed{seed}"] = val.astype(int)
        tr = np.flatnonzero(~val)
        va = np.flatnonzero(val)
        r = fit(x[tr], y[tr], x[va], y[va], hp, seed, x.shape[1])
        save_weights(args.out / f"full_seed{seed}.pt", r["net"], cols, name, hp, seed, r["best_epoch"])
        results.append({"model": f"full_seed{seed}", "data": name, "hp": hp_name(hp), "seed": seed, "train_rows": len(tr), "val_rows": len(va),
                        "best_epoch": r["best_epoch"], "epochs_run": r["epochs_run"], "diverged": int(r["diverged"]),
                        "val_mae": r["val_mae"], "val_mse": r["val_mse"]})
        print(f"[full] seed{seed}: train={len(tr)} val={len(va)} val MAE={r['val_mae']:.3f} MSE={r['val_mse']:.2f} "
              f"best_epoch={r['best_epoch']} run={r['epochs_run']} diverged={r['diverged']}  ({time.time() - t0:.0f}s)", flush=True)
    split.to_csv(args.out / "full_split.csv", index=False, lineterminator="\n")
    pd.DataFrame(results).to_csv(args.out / "full_results.csv", index=False, lineterminator="\n")
    with (args.out / "full_summary.md").open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"# 全データモードの学習結果（{name}・{hp_name(hp)}・バッチ {BATCH_SIZE}・最大 {MAX_EPOCHS} エポック・早期終了 patience {PATIENCE}）\n\n")
        f.write(f"行数 {len(y_np)}・検証用は種ごとに {INNER_VAL_RATIO:.0%}（層化）\n\n")
        f.write("| モデル | 学習行 | 検証行 | 検証 MAE | 検証 MSE | ベストエポック | 発散 |\n|---|---|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['model']} | {r['train_rows']} | {r['val_rows']} | {r['val_mae']:.3f} | {r['val_mse']:.1f} | {r['best_epoch']} | {r['diverged']} |\n")
        f.write(f"\n所要 {time.time() - t0:.0f} 秒\n")
    print(f"[done] full_seed0..{FINAL_SEEDS[-1]}.pt  ({time.time() - t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("cv", "full"), default="cv", help="cv: 交差検証で候補選び（既定） / full: 全データで 3 本学習")
    ap.add_argument("--data", nargs="+", type=Path, default=[Path("data.csv")])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0, help="分割と候補選びの学習に使う種（最終学習の種は 0,1,2）")
    ap.add_argument("--baseline", type=Path, default=None, help="比較に出す現行モデル（TorchScript）。data.csv があるときだけ使う")
    ap.add_argument("--lr", type=float, default=3e-4, help="全データモードの学習率（既定は 2026-09-10 の交差検証で選んだ値）")
    ap.add_argument("--weight-decay", type=float, default=1e-4, help="全データモードの重み減衰（既定は 2026-09-10 の交差検証で選んだ値）")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    datasets = {p.name: load_data(p) for p in args.data}
    if args.mode == "full":
        train_full(args, datasets, t0)
        return
    ys = [y for _, y, _ in datasets.values()]
    for y in ys[1:]:
        if not torch.equal(y, ys[0]):
            raise SystemExit("データファイル間で y（行順）が違う")
    y_np = ys[0].numpy()

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
    cv_rows = []
    for name, (x, y, cols) in datasets.items():
        for hp in HP_GRID:
            for k in range(N_FOLDS):
                tr = dev_rows[fold[dev_rows] != k]
                va = dev_rows[fold[dev_rows] == k]
                r = fit(x[tr], y[tr], x[va], y[va], hp, args.seed, x.shape[1])
                cv_rows.append({"data": name, "hp": hp_name(hp), "lr": hp["lr"], "weight_decay": hp["weight_decay"], "fold": k,
                                "val_mae": r["val_mae"], "val_mse": r["val_mse"], "best_epoch": r["best_epoch"],
                                "epochs_run": r["epochs_run"], "diverged": int(r["diverged"])})
                print(f"[cv] {name} {hp_name(hp)} fold{k}: MAE={r['val_mae']:.3f} MSE={r['val_mse']:.2f} "
                      f"best_epoch={r['best_epoch']} run={r['epochs_run']} diverged={r['diverged']}  ({time.time() - t0:.0f}s)", flush=True)
    cv = pd.DataFrame(cv_rows)
    cv.to_csv(args.out / "cv_results.csv", index=False, lineterminator="\n")
    summary = cv.groupby(["data", "hp"], sort=False).agg(
        mae_mean=("val_mae", "mean"), mae_std=("val_mae", "std"), mse_mean=("val_mse", "mean"), mse_std=("val_mse", "std"),
        best_epoch_mean=("best_epoch", "mean"), diverged=("diverged", "sum"), lr=("lr", "first"), weight_decay=("weight_decay", "first")).reset_index()
    summary.to_csv(args.out / "cv_summary.csv", index=False, lineterminator="\n")

    # 選ぶ
    best_all = summary.loc[summary["mae_mean"].idxmin()]
    chosen = best_all
    reason = "交差検証の平均 MAE が最小"
    if best_all["data"] != "data.csv" and "data.csv" in datasets:
        s24 = summary[summary["data"] == "data.csv"]
        best_24 = s24.loc[s24["mae_mean"].idxmin()]
        tol = max(best_all["mae_std"], best_24["mae_std"])
        diff = best_24["mae_mean"] - best_all["mae_mean"]
        if diff <= tol:
            chosen = best_24
            reason = (f"最小は {best_all['data']} {best_all['hp']}（{best_all['mae_mean']:.3f}）だが、24 入力の最良（{best_24['mae_mean']:.3f}）との差 {diff:.3f} が"
                      f"分割ごとの標準偏差 {tol:.3f} 以内なので 24 入力を採る")
        else:
            reason = f"24 入力の最良（{best_24['mae_mean']:.3f}）との差 {diff:.3f} が分割ごとの標準偏差 {tol:.3f} を超える"
    chosen_hp = {"lr": float(chosen["lr"]), "weight_decay": float(chosen["weight_decay"])}
    print(f"[select] {chosen['data']} {chosen['hp']}: {reason}")

    # 4. 最終学習（3 本）と最終確認用 2 割
    x, y, cols = datasets[chosen["data"]]
    ho = np.flatnonzero(holdout)
    finals = []
    for seed in FINAL_SEEDS:
        inner = stratified_split(y_np[dev_rows], INNER_VAL_RATIO, np.random.default_rng(1000 + seed))
        tr = dev_rows[~inner]
        va = dev_rows[inner]
        r = fit(x[tr], y[tr], x[va], y[va], chosen_hp, seed, x.shape[1])
        with torch.no_grad():
            p = r["net"](x[ho]).squeeze(1)
        ho_mae = (p - y[ho]).abs().mean().item()
        ho_mse = F.mse_loss(p, y[ho]).item()
        save_weights(args.out / f"final_seed{seed}.pt", r["net"], cols, chosen["data"], chosen_hp, seed, r["best_epoch"])
        finals.append({"model": f"final_seed{seed}", "data": chosen["data"], "hp": chosen["hp"], "seed": seed,
                       "train_rows": len(tr), "inner_val_rows": len(va), "best_epoch": r["best_epoch"], "diverged": int(r["diverged"]),
                       "inner_val_mae": r["val_mae"], "inner_val_mse": r["val_mse"], "holdout_mae": ho_mae, "holdout_mse": ho_mse})
        print(f"[final] seed{seed}: inner_val MAE={r['val_mae']:.3f} holdout MAE={ho_mae:.3f} MSE={ho_mse:.2f} best_epoch={r['best_epoch']} diverged={r['diverged']}", flush=True)

    order = sorted(finals, key=lambda f: f["inner_val_mae"])
    candidate = order[len(order) // 2]["model"]
    for f in finals:
        f["candidate"] = int(f["model"] == candidate)

    if args.baseline is not None and "data.csv" in datasets:
        x24, y24, _ = datasets["data.csv"]
        base = torch.jit.load(str(args.baseline), map_location="cpu").eval()
        with torch.no_grad():
            p = base(x24[ho]).squeeze(1)
        finals.append({"model": f"baseline({args.baseline.name})", "data": "data.csv", "hp": "", "seed": "", "train_rows": "", "inner_val_rows": "",
                       "best_epoch": "", "diverged": "", "inner_val_mae": "", "inner_val_mse": "",
                       "holdout_mae": (p - y24[ho]).abs().mean().item(), "holdout_mse": F.mse_loss(p, y24[ho]).item(), "candidate": 0})
        print(f"[baseline] holdout MAE={finals[-1]['holdout_mae']:.3f} MSE={finals[-1]['holdout_mse']:.2f}")
    pd.DataFrame(finals).to_csv(args.out / "final_results.csv", index=False, lineterminator="\n")

    with (args.out / "summary.md").open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"# 学習結果（種 {args.seed}・バッチ {BATCH_SIZE}・最大 {MAX_EPOCHS} エポック・早期終了 patience {PATIENCE}）\n\n")
        f.write(f"行数 {len(y_np)}・最終確認用 {int(holdout.sum())}・候補選び用 {len(dev_rows)}（{N_FOLDS} 分割）\n\n")
        f.write("## 交差検証（MAE 平均 ± 標準偏差 / MSE 平均）\n\n| データ | 候補 | MAE | MSE | ベストエポック平均 | 発散 |\n|---|---|---|---|---|---|\n")
        for r in summary.itertuples():
            f.write(f"| {r.data} | {r.hp} | {r.mae_mean:.3f} ± {r.mae_std:.3f} | {r.mse_mean:.1f} | {r.best_epoch_mean:.0f} | {int(r.diverged)} |\n")
        f.write(f"\n**選んだ組: {chosen['data']} {chosen['hp']}** — {reason}\n\n")
        f.write("## 最終学習 3 本と現行モデル（最終確認用 2 割）\n\n| モデル | 検証 MAE | 最終確認 MAE | 最終確認 MSE | ベストエポック | 採用候補 |\n|---|---|---|---|---|---|\n")
        for r in finals:
            iv = f"{r['inner_val_mae']:.3f}" if r["inner_val_mae"] != "" else "-"
            f.write(f"| {r['model']} | {iv} | {r['holdout_mae']:.3f} | {r['holdout_mse']:.1f} | {r['best_epoch']} | {'*' if r['candidate'] else ''} |\n")
        f.write(f"\n所要 {time.time() - t0:.0f} 秒\n")
    (args.out / "chosen.json").write_text(json.dumps({"data": chosen["data"], "hp": chosen_hp, "candidate": candidate, "reason": reason}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] candidate={candidate}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
