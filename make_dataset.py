"""`学習用songs` の全譜面をレーダー計測ツールで 4 モード計測し、学習データを作る。

出力（このフォルダ直下・git 管理外）:
  data.csv         ゲームの入力 24 個 + y。列名 x1..x24（GAME_INPUT_COLS）。並びはゲームの AutoDifficultyPrediction.h が送る順
                   （STANDARD×COLORFUL の global,local,chain,unstability,streak,color → colorNotes[0..8] → localNotes[0..8]）
  data_4mode.csv   4 モード入力 + y。列名は <モード>_<軸|color_notes_i|local_notes_i>。
                   モード接頭辞: sc=standard-colorful, sr=standard-rainbow, lc=light-colorful, lr=light-rainbow。
                   各モード 7 軸 + color_notes 9 + local_notes 9 = 25 列 × 4 のうち、全譜面 0 の列は落とす（落とした列は標準出力に出す）
  data_index.csv   行 → 譜面（source, folder, slot, label）。data.csv / data_4mode.csv と同じ行順
  measure_work/    計測ツールに渡す作業フォルダと計測 CSV（radar_official.csv / radar_user.csv）

y は `学習用songs/manifest.csv` の label 列（build_training_songs.py が決めたラベル）。各コピーの `#LEVEL` と突き合わせ、食い違えば止める。

使い方:
  uv run python make_dataset.py --tool <np_radar_measure.exe のパス>
  （--songs / --work / --game-root で場所を変えられる。--skip-measure で計測 CSV を再利用）
"""
import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from build_training_songs import DEFAULT_GAME_OFFICIAL, DEFAULT_OUT, level_of, list_charts, read_nps

HERE = Path(__file__).resolve().parent
DEFAULT_SONGS = DEFAULT_OUT
DEFAULT_WORK = HERE / "measure_work"
DEFAULT_GAME_ROOT = DEFAULT_GAME_OFFICIAL.parent.parent
DEFAULT_TOOL = Path(r"F:\nature_prhysm\game\nature_prhysm_wt_n5\tools\radar_measure\bin\Release\np_radar_measure.exe")

MODES = [("sc", "standard-colorful"), ("sr", "standard-rainbow"), ("lc", "light-colorful"), ("lr", "light-rainbow")]
AXES = ["global", "local", "chain", "unstability", "streak", "rhythm", "color"]
COLOR_NOTES = [f"color_notes_{i}" for i in range(9)]
LOCAL_NOTES = [f"local_notes_{i}" for i in range(9)]

# ゲームが送る入力に対応するツール CSV の列（standard-colorful 行）と、data.csv での列名
X24_TOOL_COLS = ["disp_global", "disp_local", "disp_chain", "disp_unstability", "disp_streak", "disp_color"] + COLOR_NOTES + LOCAL_NOTES
GAME_INPUT_COLS = [f"x{i + 1}" for i in range(len(X24_TOOL_COLS))]


def load_labels(songs: Path):
    """manifest.csv のラベルを {(source フォルダ, 曲, slot): label} で返す。各コピーの #LEVEL と突き合わせ、食い違えば止める。"""
    labels = {}
    with (songs / "manifest.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            source = "user" if row["source"] == "user" else "official"
            labels[(source, row["folder"], int(row["slot"]))] = int(row["label"])
    charts = [(source, folder, slot, p) for source in ("official", "user") for folder, slot, p in list_charts(songs / source)]
    if len(charts) != len(labels) or any((s, f, sl) not in labels for s, f, sl, _ in charts):
        sys.exit(f"manifest.csv（{len(labels)} 行）と .nps（{len(charts)} 個）が対応しない")
    for source, folder, slot, p in charts:
        level = level_of(read_nps(p))
        if level != labels[(source, folder, slot)]:
            sys.exit(f"manifest.csv のラベル {labels[(source, folder, slot)]} と #LEVEL {level} が違う: {p}")
    return labels


def prepare_root(src: Path, root: Path, game_root: Path):
    """root/songs/official/<曲>/<slot>.nps へ .nps を写し、catalog / locales をゲームから写す（読み取りのみ）。"""
    if root.exists():
        shutil.rmtree(root)
    for folder, slot, p in list_charts(src):
        dst = root / "songs" / "official" / folder / f"{slot}.nps"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dst)
    for name in ("catalog", "locales"):
        if (game_root / name).is_dir():
            shutil.copytree(game_root / name, root / name)


def measure(tool: Path, root: Path, out_csv: Path):
    if any(ord(c) > 127 for c in str(out_csv)):
        sys.exit(f"計測 CSV のパスに ASCII 以外が含まれる（ツールが異常終了する）: {out_csv}")
    r = subprocess.run([str(tool), str(root), str(out_csv)], capture_output=True, text=True)
    print(f"[measure] {root.name}: rc={r.returncode} {r.stdout.strip()} {r.stderr.strip()}")
    if r.returncode != 0:
        sys.exit("計測ツールが失敗した")


def load_measured(csv_path: Path, source: str, labels):
    df = pd.read_csv(csv_path)
    df["slot"] = df["slot"].astype(int)
    df["source"] = source
    df["label"] = [labels[(source, f, s)] for f, s in zip(df["folder"], df["slot"])]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", type=Path, default=DEFAULT_TOOL)
    ap.add_argument("--songs", type=Path, default=DEFAULT_SONGS)
    ap.add_argument("--work", type=Path, default=DEFAULT_WORK)
    ap.add_argument("--game-root", type=Path, default=DEFAULT_GAME_ROOT)
    ap.add_argument("--skip-measure", action="store_true", help="measure_work の計測 CSV をそのまま使う")
    args = ap.parse_args()

    labels = load_labels(args.songs)
    args.work.mkdir(parents=True, exist_ok=True)
    frames = []
    for source in ("official", "user"):
        root = args.work / f"root_{source}"
        out_csv = args.work / f"radar_{source}.csv"
        if not args.skip_measure:
            if not args.tool.is_file():
                sys.exit(f"計測ツールが無い: {args.tool}")
            prepare_root(args.songs / source, root, args.game_root)
            measure(args.tool, root, out_csv)
        frames.append(load_measured(out_csv, source, labels))
    m = pd.concat(frames, ignore_index=True)

    # 1 譜面 = 4 行（4 モード）になっているか
    per_chart = m.groupby(["source", "folder", "slot"])["mode"].nunique()
    if (per_chart != 4).any():
        sys.exit(f"4 モード揃っていない譜面がある: {per_chart[per_chart != 4]}")
    n_charts = len(per_chart)
    if n_charts != len(labels):
        sys.exit(f"計測した譜面数 {n_charts} と .nps の数 {len(labels)} が違う")

    key = ["source", "folder", "slot"]
    sc = m[m["mode"] == "standard-colorful"].sort_values(key).reset_index(drop=True)
    index = sc[key + ["label"]].copy()
    index.insert(0, "row", range(1, len(index) + 1))
    index["chart_path"] = [f"学習用songs/{s}/{f}/{sl}.nps" for s, f, sl in zip(index["source"], index["folder"], index["slot"])]

    # data.csv
    d24 = pd.DataFrame({name: sc[c].astype(int).values for name, c in zip(GAME_INPUT_COLS, X24_TOOL_COLS)})
    d24["y"] = sc["label"].astype(int).values
    d24.to_csv(HERE / "data.csv", index=False, lineterminator="\n")

    # data_4mode.csv
    cols = {}
    for prefix, mode in MODES:
        sub = m[m["mode"] == mode].sort_values(key).reset_index(drop=True)
        assert (sub[key].values == sc[key].values).all()
        for a in AXES:
            cols[f"{prefix}_{a}"] = sub[f"disp_{a}"].astype(int).values
        for c in COLOR_NOTES + LOCAL_NOTES:
            cols[f"{prefix}_{c}"] = sub[c].astype(int).values
    d4 = pd.DataFrame(cols)
    zero_cols = [c for c in d4.columns if (d4[c] == 0).all()]
    d4 = d4.drop(columns=zero_cols)
    d4["y"] = d24["y"].values
    d4.to_csv(HERE / "data_4mode.csv", index=False, lineterminator="\n")
    index.to_csv(HERE / "data_index.csv", index=False, encoding="utf-8-sig", lineterminator="\n")

    dup = []
    names = [c for c in d4.columns if c != "y"]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if (d4[a].values == d4[b].values).all():
                dup.append((a, b))

    print(f"charts={n_charts} ({index['source'].value_counts().to_dict()})")
    print(f"data.csv: {d24.shape[0]} rows x {d24.shape[1]} cols   y>=100: {int((d24['y'] >= 100).sum())}")
    print(f"data_4mode.csv: {d4.shape[0]} rows x {d4.shape[1]} cols (dropped all-zero: {zero_cols})")
    print(f"data_4mode.csv columns: {list(d4.columns)}")
    print(f"identical column pairs in data_4mode.csv: {dup}")
    print("y distribution:", d24["y"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
