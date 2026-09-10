"""学習用譜面フォルダ `学習用songs` を生成する。

構成:
  学習用songs/official/<曲>/<1..4>.nps  公式 437 譜面（ゲームの songs/official）＋ 旧 backup から戻す 7 譜面
  学習用songs/user/<曲>/<1..4>.nps      旧 backup の user から、除外リストを除いた譜面
  学習用songs/manifest.csv               各譜面の出どころ・元の #LEVEL・学習ラベル

`.nps` だけをコピーする（レーダー計測ツール np_radar_measure は音源・ジャケット無しで動く）。
コピーの `#LEVEL` 行だけを labels_override.csv の値に書き換え、ほかの行はバイト単位で元と同じにする
（元のエンコーディング UTF-16 LE BOM と改行 CRLF をそのまま保つ）。

何度実行しても同じ結果になる。出力先が既にある場合は、このスクリプトが作ったもの（manifest.csv がある）に限り作り直す。

使い方:
  uv run python build_training_songs.py
  （既定の場所を変えるときは --game-official / --backup / --out）
"""
import argparse
import csv
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_GAME_OFFICIAL = Path(r"F:\nature_prhysm\game\nature_prhysm\songs\official")
DEFAULT_BACKUP = HERE / "学習用songs_2024-10_backup"
DEFAULT_OUT = HERE / "学習用songs"
LABELS = HERE / "labels_override.csv"

# 今の公式には無いが学習に戻す譜面（旧 backup の official から）
RESTORE_FROM_BACKUP_OFFICIAL = [
    ("MAD ACCELERATOR DESTRUCTION", 4),
    ("lost", 1), ("lost", 2), ("lost", 3),
    ("青空のドラムンベース", 1), ("青空のドラムンベース", 2), ("青空のドラムンベース", 3),
]

# 旧 backup の user から学習に入れない譜面
EXCLUDE_USER = [
    ("test", 1), ("test", 2), ("test", 3),
    ("タイムライン・ディスコ", 1), ("タイムライン・ディスコ", 2), ("タイムライン・ディスコ", 3),
]

BOM_UTF16LE = b"\xff\xfe"


def read_nps(path: Path):
    """UTF-16 LE BOM の .nps を行リスト（改行込み）で返す。往復でバイト列が元と一致することを確かめる。"""
    raw = path.read_bytes()
    if not raw.startswith(BOM_UTF16LE):
        raise RuntimeError(f"UTF-16 LE BOM ではない: {path}")
    text = raw[2:].decode("utf-16-le")
    if BOM_UTF16LE + text.encode("utf-16-le") != raw:
        raise RuntimeError(f"エンコーディングの往復が一致しない: {path}")
    return text.splitlines(keepends=True)


def level_of(lines):
    for line in lines:
        if line.startswith("#LEVEL:"):
            return int(line[len("#LEVEL:"):].strip())
    return None


def write_nps(dst: Path, lines, new_level):
    """#LEVEL 行だけ差し替えて書く（new_level が None なら元のまま）。書いたあとに他の行が同じことを検証する。"""
    out = []
    replaced = 0
    for line in lines:
        if new_level is not None and line.startswith("#LEVEL:"):
            eol = line[len(line.rstrip("\r\n")):]
            out.append(f"#LEVEL:{new_level}{eol}")
            replaced += 1
        else:
            out.append(line)
    if new_level is not None and replaced != 1:
        raise RuntimeError(f"#LEVEL 行が {replaced} 行ある: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(BOM_UTF16LE + "".join(out).encode("utf-16-le"))
    check = read_nps(dst)
    if len(check) != len(lines):
        raise RuntimeError(f"行数が変わった: {dst}")
    for a, b in zip(check, lines):
        if a != b and not (a.startswith("#LEVEL:") and b.startswith("#LEVEL:")):
            raise RuntimeError(f"#LEVEL 以外の行が変わった: {dst}")


def load_overrides(path: Path):
    table = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            table[(row["song"], int(row["chart"]))] = int(row["label"])
    return table


def list_charts(root: Path):
    """root/<曲>/<1..4>.nps を (曲, slot, path) で列挙する。"""
    charts = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        for slot in range(1, 5):
            p = folder / f"{slot}.nps"
            if p.is_file():
                charts.append((folder.name, slot, p))
    return charts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-official", type=Path, default=DEFAULT_GAME_OFFICIAL)
    ap.add_argument("--backup", type=Path, default=DEFAULT_BACKUP)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    for p in (args.game_official, args.backup / "official", args.backup / "user", LABELS):
        if not p.exists():
            sys.exit(f"見つからない: {p}")

    if args.out.exists():
        if not (args.out / "manifest.csv").is_file():
            sys.exit(f"{args.out} は このスクリプトの生成物ではない（manifest.csv が無い）。名前を変えて退避してから実行する")
        shutil.rmtree(args.out)

    overrides = load_overrides(LABELS)

    def add(source, folder, slot, src_path, out_root, apply_overrides):
        """1 譜面を写して manifest の行を返す。apply_overrides のときだけ labels_override.csv の値で #LEVEL を書き換える。"""
        lines = read_nps(src_path)
        orig = level_of(lines)
        new = overrides.get((folder, slot)) if apply_overrides else None
        write_nps(out_root / folder / f"{slot}.nps", lines, new)
        return {
            "source": source, "folder": folder, "slot": slot, "src_path": str(src_path),
            "original_level": orig, "label": new if new is not None else orig,
            "override": 1 if new is not None else 0,
        }

    # official: 今の公式
    rows = [add("official", folder, slot, p, args.out / "official", apply_overrides=True) for folder, slot, p in list_charts(args.game_official)]
    n_game = len(rows)

    # official: 戻す 7 譜面
    for folder, slot in RESTORE_FROM_BACKUP_OFFICIAL:
        if any(r["folder"] == folder and r["slot"] == slot for r in rows):
            sys.exit(f"戻す譜面が今の公式にもある: {folder}/{slot}")
        p = args.backup / "official" / folder / f"{slot}.nps"
        if not p.is_file():
            sys.exit(f"戻す譜面が backup に無い: {p}")
        rows.append(add("official(restored)", folder, slot, p, args.out / "official", apply_overrides=True))

    # user: 除外以外
    excluded = set(EXCLUDE_USER)
    for folder, slot, p in list_charts(args.backup / "user"):
        if (folder, slot) in excluded:
            excluded.discard((folder, slot))
            continue
        rows.append(add("user", folder, slot, p, args.out / "user", apply_overrides=False))
    if excluded:
        sys.exit(f"除外リストの譜面が backup に無い: {sorted(excluded)}")

    used = {(r["folder"], r["slot"]) for r in rows if r["override"]}
    unused = set(overrides) - used
    if unused:
        sys.exit(f"labels_override.csv の譜面が見つからない: {sorted(unused)}")

    with (args.out / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_user = sum(r["source"] == "user" for r in rows)
    print(f"official: {n_game} (game) + {len(RESTORE_FROM_BACKUP_OFFICIAL)} (restored) = {n_game + len(RESTORE_FROM_BACKUP_OFFICIAL)}")
    print(f"user: {n_user}")
    print(f"total: {len(rows)}  overrides applied: {len(used)}/{len(overrides)}")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
