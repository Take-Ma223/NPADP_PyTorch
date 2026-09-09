"""学習した重み（trainer.py の final_seed<N>.pt）を C++（NPADP_pred / libtorch）が読める TorchScript に書き出す。

標準化はモデルの中にあるので、書き出したモデルにはレーダーの生の整数をそのまま入れる。
書き出す前後で同じ入力に対する出力が一致することを確かめ、--data があればその先頭行を試し推論する。

使い方:
  uv run python pred.py --weights retrain_2026-09-10/final_seed1.pt --out retrain_2026-09-10/model_candidate.pt --data data.csv
"""
import argparse
from pathlib import Path

import pandas as pd
import torch

from model import Net


def load_net(weights: Path) -> Net:
    ckpt = torch.load(weights, map_location="cpu")
    net = Net(input_size=ckpt["input_size"])
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=None, help="試し推論に使う CSV（先頭行）")
    args = ap.parse_args()

    net = load_net(args.weights)
    script = torch.jit.script(net)
    torch.jit.save(script, str(args.out))

    reloaded = torch.jit.load(str(args.out), map_location="cpu").eval()
    if args.data is not None:
        df = pd.read_csv(args.data)
        x = torch.tensor(df.drop(columns="y").values[:1], dtype=torch.float32)
        y = float(df["y"].iloc[0])
    else:
        x = torch.zeros(1, net.input_size)
        y = float("nan")
    with torch.no_grad():
        a = net(x).item()
        b = reloaded(x).item()
    if abs(a - b) > 1e-5:
        raise SystemExit(f"書き出し前後で出力が違う: {a} vs {b}")
    print(f"入力数 {net.input_size}  難易度: {b:.2f}% (ラベル {y})  → {args.out}")


if __name__ == "__main__":
    main()
