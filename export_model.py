"""学習した重み（trainer.py の final_seed<N>.pt / full_seed<N>.pt）を C++（NPADP_pred / libtorch）が読める TorchScript に書き出す。

--weights に渡した重み（1 つでも複数でも）の出力を平均する平均モデル（model.Ensemble）として書き出す。書き出しの形は常に 1 つ。
標準化はモデルの中にあるので、書き出したモデルにはレーダーの生の整数をそのまま入れる。
書き出す前後で同じ入力に対する出力が一致することを確かめ、--data があればその先頭行を試し推論する。

使い方:
  uv run python export_model.py --weights retrain_2026-09-10/cv/final_seed2.pt --out retrain_2026-09-10/cv/model_candidate.pt --data data.csv
  uv run python export_model.py --weights retrain_2026-09-10/full/full_seed0.pt retrain_2026-09-10/full/full_seed1.pt retrain_2026-09-10/full/full_seed2.pt \
      --out retrain_2026-09-10/full/model_ensemble.pt --data data.csv
"""
import argparse
from pathlib import Path

import torch

from model import Ensemble, Net
from trainer import load_data


def load_net(weights: Path) -> Net:
    ckpt = torch.load(weights, map_location="cpu")
    net = Net(input_size=ckpt["input_size"])
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, nargs="+", required=True, help="平均する重み（1 つ以上）")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=None, help="試し推論に使う CSV（先頭行）")
    args = ap.parse_args()

    nets = [load_net(w) for w in args.weights]
    input_size = nets[0].input_size
    if any(n.input_size != input_size for n in nets):
        raise SystemExit("入力数が違う重みは平均できない: " + ", ".join(str(n.input_size) for n in nets))
    model = Ensemble(nets).eval()
    torch.jit.save(torch.jit.script(model), str(args.out))

    reloaded = torch.jit.load(str(args.out), map_location="cpu").eval()
    if args.data is not None:
        x, y, _ = load_data(args.data)
        x = x[:1]
        y = y[0].item()
    else:
        x = torch.zeros(1, input_size)
        y = float("nan")
    with torch.no_grad():
        a = model(x).item()
        b = reloaded(x).item()
    if abs(a - b) > 1e-5:
        raise SystemExit(f"書き出し前後で出力が違う: {a} vs {b}")
    print(f"平均モデル（{len(nets)} 本）  入力数 {input_size}  難易度: {b:.2f}% (ラベル {y})  → {args.out}")


if __name__ == "__main__":
    main()
