"""学習した重み（trainer.py の final_seed<N>.pt / full_seed<N>.pt）を C++（NPADP_pred / libtorch）が読める TorchScript に書き出す。

--weights を 1 つ渡せばその 1 本を、複数渡せばそれらの出力を平均する平均モデル（model.Ensemble）を書き出す。
標準化はモデルの中にあるので、書き出したモデルにはレーダーの生の整数をそのまま入れる。
書き出す前後で同じ入力に対する出力が一致することを確かめ、--data があればその先頭行を試し推論する。

使い方:
  uv run python pred.py --weights retrain_2026-09-10/final_seed2.pt --out retrain_2026-09-10/model_candidate.pt --data data.csv
  uv run python pred.py --weights retrain_2026-09-10/full_seed0.pt retrain_2026-09-10/full_seed1.pt retrain_2026-09-10/full_seed2.pt \
      --out retrain_2026-09-10/model_ensemble.pt --data data.csv
"""
import argparse
from pathlib import Path

import pandas as pd
import torch

from model import Ensemble, Net


def load_net(weights: Path) -> Net:
    ckpt = torch.load(weights, map_location="cpu")
    net = Net(input_size=ckpt["input_size"])
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, nargs="+", required=True, help="1 つなら単体、複数なら平均モデル")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=None, help="試し推論に使う CSV（先頭行）")
    args = ap.parse_args()

    nets = [load_net(w) for w in args.weights]
    input_size = nets[0].input_size
    if any(n.input_size != input_size for n in nets):
        raise SystemExit("入力数が違う重みは平均できない: " + ", ".join(str(n.input_size) for n in nets))
    model = nets[0] if len(nets) == 1 else Ensemble(nets).eval()
    script = torch.jit.script(model)
    torch.jit.save(script, str(args.out))

    reloaded = torch.jit.load(str(args.out), map_location="cpu").eval()
    if args.data is not None:
        df = pd.read_csv(args.data)
        x = torch.tensor(df.drop(columns="y").values[:1], dtype=torch.float32)
        y = float(df["y"].iloc[0])
    else:
        x = torch.zeros(1, input_size)
        y = float("nan")
    with torch.no_grad():
        a = model(x).item()
        b = reloaded(x).item()
    if abs(a - b) > 1e-5:
        raise SystemExit(f"書き出し前後で出力が違う: {a} vs {b}")
    kind = "単体" if len(nets) == 1 else f"平均モデル（{len(nets)} 本）"
    print(f"{kind}  入力数 {input_size}  難易度: {b:.2f}% (ラベル {y})  → {args.out}")


if __name__ == "__main__":
    main()
