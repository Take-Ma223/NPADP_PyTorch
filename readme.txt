nature prhysm 譜面難易度予測ネットワーク（NPADP）PyTorch 実装

## 環境（2026-07-25 に uv へ移行・旧 Anaconda 環境は廃止）

依存は pyproject.toml で管理。Python 3.12（.python-version で固定）+ torch CPU 版 + Lightning。

初回セットアップ:
  1. uv をインストール（winget install astral-sh.uv）
  2. このフォルダで: uv sync

学習:
  uv run python trainer.py
  → data.csv（24特徴量+y）を 6:2:2 に分割して最大 1000 epoch 学習。
    ベストは logs/ に epoch=NN-val_loss_epoch=X.ckpt として保存される。
  ※ シード固定（manual_seed(0)）のため同一データなら再現する
    （環境移行検証: torch1.12/PL1.9/Py3.10 → torch2.13/L2.6/Py3.12 で
     ベスト epoch・val_loss が完全一致することを確認済み）

推論と C++ 用モデル出力:
  uv run python pred.py
  → weight.ckpt を読み pred.csv の 1 行を推論。TorchScript を model.pt へ保存
    （model.pt は C++ 実装 NPADP_pred が読み込む）

## 既知の将来課題

- Lightning の to_torchscript は v2.8 で削除予定（PyTorch 自体が TorchScript を
  非推奨化し torch.export へ移行中）。C++ 側（NPADP_pred / libtorch）の読み込み
  方式とセットで、いずれ torch.export ベースへの移行が必要。

## 旧環境の記録（参考）

旧 Anaconda 環境（NPADP・Python 3.10 / pytorch 1.12.1 cpu / pytorch-lightning 1.9.3）
のパッケージ一覧は git 履歴のこのファイル参照。
