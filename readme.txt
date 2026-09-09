nature prhysm 譜面難易度予測ネットワーク（NPADP）PyTorch 実装

## 環境（2026-07-25 に uv へ移行・旧 Anaconda 環境は廃止）

依存は pyproject.toml で管理。Python 3.12（.python-version で固定）+ torch CPU 版 + Lightning。

初回セットアップ:
  1. uv をインストール（winget install astral-sh.uv）
  2. このフォルダで: uv sync

## 学習データの作り方（2026-09-10 から。生成物 data.csv / data_4mode.csv / 学習用songs は git 管理外）

1. 学習用の譜面フォルダを作る:
     uv run python build_training_songs.py
   → 学習用songs/official（今の公式 songs/official 437 譜面 + 旧 backup から戻す 7 譜面）と
     学習用songs/user（旧 backup の user から test と タイムライン・ディスコ を除いた 64 譜面）に
     .nps だけを写し、labels_override.csv の譜面は #LEVEL をその値に書き換える
     （それ以外の行はバイト単位で元のまま）。manifest.csv に出どころとラベルを記録する。
   - 元になる旧フォルダは 学習用songs_2024-10_backup（2024-10 の学習用コピー・消さない）
   - ラベルの決め方: 公式の #LEVEL が基本。labels_override.csv（song, chart, label）で上書き。
     カスタム譜面（user）は旧コピーの #LEVEL のまま
2. レーダー値を測って CSV にする:
     uv run python make_dataset.py --tool <np_radar_measure.exe>
   → 計測ツール（本体リポジトリの tools/radar_measure。docs/design-notes/radar-measurement-tool.md）で
     全譜面を 4 モード計測し、data.csv（24 入力・x1..x24,y）と data_4mode.csv（4 モード入力・
     列名 sc_/sr_/lc_/lr_ + 軸名）、data_index.csv（行 → 譜面）を書く。計測の作業場所は measure_work/。
   - ツールは .nps だけで動く（音源・ジャケット不要）。計測 CSV のパスに日本語を入れない

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
