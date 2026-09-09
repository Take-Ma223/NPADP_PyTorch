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

## 学習（2026-09-10 に手順を改めた。旧手順は git 履歴の trainer.py / model.py）

  uv run python trainer.py --data data.csv data_4mode.csv --out retrain_YYYY-MM-DD --baseline model.pt.bak
  → 1. 最終確認用に 2 割を先に取り分ける（ラベルの帯で層化・種固定）
    2. 残り 8 割で 5 分割交差検証（Adam・バッチ 16・早期終了 patience 100・lr {1e-3,3e-4} × weight_decay {0,1e-4,1e-3}）
    3. データ × 候補 から 1 組を選ぶ（平均 MAE 最小。差が分割ごとの標準偏差以内なら 24 入力 data.csv を採る）
    4. 選んだ組を 8 割で 3 本学習（種 0,1,2）し、最終確認用 2 割で 1 回だけ測る。検証 MAE が中央の 1 本が採用候補
    出力は --out の下（cv_results.csv / cv_summary.csv / final_results.csv / final_seed<N>.pt / summary.md）
  - モデル（model.py）は全結合 入力→20→16→8→1・ReLU。入力の標準化（学習用の行の平均 0・標準偏差 1）を
    モデルの中に持つので、ゲーム側はレーダーの生の整数をそのまま送ればよい
  - 乱数の種は --seed（既定 0）で固定。同じデータ・同じ環境なら再現する
  - Lightning は使わなくなった（pyproject.toml の依存には残っている）

C++ 用モデル出力:
  uv run python pred.py --weights retrain_YYYY-MM-DD/final_seed1.pt --out retrain_YYYY-MM-DD/model_candidate.pt --data data.csv
  → TorchScript を書き出し、書き出し前後で出力が一致することを確かめる。
    model.pt は C++ 実装 NPADP_pred（libtorch 2.0.0）が読み込む。torch 2.13 の torch.jit.script 出力を
    libtorch 2.0.0 が読めることは 2026-09-10 に NPADP_pred.exe 単体で確認済み

現行モデル（2024-10-27 学習・ゲームに配置中）: weight.ckpt（旧 Lightning 形式）= model.pt.bak（TorchScript）

## 既知の将来課題

- PyTorch 自体が TorchScript を非推奨化し torch.export へ移行中。C++ 側（NPADP_pred / libtorch）の
  読み込み方式とセットで、いずれ torch.export ベースへの移行が必要。

## 旧環境の記録（参考）

旧 Anaconda 環境（NPADP・Python 3.10 / pytorch 1.12.1 cpu / pytorch-lightning 1.9.3）
のパッケージ一覧は git 履歴のこのファイル参照。
