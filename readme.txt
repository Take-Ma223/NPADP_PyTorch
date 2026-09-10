nature prhysm 譜面難易度予測ネットワーク（NPADP）PyTorch 実装

## 環境（2026-07-25 に uv へ移行・旧 Anaconda 環境は廃止）

依存は pyproject.toml で管理。Python 3.12（.python-version で固定）+ torch CPU 版 + pandas + numpy。
（Lightning は 2026-09-10 に学習ループを素の PyTorch にしたので外した）

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
   → 計測ツールで全譜面を 4 モード計測し、data.csv（ゲームの入力 24 個・x1..x24,y）と data_4mode.csv（4 モード入力・
     列名 sc_/sr_/lc_/lr_ + 軸名）、data_index.csv（行 → 譜面）を書く。計測の作業場所は measure_work/。
     y は 学習用songs/manifest.csv の label 列（各 .nps の #LEVEL と突き合わせ、食い違えば止まる）
   - 計測ツールは本体リポジトリを cmake --build <build dir> --target np_radar_measure してできる exe
     （tools/radar_measure。docs/design-notes/radar-measurement-tool.md）を --tool で渡す。
     既定値（make_dataset.py の DEFAULT_TOOL）は一時的な worktree nature_prhysm_wt_n5 のビルドを指している
   - ツールは .nps だけで動く（音源・ジャケット不要）。計測 CSV のパスに日本語を入れない

## 学習（2026-09-10 に手順を改めた。旧手順は git 履歴の trainer.py / model.py）

共通: モデル（model.py の Net）は全結合 入力→20→16→8→1・ReLU。入力の標準化（学習用の行の平均 0・標準偏差 1）を
モデルの中に持つので、ゲーム側はレーダーの生の整数をそのまま送ればよい。
学習は素の PyTorch ループ（Adam・バッチ 16・最大 3000 エポック・検証 MSE で早期終了 patience 100）。
乱数の種は固定なので、同じデータ・同じ環境なら再現する。1 回の学習は CPU で 20〜50 秒。

1. 交差検証モード（手順や入力の良し悪しを比べて採用候補を決める）:
     uv run python trainer.py --mode cv --data data.csv data_4mode.csv --out retrain_YYYY-MM-DD --baseline model.pt.bak
   → 1. 最終確認用に 2 割を先に取り分ける（ラベルの帯で層化・種固定）
     2. 残り 8 割で 5 分割交差検証（lr {1e-3,3e-4} × weight_decay {0,1e-4,1e-3}）
     3. データ × 候補 から 1 組を選ぶ（平均 MAE 最小。差が分割ごとの標準偏差以内なら 24 入力 data.csv を採る）
     4. 選んだ組を 8 割で 3 本学習（種 0,1,2）し、最終確認用 2 割で 1 回だけ測る。検証 MAE が中央の 1 本が採用候補
     出力は --out の下（split.csv / cv_results.csv / cv_summary.csv / final_results.csv / final_seed<N>.pt / summary.md）
   - --mode cv が既定。--seed（既定 0）で分割と候補選びの種を変えられる
   - 2026-09-10 の結果: 24 入力・lr 3e-4・weight_decay 1e-4 を採用（4 モード入力は負け）

2. 全データモード（決まった手順で最終版を作る）:
     uv run python trainer.py --mode full --data data.csv --out retrain_YYYY-MM-DD
   → 全行で種 0,1,2 の 3 本を学習する（最終確認用の取り分けはしない。lr / weight_decay は --lr / --weight-decay、
     既定は 1 の結果の 3e-4 / 1e-4）。早期終了のための検証行は種ごとに全体から 1 割を層化で取る
     出力は --out の下（full_split.csv / full_results.csv / full_seed<N>.pt / full_summary.md）

3. C++ 用モデル出力（TorchScript）:
     uv run python pred.py --weights retrain_YYYY-MM-DD/full_seed0.pt retrain_YYYY-MM-DD/full_seed1.pt retrain_YYYY-MM-DD/full_seed2.pt \
         --out retrain_YYYY-MM-DD/model_ensemble.pt --data data.csv
   → --weights が複数なら 3 本の出力を平均する平均モデル（model.py の Ensemble）、1 つならその単体を書き出す。
     書き出し前後で出力が一致することを確かめる。
     model.pt は C++ 実装 NPADP_pred（libtorch 2.0.0）が読み込む。torch 2.13 の torch.jit.script 出力
     （単体・平均モデルとも）を libtorch 2.0.0 が読めることは 2026-09-10 に NPADP_pred.exe 単体で確認済み

ゲームへの配置: 書き出した model.pt を本体の programs/application/auto_difficulty_prediction/model/model.pt に置く
（元はバックアップする）。このフォルダの model.pt は配置中のものと同じにしておく。

モデルの履歴:
- 2024-10-27 学習（2026-09-10 時点でゲームに配置中）: weight.ckpt（旧 Lightning 形式）= model.pt.bak（TorchScript）
  = model.pt.2026-07-25.bak（同じ重みを torch 2.13 で書き出し直したもの）
- 2026-09-10 学習（差し替え候補）: retrain_2026-09-10/model_ensemble.pt（508 譜面全部・3 本平均）= このフォルダの model.pt

## 既知の将来課題

- PyTorch 自体が TorchScript を非推奨化し torch.export へ移行中。C++ 側（NPADP_pred / libtorch）の
  読み込み方式とセットで、いずれ torch.export ベースへの移行が必要。

## 旧環境の記録（参考）

旧 Anaconda 環境（NPADP・Python 3.10 / pytorch 1.12.1 cpu / pytorch-lightning 1.9.3）
のパッケージ一覧は git 履歴のこのファイル参照。
