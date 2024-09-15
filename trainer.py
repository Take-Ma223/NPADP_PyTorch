import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from model import Net

# データの読み込み(df: DataFrame)
df = pd.read_csv('data.csv')

t = df['y']
x = df.drop('y', axis=1)

x = torch.tensor(x.values, dtype=torch.float32)
t = torch.tensor(t.values, dtype=torch.float32)

# 入力変数と目的変数をまとめて、ひとつのオブジェクト dataset に変換
dataset = torch.utils.data.TensorDataset(x, t)

# 各データセットのサンプル数を決定
# train : val : test = 60% : 20% : 20%
n_train = int(len(dataset) * 0.6)
n_val = int(len(dataset) * 0.2)
n_test = len(dataset) - n_train - n_val

# ランダムに分割を行うため、シードを固定して再現性を確保
torch.manual_seed(0)

# データセットの分割
train, val, test = torch.utils.data.random_split(dataset, [n_train, n_val, n_test])

model_checkpoint = pl.callbacks.ModelCheckpoint(
    "logs/",
    filename="{epoch}-{val_loss_epoch:.4f}",
    monitor="val_loss_epoch",
    mode="min",
    save_top_k=1,
    save_last=False,
)
early_stopping = pl.callbacks.EarlyStopping(
    monitor="train_loss_epoch",
    mode="min",
    patience=2000,
)



# 再現性の確保
torch.manual_seed(0)

# インスタンス化
net = Net()

# データのセット
net.setData(train, val, test)

trainer = Trainer(
    max_epochs=10000,
    callbacks=[model_checkpoint, early_stopping],
)

# 学習の実行
trainer.fit(net)

trainer.test()
