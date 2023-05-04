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
df = pd.read_csv('pred.csv')

t = df['y']
x = df.drop('y', axis=1)

x = torch.tensor(x.values, dtype=torch.float32)
t = torch.tensor(t.values, dtype=torch.float32)

# 入力変数と目的変数をまとめて、ひとつのオブジェクト dataset に変換
dataset = torch.utils.data.TensorDataset(x, t)



CKPT_PATH = "weight.ckpt"

# インスタンス化
net = Net()

net = net.load_from_checkpoint(checkpoint_path=CKPT_PATH)

print(x.dim())

print("難易度:"+str(float(net(x)))+"%")

# モデルの保存
script = net.to_torchscript()

torch.jit.save(script, "model.pt")

