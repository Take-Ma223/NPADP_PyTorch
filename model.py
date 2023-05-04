import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


# 学習データに対する処理
class TrainNet(pl.LightningModule):

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data, self.batch_size, shuffle=True)

    def training_step(self, batch, batch_nb):
        x, t = batch
        y = self.forward(x)
        loss = self.lossfun(y, t)
        results = {'loss': loss}
        self.log("train_loss",loss,
                 prog_bar=True,  # プログレスバーに表示するか？
                 logger=True,  # 結果を保存するのか？
                 on_epoch=True,  # １epoch中の結果を累積した値を利用するのか？
                 on_step=True,  # １stepの結果を利用するのか？
        )
        return results


# 検証データに対する処理
class ValidationNet(pl.LightningModule):

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data, self.batch_size)

    def validation_step(self, batch, batch_nb):
        x, t = batch
        y = self.forward(x)
        loss = self.lossfun(y, t)
        results = {'val_loss': loss}
        return results

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        results = {'val_loss': avg_loss}
        return results


# テストデータに対する処理
class TestNet(pl.LightningModule):

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_data, self.batch_size)

    def test_step(self, batch, batch_nb):
        x, t = batch
        y = self.forward(x)
        loss = self.lossfun(y, t)
        results = {'test_loss': loss}
        return results

    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        results = {'test_loss': avg_loss}
        return results


# 学習データ、検証データ、テストデータへの処理を継承したクラス
class Net(TrainNet, ValidationNet, TestNet):

    def __init__(self, input_size=24, hidden1_size=20, hidden2_size=16, hidden3_size=8, output_size=1, batch_size=100):
        super(Net, self).__init__()
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, hidden3_size)
        self.fc4 = nn.Linear(hidden3_size, output_size)
        self.batch_size = batch_size

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        return x

    # New: 平均ニ乗誤差
    def lossfun(self, y, t):
        t = t.unsqueeze(1)
        return F.mse_loss(y, t)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.00001)

    def setData(self,train_data, val_data, test_data):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data