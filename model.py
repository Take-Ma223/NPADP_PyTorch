import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """譜面難易度予測ネットワーク（全結合 入力→20→16→8→1・ReLU）。

    入力の標準化（学習用の行の平均 0・標準偏差 1）をモデルの中に持つ。
    ゲーム側（NPADP_pred）はレーダーの生の整数をそのまま送ってくるので、標準化は forward の先頭で行い、
    TorchScript にも一緒に書き出される。
    """

    def __init__(self, input_size=24, hidden1_size=20, hidden2_size=16, hidden3_size=8, output_size=1):
        super().__init__()
        self.register_buffer("in_mean", torch.zeros(input_size))
        self.register_buffer("in_std", torch.ones(input_size))
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, hidden3_size)
        self.fc4 = nn.Linear(hidden3_size, output_size)

    @property
    def input_size(self) -> int:
        return self.fc1.in_features

    def set_standardization(self, x: torch.Tensor):
        """学習用の行 x（行数 × 入力数）から平均と標準偏差を決める。全行同じ値の列は標準偏差 1 にして割り算を避ける。"""
        mean = x.mean(dim=0)
        std = x.std(dim=0, unbiased=False)
        std = torch.where(std > 0, std, torch.ones_like(std))
        self.in_mean.copy_(mean)
        self.in_std.copy_(std)

    def forward(self, x):
        x = (x - self.in_mean) / self.in_std
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)
