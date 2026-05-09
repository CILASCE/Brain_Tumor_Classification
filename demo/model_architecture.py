import torch
from torch import nn

class ModelArchitecture(nn.Module):
  def __init__(self, hidden_size_1, hidden_size_2, hidden_size_3, hidden_size_4):
    super().__init__()

    self.block_1 = nn.Sequential(
        nn.Conv2d(in_channels=1,
                  out_channels=32,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,
                     stride=2)
    )

    self.block_2 = nn.Sequential(
        nn.Conv2d(in_channels=32,
                  out_channels=64,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(in_channels=64,
                  out_channels=128,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,
                     stride=2)
    )

    self.block_3 = nn.Sequential(
        nn.Conv2d(in_channels=128,
                  out_channels=128,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(in_channels=128,
                  out_channels=128,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(in_channels=128,
                  out_channels=256,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,
                     stride=2)
    )

    self.block_4 = nn.Sequential(
        nn.Conv2d(in_channels=256,
                  out_channels=256,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Conv2d(in_channels=256,
                  out_channels=256,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Conv2d(in_channels=256,
                  out_channels=256,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Conv2d(in_channels=256,
                  out_channels=256,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,
                     stride=2)
    )

    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=256*14*14,
                  out_features=hidden_size_1),
        nn.ReLU(),
        nn.Linear(in_features=hidden_size_1,
                  out_features=hidden_size_2),
        nn.ReLU(),
        nn.Linear(in_features=hidden_size_2,
                  out_features=hidden_size_3),
        nn.ReLU(),
        nn.Linear(in_features=hidden_size_3,
                  out_features=hidden_size_4),
        nn.ReLU(),
        nn.Linear(in_features=hidden_size_4,
                  out_features=4)
    )

  def forward(self, x:torch.Tensor) -> torch.Tensor:
    return self.classifier(self.block_4(self.block_3(self.block_2(self.block_1(x)))))
