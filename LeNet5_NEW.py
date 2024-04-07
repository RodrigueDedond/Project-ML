# LeNet5 (simplified version) with Pytorch
import torch.nn as nn
import torch


class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        self.features = torch.nn.Sequential(
            nn.Conv2d(1, 6, 5),  # C1
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.AvgPool2d(2),  # S2
            nn.Conv2d(6, 16, 5),  # C3
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.AvgPool2d(2),  # C4
        )
        self.dense = torch.nn.Sequential(
            nn.Linear(5 * 5 * 16, 120),  # F5
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(120, 84),  # F6
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        y = self.dense(x)
        return y
