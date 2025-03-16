import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50TS(nn.Module):
    def __init__(self, input_dim=1, seq_len=100, num_classes=4, device="cuda:0"):
        super(ResNet50TS, self).__init__()

        self.resnet50 = models.resnet50(pretrained=False)

        self.resnet50.conv1 = nn.Conv2d(input_dim, 64, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0), bias=False)
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, num_classes)

    def forward(self, x):
        x = x.unsqueeze(2)  # (batch_size, seq_len, input_dim) -> (batch_size, seq_len, 1, input_dim)
        x = x.permute(0, 3, 1, 2)  # (batch_size, seq_len, 1, input_dim) -> (batch_size, input_dim, seq_len, 1)
        x = self.resnet50(x)  # (batch_size, num_classes)
        return x