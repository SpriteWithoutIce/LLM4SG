import torch
import torch.nn as nn
import torchvision.models as models

class EfficientNetTS(nn.Module):
    def __init__(self, input_dim=1, seq_len=100, num_classes=4, device="cuda:0"):
        super(EfficientNetTS, self).__init__()

        self.efficientnet = models.efficientnet_b0(pretrained=False)

        original_first_conv = self.efficientnet.features[0][0]
        self.efficientnet.features[0][0] = nn.Conv2d(
            input_dim, 
            original_first_conv.out_channels, 
            kernel_size=(7, 1), 
            stride=(2, 1), 
            padding=(3, 0), 
            bias=False
        )

        original_classifier = self.efficientnet.classifier[1]
        self.efficientnet.classifier[1] = nn.Linear(
            original_classifier.in_features, 
            num_classes
        )

    def forward(self, x):
        x = x.unsqueeze(2) 
        x = x.permute(0, 3, 1, 2) 
        x = self.efficientnet(x)
        return x