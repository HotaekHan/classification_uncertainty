import torch.nn as nn

class SELayer(nn.Module):
    def __init__(self, n_channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(n_channel, n_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(n_channel // reduction, n_channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        return x * y.expand_as(x)