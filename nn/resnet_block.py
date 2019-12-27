from torch import relu
from torch.nn import Module, InstanceNorm2d

from nn.conv import Conv3


class ResNetBlock(Module):
    def __init__(self, num_channels: int, initialization_method: str = 'he'):
        super().__init__()
        self.conv1 = Conv3(num_channels, num_channels, initialization_method)
        self.norm1 = InstanceNorm2d(num_features=num_channels, affine=True)
        self.conv2 = Conv3(num_channels, num_channels, initialization_method)
        self.norm2 = InstanceNorm2d(num_features=num_channels, affine=True)

    def forward(self, x):
        return x + self.norm2(self.conv2(relu(self.norm1(self.conv1(x)))))
