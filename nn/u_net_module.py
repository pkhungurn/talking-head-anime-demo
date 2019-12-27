import torch
from torch.nn import Module, ModuleList

from nn.conv import Conv7Block, DownsampleBlock, UpsampleBlock
from nn.resnet_block import ResNetBlock


class UNetModule(Module):
    def __init__(self,
                 image_size: int,
                 image_channels: int,
                 output_channels: int,
                 bottleneck_image_size: int,
                 bottleneck_block_count: int,
                 initialization_method: str = 'he'):
        super().__init__()
        self.downward_modules = ModuleList()
        self.downward_module_channel_count = {}

        self.downward_modules.append(Conv7Block(image_channels, output_channels, initialization_method))
        self.downward_module_channel_count[image_size] = output_channels

        # Downsampling
        current_channels = output_channels
        current_image_size = image_size
        while current_image_size > bottleneck_image_size:
            self.downward_modules.append(DownsampleBlock(current_channels, initialization_method))
            current_channels = current_channels * 2
            current_image_size = current_image_size // 2
            self.downward_module_channel_count[current_image_size] = current_channels

        # Bottleneck
        self.bottleneck_modules = ModuleList()
        for i in range(bottleneck_block_count):
            self.bottleneck_modules.append(ResNetBlock(current_channels, initialization_method))

        # Upsampling
        self.upsampling_modules = ModuleList()
        while current_image_size < image_size:
            if current_image_size == bottleneck_image_size:
                input_channels = current_channels
            else:
                input_channels = current_channels + self.downward_module_channel_count[current_image_size]
            self.upsampling_modules.insert(0,
                                           UpsampleBlock(input_channels, current_channels // 2, initialization_method))
            current_channels = current_channels // 2
            current_image_size = current_image_size * 2

        self.upsampling_modules.insert(
            0, Conv7Block(current_channels + output_channels, output_channels, initialization_method))

    def forward(self, x):
        downward_outputs = []
        for module in self.downward_modules:
            x = module(x)
            downward_outputs.append(x)
        for module in self.bottleneck_modules:
            x = module(x)
        x = self.upsampling_modules[-1](x)
        for i in range(len(self.upsampling_modules) - 2, -1, -1):
            y = torch.cat([x, downward_outputs[i]], dim=1)
            x = self.upsampling_modules[i](y)
        return x
