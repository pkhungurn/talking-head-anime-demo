from torch.nn import Module, ModuleList, Sequential, Conv2d

from nn.conv import Conv7Block, DownsampleBlock, UpsampleBlock
from nn.resnet_block import ResNetBlock


class EncoderDecoderModule(Module):
    def __init__(self,
                 image_size: int,
                 image_channels: int,
                 output_channels: int,
                 bottleneck_image_size: int,
                 bottleneck_block_count: int,
                 initialization_method: str = 'he'):
        super().__init__()

        self.module_list = ModuleList()
        self.module_list.append(Conv7Block(image_channels, output_channels))
        current_size = image_size
        current_channels = output_channels
        while current_size > bottleneck_image_size:
            self.module_list.append(DownsampleBlock(current_channels, initialization_method))
            current_size //= 2
            current_channels *= 2

        for i in range(bottleneck_block_count):
            self.module_list.append(ResNetBlock(current_channels, initialization_method))

        while current_size < image_size:
            self.module_list.append(UpsampleBlock(current_channels, current_channels // 2, initialization_method))
            current_size *= 2
            current_channels //= 2

    def forward(self, x):
        for module in self.module_list:
            x = module(x)
        return x
