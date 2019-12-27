from torch.nn import Conv2d, Module, Sequential, InstanceNorm2d, ReLU, ConvTranspose2d

from nn.init_function import create_init_function


def Conv7(in_channels: int, out_channels: int, initialization_method='he') -> Module:
    init = create_init_function(initialization_method)
    return init(Conv2d(in_channels, out_channels, kernel_size=7, stride=1, padding=3, bias=False))


def Conv3(in_channels: int, out_channels: int, initialization_method='he') -> Module:
    init = create_init_function(initialization_method)
    return init(Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))


def Conv7Block(in_channels: int, out_channels: int, initialization_method='he') -> Module:
    return Sequential(
        Conv7(in_channels, out_channels, initialization_method),
        InstanceNorm2d(out_channels, affine=True),
        ReLU(inplace=True))


def DownsampleBlock(in_channels: int, initialization_method='he') -> Module:
    init = create_init_function(initialization_method)
    return Sequential(
        init(Conv2d(in_channels, in_channels * 2, kernel_size=4, stride=2, padding=1, bias=False)),
        InstanceNorm2d(in_channels * 2, affine=True),
        ReLU(inplace=True))


def UpsampleBlock(in_channels: int, out_channels: int, initialization_method='he') -> Module:
    init = create_init_function(initialization_method)
    return Sequential(
        init(ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)),
        InstanceNorm2d(out_channels, affine=True),
        ReLU(inplace=True))
