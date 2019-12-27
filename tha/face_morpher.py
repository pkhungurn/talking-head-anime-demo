import torch
from torch import Tensor
from torch.nn import Sequential, Tanh, Sigmoid

from tha.batch_input_module import BatchInputModule, BatchInputModuleSpec
from nn.conv import Conv7
from nn.encoder_decoder_module import EncoderDecoderModule


class FaceMorpher(BatchInputModule):
    def __init__(self,
                 image_size: int = 256,
                 image_channels: int = 4,
                 pose_size: int = 3,
                 intermediate_channels: int = 64,
                 bottleneck_image_size: int = 32,
                 bottleneck_block_count: int = 6,
                 initialization_method: str = 'he'):
        super().__init__()
        self.main_body = EncoderDecoderModule(
            image_size=image_size,
            image_channels=image_channels + pose_size,
            output_channels=intermediate_channels,
            bottleneck_image_size=bottleneck_image_size,
            bottleneck_block_count=bottleneck_block_count,
            initialization_method=initialization_method)
        self.color_change = Sequential(
            Conv7(intermediate_channels, image_channels, initialization_method),
            Tanh())
        self.alpha_mask = Sequential(
            Conv7(intermediate_channels, image_channels, initialization_method),
            Sigmoid())

    def forward(self, image: Tensor, pose: Tensor):
        pose = pose.unsqueeze(2).unsqueeze(3)
        pose = pose.expand(pose.size(0), pose.size(1), image.size(2), image.size(3))
        x = torch.cat([image, pose], dim=1)
        y = self.main_body(x)
        color = self.color_change(y)
        alpha = self.alpha_mask(y)
        output_image = alpha * image + (1 - alpha) * color
        return [output_image, alpha, color]

    def forward_from_batch(self, batch):
        return self.forward(batch[0], batch[1])


class FaceMorpherSpec(BatchInputModuleSpec):
    def __init__(self,
                 image_size: int = 256,
                 image_channels: int = 4,
                 pose_size: int = 3,
                 intermediate_channels: int = 64,
                 bottleneck_image_size: int = 32,
                 bottleneck_block_count: int = 6,
                 initialization_method: str = 'he'):
        self.image_size = image_size
        self.image_channels = image_channels
        self.pose_size = pose_size
        self.intermediate_channels = intermediate_channels
        self.bottleneck_image_size = bottleneck_image_size
        self.bottleneck_block_count = bottleneck_block_count
        self.initialization_method = initialization_method

    def get_module(self) -> BatchInputModule:
        return FaceMorpher(
            self.image_size,
            self.image_channels,
            self.pose_size,
            self.intermediate_channels,
            self.bottleneck_image_size,
            self.bottleneck_block_count,
            self.initialization_method)