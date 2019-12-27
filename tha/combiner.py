import torch
from torch import Tensor
from torch.nn import Sequential, Sigmoid, Tanh

from tha.batch_input_module import BatchInputModule, BatchInputModuleSpec
from nn.conv import Conv7
from nn.u_net_module import UNetModule


class Combiner(BatchInputModule):
    def __init__(self,
                 image_size: int = 256,
                 image_channels: int = 4,
                 pose_size: int = 3,
                 intermediate_channels: int = 64,
                 bottleneck_image_size: int = 32,
                 bottleneck_block_count: int = 6,
                 initialization_method: str = 'he'):
        super().__init__()
        self.main_body = UNetModule(
            image_size=image_size,
            image_channels=2 * image_channels + pose_size,
            output_channels=intermediate_channels,
            bottleneck_image_size=bottleneck_image_size,
            bottleneck_block_count=bottleneck_block_count,
            initialization_method=initialization_method)
        self.combine_alpha_mask = Sequential(
            Conv7(intermediate_channels, image_channels, initialization_method),
            Sigmoid())
        self.retouch_alpha_mask = Sequential(
            Conv7(intermediate_channels, image_channels, initialization_method),
            Sigmoid())
        self.retouch_color_change = Sequential(
            Conv7(intermediate_channels, image_channels, initialization_method),
            Tanh())

    def forward(self, first_image: Tensor, second_image: Tensor, pose: Tensor):
        pose = pose.unsqueeze(2).unsqueeze(3)
        pose = pose.expand(pose.size(0), pose.size(1), first_image.size(2), first_image.size(3))

        x = torch.cat([first_image, second_image, pose], dim=1)
        y = self.main_body(x)
        combine_alpha_mask = self.combine_alpha_mask(y)
        combined_image = combine_alpha_mask * first_image + (1 - combine_alpha_mask) * second_image
        retouch_alpha_mask = self.retouch_alpha_mask(y)
        retouch_color_change = self.retouch_color_change(y)
        final_image = retouch_alpha_mask * combined_image + (1 - retouch_alpha_mask) * retouch_color_change
        return [final_image, combined_image, combine_alpha_mask, retouch_alpha_mask, retouch_color_change]

    def forward_from_batch(self, batch):
        return self.forward(batch[0], batch[1], batch[2])


class CombinerSpec(BatchInputModuleSpec):
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
        return Combiner(
            self.image_size,
            self.image_channels,
            self.pose_size,
            self.intermediate_channels,
            self.bottleneck_image_size,
            self.bottleneck_block_count,
            self.initialization_method)