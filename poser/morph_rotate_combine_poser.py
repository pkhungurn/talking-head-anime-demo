from typing import List

import torch
from torch import Tensor

from poser.poser import Poser, PoseParameter
from tha.batch_input_module import BatchInputModuleSpec
from util import torch_load


class MorphRotateCombinePoser(Poser):
    def __init__(self,
                 morph_module_spec: BatchInputModuleSpec,
                 morph_module_file_name: str,
                 morph_module_parameters: List[PoseParameter],
                 rotate_module_spec: BatchInputModuleSpec,
                 rotate_module_file_name: str,
                 rotate_module_parameters: List[PoseParameter],
                 combine_module_spec: BatchInputModuleSpec,
                 combine_module_file_name: str,
                 image_size: int,
                 device: torch.device):
        self.morph_module_spec = morph_module_spec
        self.morph_module_file_name = morph_module_file_name
        self.morph_module_parameters = morph_module_parameters
        self.rotate_module_spec = rotate_module_spec
        self.rotate_module_file_name = rotate_module_file_name
        self.rotate_module_parameters = rotate_module_parameters
        self.combine_module_spec = combine_module_spec
        self.combine_module_file_name = combine_module_file_name
        self.device = device
        self._image_size = image_size

        self.morph_module = None
        self.pose_module = None
        self.combine_module = None

    def image_size(self):
        return self._image_size

    def pose_parameters(self) -> List[PoseParameter]:
        return self.rotate_module_parameters + self.morph_module_parameters

    def get_morph_module(self):
        if self.morph_module is None:
            G = self.morph_module_spec.get_module().to(self.device)
            G.load_state_dict(torch_load(self.morph_module_file_name, map_location=self.device))
            self.morph_module = G
            G.train(False)
        return self.morph_module

    def get_rotate_module(self):
        if self.pose_module is None:
            G = self.rotate_module_spec.get_module().to(self.device)
            G.load_state_dict(torch_load(self.rotate_module_file_name, map_location=self.device))
            self.pose_module = G
            G.train(False)
        return self.pose_module

    def get_combine_module(self):
        if self.combine_module is None:
            G = self.combine_module_spec.get_module().to(self.device)
            G.load_state_dict(torch_load(self.combine_module_file_name, map_location=self.device))
            self.combine_module = G
            G.train(False)
        return self.combine_module

    def pose(self, source_image: Tensor, pose: Tensor):
        morph_param_count = len(self.morph_module_parameters)
        rotate_param_count = len(self.rotate_module_parameters)

        morph_params = pose[:, rotate_param_count:rotate_param_count + morph_param_count]
        rotate_params = pose[:, 0:rotate_param_count]

        morph_module = self.get_morph_module()
        morphed_image = morph_module(source_image, morph_params)[0]

        rotate_module = self.get_rotate_module()
        rotated_images = rotate_module(morphed_image, rotate_params)

        combine_module = self.get_combine_module()
        combined_image = combine_module(rotated_images[0], rotated_images[1], rotate_params)

        return combined_image[0]


class MorphRotateCombinePoser256Param6(MorphRotateCombinePoser):
    def __init__(self,
                 morph_module_spec: BatchInputModuleSpec,
                 morph_module_file_name: str,
                 rotate_module_spec: BatchInputModuleSpec,
                 rotate_module_file_name: str,
                 combine_module_spec: BatchInputModuleSpec,
                 combine_module_file_name: str,
                 device: torch.device):
        super().__init__(
            morph_module_spec,
            morph_module_file_name,
            [
                PoseParameter("left_eye", "Left Eye", 0.0, 1.0, 0.0),
                PoseParameter("right_eye", "Right Eye", 0.0, 1.0, 0.0),
                PoseParameter("mouth", "Mouth", 0.0, 1.0, 1.0)
            ],
            rotate_module_spec,
            rotate_module_file_name,
            [
                PoseParameter("head_x", "Head X", -1.0, 1.0, 0.0),
                PoseParameter("head_y", "Head Y", -1.0, 1.0, 0.0),
                PoseParameter("neck_z", "Neck Z", -1.0, 1.0, 0.0),
            ],
            combine_module_spec,
            combine_module_file_name,
            256,
            device)
