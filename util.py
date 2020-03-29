import os

import PIL.Image
import numpy
import torch
from torch import Tensor


def is_power2(x):
    return x != 0 and ((x & (x - 1)) == 0)


def torch_save(content, file_name):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'wb') as f:
        torch.save(content, f)


def torch_load(file_name, **kwargs):
    with open(file_name, 'rb') as f:
        return torch.load(f, **kwargs)


def srgb_to_linear(x):
    x = numpy.clip(x, 0.0, 1.0)
    return numpy.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = numpy.clip(x, 0.0, 1.0)
    return numpy.where(x <= 0.003130804953560372, x * 12.92, 1.055 * (x ** (1.0 / 2.4)) - 0.055)


def save_rng_state(file_name):
    rng_state = torch.get_rng_state()
    torch_save(rng_state, file_name)


def load_rng_state(file_name):
    rng_state = torch_load(file_name)
    torch.set_rng_state(rng_state)


def optimizer_to_device(optim, device):
    for state in optim.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def rgba_to_numpy_image_greenscreen(torch_image: Tensor):
    height = torch_image.shape[1]
    width = torch_image.shape[2]

    numpy_image = (torch_image.numpy().reshape(4, height * width).transpose().reshape(height, width, 4) + 1.0) * 0.5
    rgb_image = linear_to_srgb(numpy_image[:, :, 0:3])
    a_image = numpy_image[:, :, 3]
    rgb_image[:, :, 0:3] = rgb_image[:, :, 0:3] * a_image.reshape(a_image.shape[0], a_image.shape[1], 1)
    rgb_image[:, :, 1] = rgb_image[:, :, 1] + (1 - a_image)

    return rgb_image


def rgba_to_numpy_image(torch_image: Tensor):
    height = torch_image.shape[1]
    width = torch_image.shape[2]

    numpy_image = (torch_image.numpy().reshape(4, height * width).transpose().reshape(height, width, 4) + 1.0) * 0.5
    rgb_image = linear_to_srgb(numpy_image[:, :, 0:3])
    a_image = numpy_image[:, :, 3]
    rgba_image = numpy.concatenate((rgb_image, a_image.reshape(height, width, 1)), axis=2)
    return rgba_image


def extract_pytorch_image_from_filelike(file):
    pil_image = PIL.Image.open(file)
    image_size = pil_image.width
    image = (numpy.asarray(pil_image) / 255.0).reshape(image_size, image_size, 4)
    image[:, :, 0:3] = srgb_to_linear(image[:, :, 0:3])
    image = image \
        .reshape(image_size * image_size, 4) \
        .transpose() \
        .reshape(4, image_size, image_size)
    torch_image = torch.from_numpy(image).float() * 2.0 - 1.0
    return torch_image


def extract_numpy_image_from_filelike(file):
    pil_image = PIL.Image.open(file)
    image_size = pil_image.width
    image = (numpy.asarray(pil_image) / 255.0).reshape(image_size, image_size, 4)
    image[:, :, 0:3] = srgb_to_linear(image[:, :, 0:3])
    return image


def create_parent_dir(file_name):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
