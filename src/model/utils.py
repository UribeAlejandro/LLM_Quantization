from typing import Dict, Union

import requests
import torch
import torch.nn as nn
import transformers
from PIL import Image


def get_generation(model: nn.Module, processor: transformers.models, image: Image.Image, dtype: torch.dtype) -> str:
    """Generate text from an image.

    Parameters
    ----------
    model : nn.Module
        The model to use for generation.
    processor : transformers.Processor
        The processor to use for the model.
    image : Image.Image
        The image to generate text from.
    dtype : torch.dtype
        The dtype to use for the input tensor.
    Returns
    -------
    str
        The generated text.
    """
    inputs = processor(image, return_tensors="pt").to(dtype)
    out = model.generate(**inputs)

    return processor.decode(out[0], skip_special_tokens=True)


def load_image(img_url: str) -> Image.Image:
    """Load an image from a URL.

    Parameters
    ----------
    img_url : str
        The URL of the image to load.
    Returns
    -------
    Image.Image
        The loaded image.
    """
    image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

    return image


def print_param_dtype(model: nn.Module) -> None:
    """Print the name of the parameters and the dtype they are loaded in.

    Parameters
    ----------
    model : nn.Module
        The model to analyze.
    """
    for name, param in model.named_parameters():
        print(f"{name} is loaded in {param.dtype}")


def named_module_tensors(module: nn.Module, recurse: bool = False) -> Dict[str, torch.Tensor]:
    """Returns an iterator over the named tensors of a module.

    Parameters
    ----------
    module : nn.Module
        The module to analyze.
    recurse : bool
        Whether to recurse into submodules.
    Returns
    -------
    Dict[str, torch.Tensor]
        The named tensors.
    """
    for named_parameter in module.named_parameters(recurse=recurse):
        name, val = named_parameter
        # flag = True
        if hasattr(val, "_data") or hasattr(val, "_scale"):
            if hasattr(val, "_data"):
                yield name + "._data", val._data
            if hasattr(val, "_scale"):
                yield name + "._scale", val._scale
        else:
            yield named_parameter

    for named_buffer in module.named_buffers(recurse=recurse):
        yield named_buffer


def dtype_byte_size(dtype: torch.dtype) -> Union[float, int]:
    """Returns the size (in bytes) occupied by one parameter of type `dtype`.

    Parameters
    ----------
    dtype : torch.dtype
        The dtype to analyze.
    """
    import re

    if dtype == torch.bool:
        return 1 / 8
    bit_search = re.search(r"[^\d](\d+)$", str(dtype))
    if bit_search is None:
        raise ValueError(f"`dtype` is not a valid dtype: {dtype}.")
    bit_size = int(bit_search.groups()[0])
    return bit_size // 8


def compute_module_sizes(model: nn.Module) -> Dict[str, int]:
    """Compute the size of each submodule of a given model.

    Parameters
    ----------
    model : nn.Module
        The model to analyze.
    """
    from collections import defaultdict

    module_sizes = defaultdict(int)
    for name, tensor in named_module_tensors(model, recurse=True):
        size = tensor.numel() * dtype_byte_size(tensor.dtype)
        name_parts = name.split(".")
        for idx in range(len(name_parts) + 1):
            module_sizes[".".join(name_parts[:idx])] += size

    return module_sizes
