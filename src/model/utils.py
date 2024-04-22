import requests
import torch
import torch.nn as nn
from PIL import Image


def get_generation(model: nn.Module, processor, image: Image.Image, dtype: torch.dtype) -> str:
    inputs = processor(image, return_tensors="pt").to(dtype)
    out = model.generate(**inputs)

    return processor.decode(out[0], skip_special_tokens=True)


def load_image(img_url: str) -> Image.Image:
    image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

    return image


def print_param_dtype(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        print(f"{name} is loaded in {param.dtype}")
