# Large Language Models - Quantization <!-- omit in toc -->

In this project, we explore the concept of quantization in large language models. Quantization is a technique used to reduce the memory footprint and computational requirements of these models, making them more efficient for deployment on resource-constrained devices.

By applying quantization, we aim to strike a balance between model size and performance. In this README, we will discuss the process of quantization, its benefits, and how to implement it in large language models.

Let's dive into the world of quantization and discover how it can optimize the deployment of large language models!

## Table of Contents <!-- omit in toc -->

- [Installation](#installation)
- [Overview](#overview)
  - [Downcasting](#downcasting)
  - [Quantization](#quantization)
    - [Linear Quantization vs Downcasting](#linear-quantization-vs-downcasting)

## Installation

**Optional**: Create a virtual environment using `venv`:

```bash
make venv
```

Install the dependencies, the command will install the required packages/libraries listed in the `requirements-dev.txt` & `requirements.txt` files.

```bash
make install
```

## Overview

Large language models, such as GPT-3, BERT, and T5, have revolutionized natural language processing (NLP) tasks by achieving state-of-the-art performance on a wide range of benchmarks. However, these models come with a significant computational cost due to their large size and complex architecture.

`Pruning`, `Quantization` & `Distillation` are techniques used to reduce the memory footprint and computational requirements of large language models. The same are discussed below:

1. **Pruning**: Pruning involves removing unnecessary connections (weights) from the model, which reduces the number of parameters and the computational cost of the model. It helps in reducing the model size and improving the inference speed.
2. **Quantization**: Quantization is a technique used to reduce the memory footprint of the model by representing the model's parameters in `lower bit-width` float or integers. It helps in reducing the memory requirements and speeding up the computation.
3. **Distillation**: Distillation is a technique used to transfer knowledge from a large, complex model (teacher) to a smaller, simpler model (student). It helps in compressing the knowledge of the teacher model into the student model, making it more efficient for deployment.

The problem at hand is to implement `quantization` in large language models to reduce the memory footprint and computational requirements of the model. It involves converting the model's parameters from `floating-point` precision (e.g. `FP32`) to `lower bit-width` `float` or `integers` (e.g. `FP16`, `INT8`, `UINT8`, etc.).

`Quantization` can be applied to different parts of the model, such as `weights` and/or `activations`, to optimize the model for deployment on resource-constrained devices. It helps in reducing the memory requirements, improving the inference speed, and making the model more efficient for deployment in real-world applications.

However, the quantization error introduced during the quantization process can affect the model's performance. Therefore, it is essential to carefully choose the quantization parameters and techniques to minimize the quantization error while maximizing the benefits of quantization.

Let's review the most common quantization: `FP32` -> `INT8`. For instance, a `FP32` uses `1bit` to represent the `sign`, `8bits` to represent the `exponent`, and `23bits` to represent the `mantissa` or `fraction`. The different ranges per precision (`PyTorch` data types) are shown below:

- `FP64`: `-1.7^308` to `1.7^308`
- `FP32`: `-3.4^38` to `3.5^38`
- `FP16`: `-65504` to `65504`
- `BFLOAT16`: `-3.4^38` to `3.5^38`\*
- `INT64`: `-9.2^18` to `9.2^18`
- `INT32`: `-2.1^9` to `2.1^9`
- `INT16`: `-32768` to `32767`
- `INT8`: `-128` to `127`
- `UINT8`: `0` to `255`

Further details can be found in [DataTypes_Experiments.ipynb](notebooks/1_DataTypes_Experiments.ipynb).

> **Note:** *`BFLOAT16` is a `16-bit` floating-point format that is used in some deep learning frameworks, such as `TensorFlow`, for training neural networks. It has a `8-bit` exponent and an `7-bit` mantissa, which provides a good balance between precision and memory efficiency.

### Downcasting

The process of converting a tensor from a higher to a lower precision is known as `downcasting`. A known application of this process is the `Mixed Precision Training`, where te comutation are carried out in smaller precision (e.g. `FP16`, `BF16`, `FP8`) and the store/update of the weights are computed in a higher precision (e.g. `FP32`).

`PyTorch` has a built-in support for `FP16` (half-precision) tensors, which can be used to reduce the memory footprint and computational requirements of the model. The `FP16` tensors are stored in `16-bit` floating-point format, which requires half the memory of `FP32` tensors. To do so, the methods are shown below:

```python
model.half()
model.bfloat16()
model.to(torch.float16)
```

The implementation can be found in [PyTorch Model Downcasting](notebooks/2_TorchModel_Downcasting.ipynb).

The downcasting process can be applied on HuggingFace models, as shown in [Hugging Face Model Downcasting](notebooks/3_Transformers_Model_Downcasting.ipynb).

### Quantization

The process of converting a tensor from a floating-point precision to a lower bit-width integer or float is known as `quantization`. The quantization process can be applied to different parts of the model, such as `weights` and/or `activations`, to reduce the memory footprint and computational requirements of the model.

[Quanto](https://huggingface.co/blog/quanto-introduction) is a library developed by `Hugging Face` that provides an easy-to-use interface for quantizing models.

A hands-on example is in [Quantization with Quanto](notebooks/4_Quantization_with_Quanto.ipynb).

The process is listed below:

1. **Quantize**: Quantize the model's weights and/or activations to lower bit-width integers or floats.

The formula of `linear quantization` is shown below:

$$ r = s(q-z) $$

Where:
- $r$: the original value (e.g. `FP32`)
- $q$: the quantized value (e.g. `INT8`)
- $s$: the scale factor
- $z$: the zero-point

To calculate $r$ & $z$, the formulas are shown below:

$$ s = \frac{r_{max} - r_{min}}{q_{max} - q_{min}} $$

$$ z = int(round(q_{min} - r_{min} / s)) $$

2. `Optional` **Calibration**: Calibrate the quantized model to minimize the quantization error and maximize the performance.

3. `Optional` **Quantization Aware Training**: Train the quantized model using quantization-aware training to further optimize the model for deployment.

#### Linear Quantization vs Downcasting

The difference between the `linear quantization` method with the `downcasting` method take into account:

- `Downcasting`: convert the model's parameters to a more compact data type (e.g. `bfloat16`). `Downcasting` may work with the `bfloat16` data type, but the model performance will likely degrade with any smaller data type, and won't work if you convert to an integer data type (e.g. `int8`).
  - During inference: the model performs its calculations in `bfloat16` data type, and its activations are in same data type.
- `Linear Quantization`: enables the `quantized model` to maintain performance much closer to the original model by converting from the compressed data type back to the original `FP32` data type `during inference`.
  - During inference: the model performs the matrix multiplications in `FP32`, and the activations are in `FP32`. This enables you to quantize the model in data types smaller than `bfloat16`, such as `int8`, in this example.
