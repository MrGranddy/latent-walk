import argparse
import os
import sys
import pickle
import math

import torch
import numpy as np
from torchvision import utils

import math
import random

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d


def expand_to_bach(value, batch_size, target_type):
    try:
        assert (
            value.shape[0] == batch_size
        ), f"batch size is not equal to the tensor size"
    except Exception:
        value = value * torch.ones(batch_size, dtype=target_type)
    return value.to(target_type)


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer("kernel", kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return "{}({}, {},".format(
            self.__class__.__name__, self.weight.shape[1], self.weight.shape[0]
        ) + " {}, stride={}, padding={})".format(
            self.weight.shape[2], self.stride, self.padding
        )


class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return "{}({}, {})".format(
            self.__class__.__name__, self.weight.shape[1], self.weight.shape[0]
        )


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return "{}({}, {}, {}, ".format(
            self.__class__.__name__, self.in_channel, self.out_channel, self.kernel_size
        ) + "upsample={}, downsample={})".format(self.upsample, self.downsample)

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class ModulatedConv2dPatchedFixedBasisDelta(nn.Module):
    def __init__(self, conv_to_patch, basis_vectors, directions_count):
        super(ModulatedConv2dPatchedFixedBasisDelta, self).__init__()

        self.eps = 1e-8
        self.kernel_size = conv_to_patch.kernel_size
        self.in_channel = conv_to_patch.in_channel
        self.out_channel = conv_to_patch.out_channel
        self.upsample = conv_to_patch.upsample
        self.downsample = conv_to_patch.downsample

        if self.upsample or self.downsample:
            self.blur = conv_to_patch.blur

        self.scale = conv_to_patch.scale
        self.padding = conv_to_patch.padding
        self.weight = conv_to_patch.weight
        self.modulation = conv_to_patch.modulation
        self.demodulate = conv_to_patch.demodulate
        self.basis_vectors = basis_vectors

        assert (
            self.basis_vectors.shape[1:] == self.weight.shape
        ), f"unconsisted basis and weight {self.basis_vectors.shape[1:]} != {self.weight.shape}"

        basis_dim = len(self.basis_vectors)
        self.direction_to_basis_coefs = nn.Parameter(
            torch.randn((directions_count, basis_dim))
        )

    def weight_shifts(self, batch):
        # expand scalar shift params if required
        basis_size = len(self.basis_vectors)
        batch_directions = expand_to_bach(
            self.batch_directions, batch, torch.long
        ).cuda()
        batch_shifts = expand_to_bach(self.batch_shifts, batch, torch.float32).cuda()

        # deformation
        batch_weight_delta = torch.stack(
            batch * [self.basis_vectors], dim=0
        )  # (batch_size, basis_size, *weight.shape)
        batch_basis_coefs = self.direction_to_basis_coefs[
            batch_directions
        ]  # (batch_size, basis_size)

        batch_weight_delta = batch_weight_delta.view(
            batch, basis_size, -1
        )  # (batch_size, basis_size, -1)
        batch_basis_coefs = batch_basis_coefs.unsqueeze(
            -1
        )  # (batch_size, basis_size, -1)

        batch_weight_delta = (batch_weight_delta * batch_basis_coefs).sum(
            dim=1
        )  # (batch_size, -1)
        batch_weight_delta = F.normalize(batch_weight_delta, p=2, dim=1)
        batch_weight_delta *= batch_shifts[:, None]  # (batch_size, -1)

        batch_weight_delta = batch_weight_delta.view(
            batch, *self.weight.shape[-4:]
        )  # (batch_size, c_out, c_in, k_x, k_y)
        return batch_weight_delta

    def forward(self, input, style):
        # need to set self.is_deformated (bool), self.batch_directions(LongTensor), self.batch_shifts(FloatTensor)
        batch, in_channel, height, width = input.shape
        weight = self.weight
        if self.is_deformated:
            weight = weight + self.weight_shifts(batch)

        # further code is from original ModulatedConv2d
        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class ModulatedConv2dPatchedSVDBasisDelta(nn.Module):
    def __init__(self, conv_to_patch, directions_count):
        super(ModulatedConv2dPatchedSVDBasisDelta, self).__init__()

        self.eps = 1e-8
        self.kernel_size = conv_to_patch.kernel_size
        self.in_channel = conv_to_patch.in_channel
        self.out_channel = conv_to_patch.out_channel
        self.upsample = conv_to_patch.upsample
        self.downsample = conv_to_patch.downsample

        if self.upsample or self.downsample:
            self.blur = conv_to_patch.blur

        self.scale = conv_to_patch.scale
        self.padding = conv_to_patch.padding
        self.weight = conv_to_patch.weight
        self.modulation = conv_to_patch.modulation
        self.demodulate = conv_to_patch.demodulate

        weight_matrix = (
            conv_to_patch.weight.cpu()
            .detach()
            .numpy()
            .reshape((conv_to_patch.weight.shape[-4:]))
        )
        c_out, c_in, k_x, k_y = weight_matrix.shape
        weight_matrix = np.transpose(weight_matrix, (2, 3, 1, 0))
        weight_matrix = np.reshape(weight_matrix, (k_x * k_y * c_in, c_out))

        u, s, vh = np.linalg.svd(weight_matrix, full_matrices=False)
        u = torch.FloatTensor(u).cuda()
        vh = torch.FloatTensor(vh).cuda()

        self.u = nn.Parameter(u, requires_grad=False)
        self.vh = nn.Parameter(vh, requires_grad=False)

        self.direction_to_eigenvalues_delta = torch.randn((directions_count, len(s)))
        self.direction_to_eigenvalues_delta = nn.Parameter(
            self.direction_to_eigenvalues_delta, requires_grad=True
        )

    def weight_shifts(self, batch):
        # expand scalar shift params if required
        batch_directions = expand_to_bach(
            self.batch_directions, batch, torch.long
        ).cuda()
        batch_shifts = expand_to_bach(self.batch_shifts, batch, torch.float32).cuda()

        batch_eigenvalues_delta = self.direction_to_eigenvalues_delta[
            batch_directions
        ]  # (batch, len(s))
        batch_weight_delta = (
            self.u @ torch.diag_embed(batch_eigenvalues_delta) @ self.vh
        )

        c_out, c_in, k_x, k_y = self.weight.shape[-4:]
        batch_weight_delta = F.normalize(batch_weight_delta.view(batch, -1), dim=1, p=2)
        batch_weight_delta = batch_weight_delta.view(batch, k_x, k_y, c_in, c_out)
        batch_weight_delta = batch_weight_delta.permute(0, 4, 3, 1, 2)
        assert batch_weight_delta.shape == (batch, c_out, c_in, k_x, k_y)

        batch_weight_delta *= batch_shifts[:, None, None, None, None]
        return batch_weight_delta

    def forward(self, input, style):
        # need to set self.is_deformated (bool), self.batch_directions(LongTensor), self.batch_shifts(FloatTensor)
        batch, in_channel, height, width = input.shape

        weight = self.weight
        if self.is_deformated:
            weight = self.weight
            weight = weight + self.weight_shifts(batch)

        # further code is from original ModulatedConv2d
        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out


class Generator(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
    ):
        super().__init__()

        self.size = size

        self.style_dim = style_dim

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"
                )
            )

        self.style = nn.Sequential(*layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(
                "noise_{}".format(layer_idx), torch.randn(*shape)
            )

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )

            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, style_dim))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def make_noise(self):
        device = self.input.input.device

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)

    def forward(
        self,
        styles,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=False,
    ):
        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, "noise_{}".format(i))
                    for i in range(self.num_layers)
                ]

        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                latent = styles[0]

        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        out = self.input(latent)
        out = self.conv1(out, latent[:, 0], noise=noise[0])

        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)

            i += 2

        image = skip

        if return_latents:
            return image, latent

        else:
            return image, None


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))

            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out


def convert_modconv(vars, source_name, target_name, flip=False):
    weight = vars[source_name + "/weight"].value().eval()
    mod_weight = vars[source_name + "/mod_weight"].value().eval()
    mod_bias = vars[source_name + "/mod_bias"].value().eval()
    noise = vars[source_name + "/noise_strength"].value().eval()
    bias = vars[source_name + "/bias"].value().eval()

    dic = {
        "conv.weight": np.expand_dims(weight.transpose((3, 2, 0, 1)), 0),
        "conv.modulation.weight": mod_weight.transpose((1, 0)),
        "conv.modulation.bias": mod_bias + 1,
        "noise.weight": np.array([noise]),
        "activate.bias": bias,
    }

    dic_torch = {}

    for k, v in dic.items():
        dic_torch[target_name + "." + k] = torch.from_numpy(v)

    if flip:
        dic_torch[target_name + ".conv.weight"] = torch.flip(
            dic_torch[target_name + ".conv.weight"], [3, 4]
        )

    return dic_torch


def convert_conv(vars, source_name, target_name, bias=True, start=0):
    weight = vars[source_name + "/weight"].value().eval()

    dic = {"weight": weight.transpose((3, 2, 0, 1))}

    if bias:
        dic["bias"] = vars[source_name + "/bias"].value().eval()

    dic_torch = {}
    dic_torch[target_name + ".{}.weight".format(start)] = torch.from_numpy(
        dic["weight"]
    )

    if bias:
        dic_torch[target_name + ".{}.bias".format(start + 1)] = torch.from_numpy(
            dic["bias"]
        )

    return dic_torch


def convert_torgb(vars, source_name, target_name):
    weight = vars[source_name + "/weight"].value().eval()
    mod_weight = vars[source_name + "/mod_weight"].value().eval()
    mod_bias = vars[source_name + "/mod_bias"].value().eval()
    bias = vars[source_name + "/bias"].value().eval()

    dic = {
        "conv.weight": np.expand_dims(weight.transpose((3, 2, 0, 1)), 0),
        "conv.modulation.weight": mod_weight.transpose((1, 0)),
        "conv.modulation.bias": mod_bias + 1,
        "bias": bias.reshape((1, 3, 1, 1)),
    }

    dic_torch = {}

    for k, v in dic.items():
        dic_torch[target_name + "." + k] = torch.from_numpy(v)

    return dic_torch


def convert_dense(vars, source_name, target_name):
    weight = vars[source_name + "/weight"].value().eval()
    bias = vars[source_name + "/bias"].value().eval()

    dic = {"weight": weight.transpose((1, 0)), "bias": bias}

    dic_torch = {}

    for k, v in dic.items():
        dic_torch[target_name + "." + k] = torch.from_numpy(v)

    return dic_torch


def update(state_dict, new):
    for k, v in new.items():
        if k not in state_dict:
            raise KeyError(k + " is not found")

        if v.shape != state_dict[k].shape:
            raise ValueError(
                "Shape mismatch: {} vs {}".format(v.shape, state_dict[k].shape)
            )

        state_dict[k] = v


def discriminator_fill_statedict(statedict, vars, size):
    log_size = int(math.log(size, 2))

    update(statedict, convert_conv(vars, "{}x{}/FromRGB".format(size, size), "convs.0"))

    conv_i = 1

    for i in range(log_size - 2, 0, -1):
        reso = 4 * 2 ** i
        update(
            statedict,
            convert_conv(
                vars, "{}x{}/Conv0".format(reso, reso), "convs.{}.conv1".format(conv_i)
            ),
        )
        update(
            statedict,
            convert_conv(
                vars,
                "{}x{}/Conv1_down".format(reso, reso),
                "convs.{}.conv2".format(conv_i),
                start=1,
            ),
        )
        update(
            statedict,
            convert_conv(
                vars,
                "{}x{}/Skip".format(reso, reso),
                "convs.{}.skip".format(conv_i),
                start=1,
                bias=False,
            ),
        )
        conv_i += 1

    update(statedict, convert_conv(vars, "4x4/Conv", "final_conv"))
    update(statedict, convert_dense(vars, "4x4/Dense0", "final_linear.0"))
    update(statedict, convert_dense(vars, "Output", "final_linear.1"))

    return statedict


def fill_statedict(state_dict, vars, size):
    log_size = int(math.log(size, 2))

    for i in range(8):
        update(
            state_dict,
            convert_dense(
                vars, "G_mapping/Dense{}".format(i), "style.{}".format(i + 1)
            ),
        )

    update(
        state_dict,
        {
            "input.input": torch.from_numpy(
                vars["G_synthesis/4x4/Const/const"].value().eval()
            )
        },
    )

    update(state_dict, convert_torgb(vars, "G_synthesis/4x4/ToRGB", "to_rgb1"))

    for i in range(log_size - 2):
        reso = 4 * 2 ** (i + 1)
        update(
            state_dict,
            convert_torgb(
                vars,
                "G_synthesis/{}x{}/ToRGB".format(reso, reso),
                "to_rgbs.{}".format(i),
            ),
        )

    update(state_dict, convert_modconv(vars, "G_synthesis/4x4/Conv", "conv1"))

    conv_i = 0

    for i in range(log_size - 2):
        reso = 4 * 2 ** (i + 1)
        update(
            state_dict,
            convert_modconv(
                vars,
                "G_synthesis/{}x{}/Conv0_up".format(reso, reso),
                "convs.{}".format(conv_i),
                flip=True,
            ),
        )
        update(
            state_dict,
            convert_modconv(
                vars,
                "G_synthesis/{}x{}/Conv1".format(reso, reso),
                "convs.{}".format(conv_i + 1),
            ),
        )
        conv_i += 2

    for i in range(0, (log_size - 2) * 2 + 1):
        update(
            state_dict,
            {
                "noises.noise_{}".format(i): torch.from_numpy(
                    vars["G_synthesis/noise{}".format(i)].value().eval()
                )
            },
        )

    return state_dict


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, required=True)
    parser.add_argument("--gen", action="store_true")
    parser.add_argument("--disc", action="store_true")
    parser.add_argument("path", metavar="PATH")

    args = parser.parse_args()

    sys.path.append(args.repo)

    import dnnlib
    from dnnlib import tflib

    tflib.init_tf()

    with open(args.path, "rb") as f:
        generator, discriminator, g_ema = pickle.load(f)

    size = g_ema.output_shape[2]

    g = Generator(size, 512, 8)
    state_dict = g.state_dict()
    state_dict = fill_statedict(state_dict, g_ema.vars, size)

    g.load_state_dict(state_dict)

    latent_avg = torch.from_numpy(g_ema.vars["dlatent_avg"].value().eval())

    ckpt = {"g_ema": state_dict, "latent_avg": latent_avg}

    if args.gen:
        g_train = Generator(size, 512, 8)
        g_train_state = g_train.state_dict()
        g_train_state = fill_statedict(g_train_state, generator.vars, size)
        ckpt["g"] = g_train_state

    if args.disc:
        disc = Discriminator(size)
        d_state = disc.state_dict()
        d_state = discriminator_fill_statedict(d_state, discriminator.vars, size)
        ckpt["d"] = d_state

    name = os.path.splitext(os.path.basename(args.path))[0]
    torch.save(ckpt, name + ".pt")

    batch_size = {256: 16, 512: 9, 1024: 4}
    n_sample = batch_size.get(size, 25)

    g = g.to(device)

    z = np.random.RandomState(0).randn(n_sample, 512).astype("float32")

    with torch.no_grad():
        img_pt, _ = g(
            [torch.from_numpy(z).to(device)],
            truncation=0.5,
            truncation_latent=latent_avg.to(device),
        )

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.randomize_noise = False
    img_tf = g_ema.run(z, None, **Gs_kwargs)
    img_tf = torch.from_numpy(img_tf).to(device)

    img_diff = ((img_pt + 1) / 2).clamp(0.0, 1.0) - ((img_tf.to(device) + 1) / 2).clamp(
        0.0, 1.0
    )

    img_concat = torch.cat((img_tf, img_pt, img_diff), dim=0)
    utils.save_image(
        img_concat, name + ".png", nrow=n_sample, normalize=True, range=(-1, 1)
    )
