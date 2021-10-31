from os import stat_result
from typing import OrderedDict
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from loading import load_generator
import matplotlib.pyplot as plt

from psp_models.psp import pSp
from psp_models.encoders.psp_encoders import GradualStyleEncoder, GradualStyleBlock
from options.test_options import TestOptions

from training.networks_stylegan2 import Generator

from argparse import Namespace

import pickle
from PIL import Image
import shutil
import uuid
import numpy as np
import math
from collections import OrderedDict

device_name = "cuda:0"


def to_image(tensor, img_res):
    tensor = torch.clamp(tensor, -1, 1)
    tensor = (tensor + 1) / 2
    tensor.clamp(0, 1)
    tensor = F.interpolate(
        tensor, size=(img_res, img_res), mode="bilinear", align_corners=True
    )
    return (255 * tensor.cpu().detach()).to(torch.uint8).permute(0, 2, 3, 1).numpy()


def get_opts_and_encoder_sd(path="pretrained/psp_ffhq_encode.pt"):
    test_opts = TestOptions().parse()
    ckpt = torch.load(path)
    state_dict = OrderedDict()

    for key, val in ckpt["state_dict"].items():
        if "encoder" in key and "styles" not in key:
            state_dict[".".join(key.split(".")[1:])] = val

    opts = ckpt["opts"]
    opts.update(vars(test_opts))
    opts["learn_in_w"] = False
    opts["output_size"] = 1024
    opts["n_styles"] = int(math.log(opts["output_size"], 2)) * 2 - 2

    return opts, state_dict, ckpt["latent_avg"]


class ImageEncoder(nn.Module):
    def __init__(self, opts, encoder_sd, input_size=256):
        super(ImageEncoder, self).__init__()

        self.input_size = input_size
        self.opts = opts

        self.encoder = GradualStyleEncoder(50, "ir_se", self.opts)
        self.encoder.load_state_dict(encoder_sd)

        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = F.interpolate(x, size=self.input_size)
        return self.encoder(x)


class LatentMapping(nn.Module):
    def __init__(self, opts, latent_avg):
        super(LatentMapping, self).__init__()

        self.styles = nn.ModuleList()
        self.style_count = opts["n_styles"]
        self.coarse_ind = 3
        self.middle_ind = 7
        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)

        for style in self.styles:
            for param in style.parameters():
                param.requires_grad = True

        self.latent_avg = latent_avg.to(device_name)

    def forward(self, inps):

        c3, p2, p1 = inps
        latents = []

        for j in range(self.coarse_ind):
            latents.append(self.styles[j](c3))

        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p2))

        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](p1))

        out = torch.stack(latents, dim=1)

        return out + self.latent_avg


class Discriminator(nn.Module):
    def __init__(self, input_size=512 * 18):
        super(Discriminator, self).__init__()

        self.linears = nn.ModuleList(
            [
                nn.Linear(input_size, 256),
                nn.Linear(256, 128),
                nn.Linear(128, 64),
            ]
        )

        self.final_fc = nn.Linear(64, 1)

        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):

        x = x.view(x.shape[0], -1)

        for linear in self.linears:
            x = linear(x)
            x = self.relu(x)

        return self.final_fc(x)


class StyleGan(nn.Module):
    def __init__(self):
        super(StyleGan, self).__init__()
        with open("pretrained/StyleGAN2/stylegan2-ffhq-1024x1024.pkl", "rb") as f:
            gen = pickle.load(f)["G_ema"]

        self.G = Generator(
            gen.z_dim, gen.c_dim, gen.w_dim, gen.img_resolution, gen.img_channels
        )
        self.G.load_state_dict(gen.state_dict())

        self.G = self.G.eval().to(device_name)
        for param in self.G.parameters():
            param.requires_grad = False

    def forward_latent(self, ws):
        return self.G.forward_with_latent(ws)

    def forward_sample(self, z):
        c = torch.zeros(z.shape[0], 0).to(device_name)
        return self.G.forward_get_latent(z, c)

    def forward_image(self, z):
        c = torch.zeros(z.shape[0], 0).to(device_name)
        return self.G(z, c)


"""
opts, sd, latent_avg = get_opts_and_encoder_sd()
print(latent_avg.shape)
enc = ImageEncoder(opts, sd, 256).to(device_name)
img = torch.rand(7, 3, 1024, 1024).to(device_name)
c3, p2, p1 = enc(img)
print(c3.shape, p2.shape, p1.shape)

mapping = LatentMapping(opts, latent_avg).to(device_name)
encoded = mapping((c3, p2, p1))

print(encoded.shape)
"""
