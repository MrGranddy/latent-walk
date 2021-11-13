
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

from psp_models.psp import pSp
from psp_models.encoders.psp_encoders import GradualStyleEncoder, GradualStyleBlock
from options.test_options import TestOptions

from training.networks_stylegan2 import Generator

from torchvision.models.resnet import BasicBlock


import pickle
from PIL import Image
import shutil
import uuid
import numpy as np
import math
from collections import OrderedDict

device_name = "cuda:0"

# Bismillahirrahmanirrahim


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

    state_dict_encoder = OrderedDict()
    state_dict_styles = OrderedDict()

    for key, val in ckpt["state_dict"].items():
        if "encoder" in key:
            if "styles" not in key:
                state_dict_encoder[".".join(key.split(".")[1:])] = val
            else:
                state_dict_styles[".".join(key.split(".")[2:])] = val

    opts = ckpt["opts"]
    opts.update(vars(test_opts))
    opts["learn_in_w"] = False
    opts["output_size"] = 1024
    opts["n_styles"] = int(math.log(opts["output_size"], 2)) * 2 - 2

    return opts, state_dict_encoder, state_dict_styles, ckpt["latent_avg"]


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
        x = F.interpolate(x, size=self.input_size, mode="bilinear")
        return self.encoder(x)


class LatentMapping(nn.Module):
    def __init__(self, opts, styles_sd, latent_avg):
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

        self.styles.load_state_dict(styles_sd)

        for style in self.styles:
            for param in style.parameters():
                param.requires_grad = False

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

        return out


class Walker(nn.Module):
    def __init__(self, num_walks):
        super(Walker, self).__init__()

        self.num_walks = num_walks
        self.log_mat_half = nn.Parameter(torch.randn([num_walks, 512 * 4]), True)


    def forward(self, x, w, eps): # bs, 512
        bs = x.shape[0]
        #walks = torch.matrix_exp(self.log_mat_half - self.log_mat_half.transpose(0, 1))[w]
        walks = self.log_mat_half[w]
        walks = walks * eps.view(bs, 1) * (4/22)
        walked = x
        walked[:, 7, :] += walks[:, :512]
        walked[:, 8, :] += walks[:, 512:512*2]
        walked[:, 9, :] += walks[:, 512*2:512*3]
        walked[:, 10, :] += walks[:, 512*3:]

        return walked


# TODO: BAĞIMSIZ YAAAAAP AAAA BAĞIMSIZ OLSUUUUN 20 TANE SEEEÇ
# TODO: FARK BESSSLEEEEE
# TODO: GERİ SOK FEATURECUYA O DAHA İYİ VALLA BAK
class WalkClassifier(nn.Module):
    def __init__(self, num_walks):
        super(WalkClassifier, self).__init__()

        self.num_walks = num_walks
        self.input_size = 256

        self.layers = nn.ModuleList([])
    

        for _ in range(5):
            downsample = nn.Sequential(
                nn.Conv2d(512, 512, 1, stride=2), nn.BatchNorm2d(512)
            )
            self.layers.append(
                nn.Sequential(
                    BasicBlock(512, 512, stride=1, downsample=None),
                    BasicBlock(512, 512, stride=2, downsample=downsample),
                )
            )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifying_head = nn.Linear(512, self.num_walks)
        self.regression_head = nn.Linear(512, 1)

        #self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, y):

        bs = x.shape[0]

        # xy = torch.cat([x, y], dim=1) # 1024
        xy = x - y

        for layer in self.layers:
            xy = layer(xy)

        xy = self.avgpool(xy)
        xy = xy.view(bs, -1)

        return self.classifying_head(xy), self.regression_head(xy)


class Discriminator(nn.Module):
    def __init__(self, input_size=512 * 18):
        super(Discriminator, self).__init__()

        self.linears = nn.ModuleList(
            [
                nn.Linear(input_size, 512),
                nn.Linear(512, 256),
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
model = WalkClassifier(20).to(device_name)
inp1 = torch.randn(32, 512, 64, 64).to(device_name)
inp2 = torch.randn(32, 512, 64, 64).to(device_name)
a, b = model(inp1, inp2)

print(a.shape, b.shape)

model = Walker(20).to(device_name)
walks = torch.randint(low=0, high=20, size=(32,)).to(device_name)
eps = (torch.rand(32) - 0.5).to(device_name)
p1 = torch.randn(32, 512, 64, 64).to(device_name)
print(model(p1, walks, eps).shape)
"""

"""
opts, sde, sds, lavg = get_opts_and_encoder_sd()
encoder = ImageEncoder(opts, sde, 256).eval().to(device_name)
mapper = LatentMapping(opts, sds, lavg).eval().to(device_name)
gan = StyleGan().eval().to(device_name)

path = "bby14_a.png"

img = np.array(Image.open(path))[..., :3].astype("float32") / 255
img -= 0.5
img /= 0.5
img = torch.tensor(img).unsqueeze(0).to(device_name)
img = img.transpose(3,2)
img = img.transpose(2,1)
encoding = encoder(img)
diff_ws = mapper(encoding)
ws = diff_ws + mapper.latent_avg.unsqueeze(0)

img = gan.forward_latent(ws)
remapped = to_image(img, 1024)[0, ...]
org_img = np.array(Image.open(path))[..., :3].astype("uint8")
final = np.concatenate( (org_img, remapped), axis=1 )
Image.fromarray(final).save("demo.png")

"""

"""
for i in range(10):
    img = gan.forward_latent(ws + torch.randn(ws.shape, device=device_name) * 0.3)
    Image.fromarray(to_image(img, 1024)[0, ...]).save("demo%d.png" % i)
"""

"""
classifier = WalkClassifier(20)
x, y = torch.randn(7, 3, 256, 256), torch.randn(7, 3, 256, 256)
a, b = classifier(x, y)

print(a.shape, b.shape)
"""

"""
opts, sd, latent_avg = get_opts_and_encoder_sd()

w = latent_avg.unsqueeze(0).to(device_name)
gan = StyleGan().to(device_name)
img = gan.forward_latent(w)
Image.fromarray(to_image(img, 1024)[0, ...]).save("demo.png")
"""
"""
opts, sde, sds = get_opts_and_encoder_sd()

enc = ImageEncoder(opts, sde, 256).to(device_name)
img = torch.rand(7, 3, 1024, 1024).to(device_name)
c3, p2, p1 = enc(img)
print(c3.shape, p2.shape, p1.shape)

mapping = LatentMapping(opts, sds).to(device_name)
encoded = mapping((c3, p2, p1))

print(encoded.shape)
"""
