import json
import numpy as np
import torch
from torch import nn
from models.gan_with_shift import gan_with_shift

try:
    from models.StyleGAN2.model import Generator as StyleGAN2Generator
    from models.StyleGAN2.model import Discriminator as StyleGAN2Discriminator
except Exception as e:
    print('StyleGAN2 load fail: {}'.format(e))

device_name = "cuda:0"

def make_style_gan2(size, weights, shift_in_w=True):
    G = StyleGAN2Generator(size, 512, 8).to(device_name)
    #D = StyleGAN2Discriminator(size)

    G.load_state_dict(torch.load(weights, map_location=device_name)['g_ema'])
    #D.load_state_dict(torch.load(weights, map_location=device_name)['d'])
    G.to(device_name).eval()
    #D.to(device_name).eval()

    return G
