import os
import json
import torch

from models.gan_load import make_style_gan2

device_name = "cuda:0"


def load_generator(args, G_weights, shift_in_w=False):
    G = make_style_gan2(args["resolution"], G_weights, shift_in_w)

    return G
