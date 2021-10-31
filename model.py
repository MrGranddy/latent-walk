import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from loading import load_generator
import matplotlib.pyplot as plt

from efficientnet_pytorch import EfficientNet
from facenet_pytorch import MTCNN, InceptionResnetV1

from training.networks_stylegan2 import Generator

import pickle
from PIL import Image
import shutil
import uuid
import numpy as np

device_name = "cuda:0"


def to_image(tensor, img_res):
    tensor = torch.clamp(tensor, -1, 1)
    tensor = (tensor + 1) / 2
    tensor.clamp(0, 1)
    tensor = F.interpolate(tensor, size=(img_res, img_res), mode="bilinear", align_corners=True)
    return (255 * tensor.cpu().detach()).to(torch.uint8).permute(0, 2, 3, 1).numpy()


class ImageEncoder(nn.Module):
    def __init__(self, input_size=256):
        super(ImageEncoder, self).__init__()

        self.encoder = InceptionResnetV1(pretrained='vggface2').eval()
        self.encoder.last_bn = nn.Identity()
        self.encoder.logits = nn.Identity()

        #self.encoder = EfficientNet.from_pretrained('efficientnet-b7')
        #self.encoder._fc = nn.Identity()

        self.encoder = self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.input_size = (input_size, input_size)

    def forward(self, x):

        x = F.interpolate(x, size=self.input_size)
        return self.encoder(x)

class LatentMapping(nn.Module):
    def __init__(self, input_size=512):
        super(LatentMapping, self).__init__()

        self.relu = nn.LeakyReLU(0.2)

        self.linears = nn.ModuleList([
            nn.Linear(input_size, 2048),
            nn.Linear(2048, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 512),
            nn.Linear(512, 512)
        ])


    def forward(self, x):

        for linear in self.linears:
            x = linear(x)
            x = self.relu(x)

        return self.final_fc(x).unsqueeze(1).repeat(1, 18, 1)


class Discriminator(nn.Module):
    def __init__(self, input_size=512 * 18):
        super(Discriminator, self).__init__()

        self.linears = nn.ModuleList([
            nn.Linear(input_size, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 64),
        ])

        self.final_fc = nn.Linear(64, 1)

        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):

        x = x.view(x.shape[0], -1) # değişcek bu şimdi idareten denemelik

        for linear in self.linears:
            x = linear(x)
            x = self.relu(x)

        return self.final_fc(x)

"""
class StyleGan(nn.Module):
    def __init__(self):
        super(StyleGan, self).__init__()
        self.gen = load_generator(
            args={
                'resolution': {'horse': 256, 'church': 256, 'car': 512, 'ffhq': 1024}['ffhq'],
            },
            #G_weights='pretrained/StyleGAN2/stylegan2-car-config-f.pt'
            G_weights='pretrained/StyleGAN2/stylegan2-ffhq-config-f.pt'
        )

        self.gen = self.gen.eval().to(device_name)
        for param in self.gen.parameters():
            param.requires_grad = False


    def forward_latent(self, x):
        return self.gen([x], input_is_latent=True)[0]

    def forward_sample(self, x):
        return self.gen.forward_latent([x])[0]

    def forward_image(self, x):
        return self.gen([x])[0]
"""

class StyleGan(nn.Module):
    def __init__(self):
        super(StyleGan, self).__init__()
        with open("pretrained/StyleGAN2/stylegan2-ffhq-1024x1024.pkl", "rb") as f:
            gen = pickle.load(f)["G_ema"]
        
        self.G = Generator( gen.z_dim, gen.c_dim, gen.w_dim, gen.img_resolution, gen.img_channels )
        self.G.load_state_dict( gen.state_dict() )

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
print(G.z_dim, G.c_dim)
z = torch.from_numpy(np.random.randn(1, G.z_dim))
c = torch.zeros(1, 0)
img = to_image(G(z, c), 512)

Image.fromarray(img[0, ...]).save("demo.png")
"""

"""
encoder = ImageEncoder()
inp = torch.randn(1, 3, 1024, 1024)
out = encoder(inp)
print(out.shape) # 2560
"""

"""
z = z.to(device_name).double()
gan = StyleGan()
img = gan.forward_image(z)

Image.fromarray( to_image(img, 1024)[0, ...] ).save("demo2.png")
"""



"""
gan = StyleGan().to(device_name)
encoder = ImageEncoder().to(device_name)
mapping = LatentMapping().to(device_name)

z = torch.randn(2, 2560).to(device_name)
w = mapping(z)
img = gan.forward_latent(w)

#plt.imshow( to_image(img, 1024)[0, ...] )
#plt.savefig("demo.png")

print(gan.forward_sample(w).shape) # torch.Size([2, 512])
print(img.shape) # torch.Size([2, 3, 1024, 1024])
"""