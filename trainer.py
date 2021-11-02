import os
import shutil
import json
import torch
from torch import nn
import datetime
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import argparse
from model import (
    StyleGan,
    ImageEncoder,
    Discriminator,
    LatentMapping,
    get_opts_and_encoder_sd,
)

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# Sadakallahülazim Bismillahirrahmanirrahim

n_steps = 100000
edge_value = 6.0
min_value = 0.5

device_name = "cuda:0"


def scheduler(optim, index, init_lr):
    for param_group in optim.param_groups:
        param_group["lr"] = init_lr * (0.9998 ** (index))


def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical", fontsize=4)
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)


def prepare_folder(x):
    if os.path.exists(x) and os.path.isdir(x):
        shutil.rmtree(x)
    os.makedirs(x)


def sample_zs(batch_size, dim_z, mean=0.0, std=1.0):
    zs = torch.empty([batch_size] + [dim_z], device=device_name).normal_(
        mean=mean, std=std
    )
    return zs


def to_image(tensor, img_res):
    tensor = torch.clamp(tensor, -1, 1)
    tensor = (tensor + 1) / 2
    #tensor.clamp(0, 1)
    tensor = F.interpolate(
        tensor, size=(img_res, img_res), mode="bilinear", align_corners=True
    )
    return (255 * tensor.cpu().detach()).to(torch.uint8).permute(0, 2, 3, 1).numpy()


def aug(tensor):

    tensor.clamp(-1, 1)
    return tensor


class Trainer:
    def __init__(self):
        a = datetime.datetime.now()
        self.exp_name = "{}_{}_{}-{}_{}_{}_{}".format(
            a.year, a.month, a.day, a.hour, a.minute, a.second, a.microsecond
        )
        self.main_path = "results/{}_{}_{}-{}_{}_{}_{}".format(
            a.year, a.month, a.day, a.hour, a.minute, a.second, a.microsecond
        )
        self.checkpoint_dir = "{}/checkpoints".format(self.main_path)
        self.loss_dir = "{}/losses".format(self.main_path)
        self.results_dir = "{}/visual_results".format(self.main_path)
        self.grads_dir = "{}/grads".format(self.main_path)
        self.codes_dir = "{}/codes".format(self.main_path)
        os.mkdir(self.main_path)
        os.mkdir(self.checkpoint_dir)
        os.mkdir(self.loss_dir)
        os.mkdir(self.results_dir)
        os.mkdir(self.grads_dir)
        os.mkdir(self.codes_dir)

        for f_name in os.listdir("."):
            if f_name[-3:] == ".py":
                shutil.copyfile(f_name, os.path.join(self.codes_dir, f_name))

        self.losses = {"id": [], "discriminator": [], "generator": []}

        opts, sde, sds, lavg = get_opts_and_encoder_sd()

        self.gan = StyleGan().eval().to(device_name)
        self.encoder = ImageEncoder(opts, sde, 256).eval().to(device_name)
        self.mapping = LatentMapping(opts, sds, lavg).train().to(device_name)
        self.discriminator = Discriminator().train().to(device_name)

        self.dim_gan_z = 512
        self.dim_enc_z = 512
        self.dim_w = 512
        self.img_res = 128

        self.batch_size = 24

        self.mapping_optimizer = torch.optim.Adam(
            list(self.mapping.parameters()) + list(self.encoder.parameters()),
            lr=2.e-4
        )
        self.discriminator_optimizer = torch.optim.SGD(
            self.discriminator.parameters(),
            lr=1.e-3,
            momentum=0.9,
        )

        self.best_model = None
        self.best_loss = 9999

        self.bce_loss = nn.BCELoss()

    def create_recon_grid(self, index):

        batch_size = 8
        part_size = 2
        num_parts = batch_size // part_size

        # prepare_folder( os.path.join(self.results_dir, str(index)) )

        with torch.set_grad_enabled(False):

            self.mapping.eval()
            self.encoder.eval()
            self.gan.eval()

            self.mapping.zero_grad()
            self.gan.zero_grad()
            self.encoder.zero_grad()

            zs = sample_zs(batch_size, self.dim_gan_z)
            zs = zs.view(num_parts, part_size, -1)
            grid = np.zeros(
                (num_parts, part_size, 2, self.img_res, self.img_res, 3), dtype="uint8"
            )

            for part_idx in range(num_parts):

                images = self.gan.forward_image(zs[part_idx, ...])
                images = aug(images)
                encodings = self.encoder(images)
                ws = self.mapping(encodings)
                recon_images = self.gan.forward_latent(ws)

                img = to_image(recon_images, self.img_res).reshape(
                    part_size, self.img_res, self.img_res, 3
                )
                grid[part_idx, :, 1, :, :, :] = img
                img = to_image(images, self.img_res).reshape(
                    part_size, self.img_res, self.img_res, 3
                )
                grid[part_idx, :, 0, :, :, :] = img

            grid = np.moveaxis(grid, [0, 1, 2, 3, 4, 5], [0, 1, 3, 2, 4, 5]).reshape(
                num_parts, part_size * self.img_res, 2 * self.img_res, 3
            )
            grid = grid.reshape(batch_size * self.img_res, 2 * self.img_res, 3)

            # Image.fromarray( grid ).save( os.path.join(self.results_dir, str(index), "grid.png") )
            Image.fromarray(grid).save(os.path.join(self.results_dir, "%d.png" % index))

    def save_plot(self, index):

        fig, ax = plt.subplots()
        for key, val in self.losses.items():
            ax.plot(val, label=key)
        ax.legend()
        plt.savefig(os.path.join(self.loss_dir, "loss_%d.png" % index), dpi=300)
        plt.cla()

    def save_model(self, index):

        torch.save(
            {
                "step": index,
                "mapping_state_dict": self.mapping.state_dict(),
                "discriminator_state_dict": self.discriminator.state_dict(),
            },
            os.path.join(
                self.checkpoint_dir, "%s_step_%d.pth" % (self.exp_name, index + 1)
            ),
        )

    def save_grads(self, index):
        plt.figure(figsize=(8, 8))
        plot_grad_flow(self.discriminator.named_parameters())
        plt.savefig(
            os.path.join(self.grads_dir, "%d_discriminator.png" % index), dpi=200
        )
        plt.close("all")

        plt.figure(figsize=(8, 8))
        plot_grad_flow(self.mapping.named_parameters())
        plt.savefig(os.path.join(self.grads_dir, "%d_mapping.png" % index), dpi=200)
        plt.close("all")

    def train_step(self, index):

        with torch.set_grad_enabled(True):

            self.mapping.train()
            self.mapping.zero_grad()

            self.discriminator.train()
            self.discriminator.zero_grad()

            self.gan.zero_grad()
            self.encoder.zero_grad()

            real_zs = sample_zs(self.batch_size // 2, self.dim_gan_z)
            real_ws = self.gan.forward_sample(real_zs)

            one_label = torch.ones(self.batch_size // 2, device=device_name)
            real_pred = torch.sigmoid(self.discriminator(real_ws).squeeze(1))

            zero_label = torch.zeros(self.batch_size // 2, device=device_name)
            fake_zs = sample_zs(self.batch_size // 2, self.dim_gan_z)
            fake_images = self.gan.forward_image(fake_zs)

            fake_encodings = self.encoder(fake_images)

            errD_real = self.bce_loss(real_pred, one_label)
            errD_real.backward()

            fake_ws = self.mapping(fake_encodings)
            fake_pred = torch.sigmoid(self.discriminator(fake_ws).squeeze(1))

            errD_fake = self.bce_loss(fake_pred, zero_label)
            errD_fake.backward()

            discriminator_loss = (errD_fake + errD_real) / 2
            self.discriminator_optimizer.step()

            loss = discriminator_loss
            tot_gen_loss = 0

            for _ in range(2):
                self.mapping.zero_grad()
                self.encoder.zero_grad()

                fake_encodings = self.encoder(fake_images)
                fake_ws = self.mapping(fake_encodings)
                remapped_images = self.gan.forward_latent(fake_ws)
                remapped_encodings = self.encoder(remapped_images)
                fake_pred = torch.sigmoid(self.discriminator(fake_ws).squeeze(1))

                fc3, fp2, fp1 = fake_encodings
                rc3, rp2, rp1 = remapped_encodings

                id_loss = (
                    torch.mean(torch.abs(fc3 - rc3))
                    + torch.mean(torch.abs(fp2 - rp2))
                    + torch.mean(torch.abs(fp1 - rp1))
                ) * 10

                gen_loss = self.bce_loss(fake_pred, one_label)
                mapping_loss = id_loss + gen_loss

                mapping_loss.backward()
                self.mapping_optimizer.step()

                loss += mapping_loss
                tot_gen_loss += gen_loss

            tot_gen_loss /= 2

            if mapping_loss < self.best_loss:
                print(
                    "Best Model Yet Achieved -> Prev: %6f, Now: %.6f"
                    % (self.best_loss, mapping_loss)
                )
                self.best_loss = float(mapping_loss.detach().cpu().data)
                self.best_model = self.mapping.state_dict()

            if (index + 1) % 100 == 0:
                self.save_grads(index + 1)

            self.losses["id"].append(float(id_loss.detach().cpu().data))
            self.losses["discriminator"].append(
                float(discriminator_loss.detach().cpu().data)
            )
            self.losses["generator"].append(float(tot_gen_loss.detach().cpu().data))

        print(
            "Step: %d/%d ID Loss: %.6f, GEN Loss: %.6f, DISC Loss: %.6f"
            % (
                index + 1,
                n_steps,
                id_loss.detach().cpu().data,
                tot_gen_loss.detach().cpu().data,
                discriminator_loss.detach().cpu().data,
            )
        )

    def train(self):

        for i in range(n_steps):
            self.train_step(i)

            if (i + 1) % 1000 == 0:
                self.save_model(i)
            if (i + 1) % 100 == 0:
                self.save_plot(i + 1)
            if (i + 1) % 10 == 0:
                self.create_recon_grid(i + 1)

        torch.save(
            {"mapping": self.best_model}, os.path.join(self.main_path, "best_model.pth")
        )


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
