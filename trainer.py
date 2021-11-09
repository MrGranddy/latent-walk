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
    Walker,
    WalkClassifier,
    get_opts_and_encoder_sd,
)

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# Sadakallahülazim Bismillahirrahmanirrahim

n_steps = 100000
edge_value = 3.0
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


def sample_walks(batch_size, num_walks):

    eps = torch.rand(batch_size) * (edge_value - min_value) + min_value
    eps[torch.rand(batch_size) < 0.5] *= -1

    walks = torch.randint(low=0, high=num_walks, size=(batch_size,))

    return walks.to(device_name), eps.to(device_name)


def to_image(tensor, img_res):
    tensor = torch.clamp(tensor, -1, 1)
    tensor = (tensor + 1) / 2
    # tensor.clamp(0, 1)
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
        self.walks_dir = "{}/walk_results".format(self.main_path)
        self.grads_dir = "{}/grads".format(self.main_path)
        self.codes_dir = "{}/codes".format(self.main_path)
        os.mkdir(self.main_path)
        os.mkdir(self.checkpoint_dir)
        os.mkdir(self.loss_dir)
        os.mkdir(self.results_dir)
        os.mkdir(self.walks_dir)
        os.mkdir(self.grads_dir)
        os.mkdir(self.codes_dir)

        for f_name in os.listdir("."):
            if f_name[-3:] == ".py":
                shutil.copyfile(f_name, os.path.join(self.codes_dir, f_name))

        self.losses = {"cls": [], "reg": [], "dist": [], "p2_reg": [], "c3_reg": []}

        opts, sde, sds, lavg = get_opts_and_encoder_sd()

        self.dim_gan_z = 512
        self.dim_enc_z = 512
        self.dim_w = 512
        self.img_res = 128
        self.batch_size = 6
        self.num_walks = 20

        self.gan = StyleGan().eval().to(device_name)
        self.encoder = ImageEncoder(opts, sde, 256).eval().to(device_name)
        self.mapping = LatentMapping(opts, sds, lavg).eval().to(device_name)
        self.walker = Walker(self.num_walks).train().to(device_name)
        self.classifier = WalkClassifier(self.num_walks).train().to(device_name)

        self.walk_optimizer = torch.optim.Adam(
            self.walker.parameters(),
            lr=1.0e-4,
        )

        self.classifier_optimizer = torch.optim.Adam(
            self.classifier.parameters(),
            lr=1.0e-4,
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
                diff_ws = self.mapping(encodings)
                ws = diff_ws + self.mapping.latent_avg.unsqueeze(0)
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

    def create_walk_grid(self, index):

        part_size = 2
        num_parts = self.num_walks // part_size

        path = os.path.join(self.walks_dir, str(index))
        prepare_folder(path)

        with torch.set_grad_enabled(False):

            self.mapping.zero_grad()
            self.encoder.zero_grad()
            self.gan.zero_grad()
            self.walker.zero_grad()
            self.classifier.zero_grad()

            self.walker.eval()
            self.classifier.eval()

            zs = sample_zs(1, self.dim_gan_z).to(device_name)
            eps = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]).to(device_name)
            walks = torch.arange(self.num_walks).to(device_name)

            eps_shape = eps.shape[0]

            org_img = self.gan.forward_image(zs)
            c3, p2, p1 = self.encoder(aug(org_img))

            org_img = to_image(org_img, 256)
            Image.fromarray(org_img[0, ...]).save(os.path.join(path, "org_img.png"))

            zs = zs.view(1, 1, self.dim_gan_z).repeat(self.num_walks, eps_shape, 1)
            zs = zs.view(num_parts, part_size, eps_shape, self.dim_gan_z)

            grid = np.zeros(
                (num_parts, part_size, eps_shape, self.img_res, self.img_res, 3),
                dtype="uint8",
            )

            walks = walks.view(num_parts, part_size, 1).repeat(1, 1, eps_shape)
            eps = eps.view(1, -1).repeat(part_size, 1)

            c3 = c3.repeat(eps_shape * part_size, 1, 1, 1)
            p2 = p2.repeat(eps_shape * part_size, 1, 1, 1)
            p1 = p1.repeat(eps_shape * part_size, 1, 1, 1)

            for part_idx in range(num_parts):

                walked_p1 = self.walker(p1, walks[part_idx, ...].view(-1), eps.view(-1))
                diff_ws = self.mapping((c3, p2, walked_p1))
                ws = diff_ws + self.mapping.latent_avg.unsqueeze(0)
                walked_images = self.gan.forward_latent(ws)
                img = to_image(walked_images, self.img_res).reshape(
                    part_size, eps_shape, self.img_res, self.img_res, 3
                )
                grid[part_idx, ...] = img

            grid = np.moveaxis(grid, [0, 1, 2, 3, 4, 5], [0, 1, 3, 2, 4, 5]).reshape(
                num_parts, part_size * self.img_res, eps_shape * self.img_res, 3
            )

            for part_idx in range(num_parts):
                Image.fromarray(grid[part_idx, ...]).save(
                    os.path.join(path, str(part_idx) + ".png")
                )

    def save_plot(self, index):

        fig, ax = plt.subplots()
        for key, val in self.losses.items():
            ax.plot(val, label=key)
        ax.legend()
        plt.savefig(os.path.join(self.loss_dir, "loss_%d.png" % index), dpi=300)
        plt.cla()

    def save_model(self, index):

        torch.save(
            {"step": index, "walker": self.walker.state_dict()},
            os.path.join(
                self.checkpoint_dir, "%s_step_%d.pth" % (self.exp_name, index + 1)
            ),
        )

    def save_grads(self, index):

        models = [("classifier", self.classifier)]

        for name, model in models:
            plt.figure(figsize=(8, 8))
            plot_grad_flow(model.named_parameters())
            plt.savefig(
                os.path.join(self.grads_dir, "%d_%s.png" % (index, name)), dpi=200
            )
            plt.close("all")

        plt.figure(figsize=(8, 8))
        plt.imshow(self.walker.log_mat_half.grad.detach().cpu().numpy())
        plt.savefig(
            os.path.join(self.grads_dir, "%d_%s.png" % (index, "walker")), dpi=200
        )
        plt.close("all")

    def train_step(self, index):

        with torch.set_grad_enabled(True):

            self.mapping.zero_grad()
            self.encoder.zero_grad()
            self.gan.zero_grad()
            self.walker.zero_grad()
            self.classifier.zero_grad()

            self.walker.train()
            self.classifier.train()

            zs = sample_zs(self.batch_size, self.dim_gan_z)
            org_images = self.gan.forward_image(zs)

            c3, p2, p1 = self.encoder(aug(org_images))

            walks, eps = sample_walks(self.batch_size, self.num_walks)
            walked_p1 = self.walker(p1, walks, eps)
            diff_ws = self.mapping((c3, p2, walked_p1))
            ws = diff_ws + self.mapping.latent_avg.unsqueeze(0)

            walked_images = self.gan.forward_latent(ws)
            c3w, p2w, p1w = self.encoder(aug(walked_images))

            cls_out, reg_out = self.classifier(walked_p1, p1w)

            cls_loss = F.cross_entropy(cls_out, walks)
            reg_loss = torch.mean((reg_out - (eps / edge_value)) ** 2) * 1.0e-1
            w_reg_loss = torch.mean(torch.abs(diff_ws)) * 1.0e-3
            c3_reg = torch.mean( (c3 - c3w) ** 2 )
            p2_reg = torch.mean( (p2 - p2w) ** 2 )

            loss = cls_loss + reg_loss + w_reg_loss + c3_reg + p2_reg
            loss.backward()

            self.walk_optimizer.step()
            self.classifier_optimizer.step()

            if cls_loss < self.best_loss:
                print(
                    "Best Model Yet Achieved -> Prev: %6f, Now: %.6f"
                    % (self.best_loss, cls_loss)
                )
                self.best_loss = float(cls_loss.detach().cpu().data)
                self.best_model = self.walker.state_dict()

            if (index + 1) % 1000 == 0:
                self.save_grads(index + 1)

            self.losses["cls"].append(float(cls_loss.detach().cpu().data))
            self.losses["reg"].append(float(reg_loss.detach().cpu().data))
            self.losses["dist"].append(float(w_reg_loss.detach().cpu().data))
            self.losses["p2_reg"].append(float(c3_reg.detach().cpu().data))
            self.losses["c3_reg"].append(float(c3_reg.detach().cpu().data))

        print(
            "Step: %d/%d CLS Loss: %.6f, REG Loss: %.6f, DIST Loss: %.6f, C3: %.6f, P2: %.6f"
            % (
                index + 1,
                n_steps,
                cls_loss.detach().cpu().data,
                reg_loss.detach().cpu().data,
                w_reg_loss.detach().cpu().data,
                c3_reg.detach().cpu().data,
                c3_reg.detach().cpu().data
            )
        )

    def train(self):

        for i in range(n_steps):
            self.train_step(i)

            if (i + 1) % 1000 == 0:
                self.save_model(i)
            if (i + 1) % 1000 == 0:
                self.save_plot(i + 1)
            if (i + 1) % 1000 == 0:
                self.create_recon_grid(i + 1)
            if (i + 1) % 1000 == 0:
                self.create_walk_grid(i + 1)

        torch.save(
            {"mapping": self.best_model}, os.path.join(self.main_path, "best_model.pth")
        )


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
