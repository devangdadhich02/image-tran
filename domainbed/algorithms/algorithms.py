# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
from typing import List

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path
import numpy as np
from PIL import Image

from domainbed.algorithms.vae_dg import *
from domainbed import networks
from domainbed.optimizers import get_optimizer


def to_minibatch(x, y):
    minibatches = list(zip(x, y))
    return minibatches


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """

    transforms = {}

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.hparams = hparams

    def update(self, x, y, **kwargs):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def forward(self, x):
        return self.predict(x)

    def new_optimizer(self, parameters):
        optimizer = get_optimizer(
            self.hparams["optimizer"],
            parameters,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        return optimizer

    def clone(self):
        clone = copy.deepcopy(self)
        clone.optimizer = self.new_optimizer(clone.network.parameters())
        clone.optimizer.load_state_dict(self.optimizer.state_dict())

        return clone


class ERM(Algorithm):
    """
    Baseline --- Empirical Risk Minimization (ERM) 
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)


class VAE_DG(Algorithm):
    """
    Our proposed method
    Variational Autoencoders for Domain Generalization
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 1024
        CNN_embed_dim = 256   # latent dim extracted by 2D CNN
        self.network = ResNet_VAE(input_shape, num_classes, hparams, fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, CNN_embed_dim=CNN_embed_dim)
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        
        # Get current step for KL annealing (default to 0 if not provided)
        step = kwargs.get('step', 0)

        loss, recon_loss , KLD_loss, y_loss = self.network.loss_function(all_x, all_y, step=step)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"total_loss": loss.item(),
                "recon_loss": recon_loss.item(),
                "KLD_loss": KLD_loss.item(),
                "y_loss": y_loss.item()}

    def predict(self, x):
        return self.network.classifier(x)

    def save_final_reconstruction(self, batch, save_dir, max_images=8):
        # Ensure save_dir is a Path object and create directory
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # prepare batch (list of env tensors -> single batch)
        x = torch.cat(batch["x"], dim=0)[:max_images]
        print("x shape:", x.shape)

        # move inputs to model device
        model_device = next(self.parameters()).device if len(list(self.parameters())) > 0 else x.device
        if x.device != model_device:
            x = x.to(model_device)

        # inference: encode then decode (deterministic: mu)
        self.network.eval()
        with torch.no_grad():
            mu, logvar = self.network.encode(x)
            z = mu  # deterministic reconstruction for debugging
            recon = self.network.decode(z)  # decoder outputs in [0,1] due to Sigmoid

        # move to CPU and detach
        x_cpu = x.detach().cpu()
        recon_cpu = recon.detach().cpu()
        mu_cpu = mu.detach().cpu()

        # print stats for debugging
        def stats(t):
            return (float(t.min()), float(t.max()), float(t.mean()), float(t.std()))
        print("[recon debug] x min/max/mean/std:", stats(x_cpu))
        print("[recon debug] recon min/max/mean/std:", stats(recon_cpu))
        print("[recon debug] mu mean/std:", float(mu_cpu.mean()), float(mu_cpu.std()))

        # Save original and recon (no ImageNet denorm applied to recon)
        for i in range(x_cpu.size(0)):
            # get the i-th image (should be C,H,W)
            x_i = x_cpu[i]

            # defensive: if x_i somehow has a leading singleton batch dim (1,C,H,W), remove it
            if x_i.dim() == 4 and x_i.size(0) == 1:
                x_i = x_i.squeeze(0)  # now (C,H,W)
            # if it's 2D or other unexpected shape, raise a clear error rather than failing silently
            if x_i.dim() != 3:
                raise RuntimeError(f"Unexpected per-image tensor shape: {tuple(x_i.shape)} (expected (C,H,W))")

            # prepare mean/std shaped for (C,H,W) arithmetic (CPU dtype match)
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=x_i.dtype).view(3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225], dtype=x_i.dtype).view(3, 1, 1)

            # decide whether to un-normalize (heuristic)
            if x_i.min() < -0.5:
                # x_i is normalized (e.g. ImageNet). Undo normalization to get pixel values in [0,1]
                x_vis_t = (x_i * std + mean).clamp(0, 1)  # still (C,H,W)
            else:
                # already in pixel space
                x_vis_t = x_i.clamp(0, 1)

            # convert to HWC numpy array in [0,1]
            x_vis = x_vis_t.permute(1, 2, 0).numpy()  # (H,W,C)

            # recon is already in [0,1] and should be (C,H,W) -> convert safely
            recon_i = recon_cpu[i]
            if recon_i.dim() == 4 and recon_i.size(0) == 1:
                recon_i = recon_i.squeeze(0)
            if recon_i.dim() != 3:
                raise RuntimeError(f"Unexpected recon tensor shape: {tuple(recon_i.shape)} (expected (C,H,W))")
            
            recon_raw = recon_i.clamp(0, 1)                              
            recon_raw_np = recon_raw.permute(1, 2, 0).numpy()           
            recon_raw_uint8 = (recon_raw_np * 255.0).astype("uint8") 
            
            # Paper-quality normalization: preserve vessel contrast
            # Use percentile-based normalization for better vessel visibility
            recon_np = recon_i.permute(1, 2, 0).numpy()
            # Percentile-based normalization (better for medical images)
            p1, p99 = np.percentile(recon_np, (1, 99))
            if p99 > p1:
                recon_normed = np.clip((recon_np - p1) / (p99 - p1 + 1e-8), 0, 1)
            else:
                # Fallback to min/max if percentile range is too small
                recon_min = recon_np.min()
                recon_max = recon_np.max()
                if recon_max > recon_min:
                    recon_normed = (recon_np - recon_min) / (recon_max - recon_min + 1e-8)
                else:
                    recon_normed = recon_np
            recon_vis = np.clip(recon_normed, 0, 1)

            # convert to uint8 pixels for PNG
            orig = (x_vis * 255.0).astype("uint8")
            recon_img = (recon_vis * 255.0).astype("uint8")


            Image.fromarray(orig).save(save_dir / f"{i:02d}_orig.png")
            Image.fromarray(recon_raw_uint8).save(save_dir / f"{i:02d}_recon_unchanged.png")
            Image.fromarray(recon_img).save(save_dir / f"{i:02d}_recon_normalised.png")

            # difference (amplified) — leave it, but explain: if recon ~ constant, diff looks like original
            diff = np.abs(orig.astype(np.int16) - recon_raw_uint8.astype(np.int16)).astype(np.uint8)
            amp = np.clip(diff.astype(np.int16) * 8, 0, 255).astype(np.uint8)
            Image.fromarray(amp).save(save_dir / f"{i:02d}_diff_amp.png")

        # optional latent sensitivity check (helps know if decoder uses z)
        # create a perturbed sample and save the first one
        # creating random noise to see if decoder is using r.
        with torch.no_grad():
            eps = torch.randn_like(mu) * 0.1
            pert = self.network.decode((mu + eps).to(model_device))[: x_cpu.size(0)].detach().cpu()
        pert_vis = (pert[0].clamp(0,1).permute(1,2,0).numpy() * 255).astype("uint8")
        Image.fromarray(pert_vis).save(save_dir / "00_recon_perturbed.png")

        print(f"[✓] Saved {x_cpu.size(0)} reconstructions + diagnostics to {save_dir}")
