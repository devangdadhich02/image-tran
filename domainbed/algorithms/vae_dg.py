import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from domainbed import networks

"""
This code contains the VAE architecture from the project Variational Autoencoder (VAE) + Transfer learning (ResNet + VAE)
https://github.com/hsinyilin19/ResNetVAE/tree/master
"""

def conv2D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv2D
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int))
    return outshape

def convtrans2D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv2D
    outshape = ((img_size[0] - 1) * stride[0] - 2 * padding[0] + kernel_size[0],
                (img_size[1] - 1) * stride[1] - 2 * padding[1] + kernel_size[1])
    return outshape

class ResNet_VAE(nn.Module):
    def __init__(self, input_shape, num_classes, hparams, fc_hidden1=1024, fc_hidden2=768, CNN_embed_dim=256):
        super(ResNet_VAE, self).__init__()
        self.fc_hidden1, self.fc_hidden2, self.CNN_embed_dim = fc_hidden1, fc_hidden2, CNN_embed_dim
        self.num_classes = num_classes
        self.loss_multiplier_y = hparams.loss_multiplier_y
        self.loss_multiplier_kl = hparams.loss_multiplier_kl
        # KL annealing parameters (support both dict and object access for sconf.Config)
        if hasattr(hparams, 'kl_anneal_start'):
            self.kl_anneal_start = hparams.kl_anneal_start
        elif 'kl_anneal_start' in hparams:
            self.kl_anneal_start = hparams['kl_anneal_start']
        else:
            self.kl_anneal_start = 0
            
        if hasattr(hparams, 'kl_anneal_end'):
            self.kl_anneal_end = hparams.kl_anneal_end
        elif 'kl_anneal_end' in hparams:
            self.kl_anneal_end = hparams['kl_anneal_end']
        else:
            self.kl_anneal_end = 10000
            
        if hasattr(hparams, 'kl_anneal_cyclical'):
            self.kl_anneal_cyclical = hparams.kl_anneal_cyclical
        elif 'kl_anneal_cyclical' in hparams:
            self.kl_anneal_cyclical = hparams['kl_anneal_cyclical']
        else:
            self.kl_anneal_cyclical = False
            
        if hasattr(hparams, 'kl_anneal_cycle_length'):
            self.kl_anneal_cycle_length = hparams.kl_anneal_cycle_length
        elif 'kl_anneal_cycle_length' in hparams:
            self.kl_anneal_cycle_length = hparams['kl_anneal_cycle_length']
        else:
            self.kl_anneal_cycle_length = 10000
        self.qy = qy(CNN_embed_dim, self.num_classes)

        # CNN architechtures
        self.ch1, self.ch2, self.ch3, self.ch4 = 16, 32, 64, 128
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)      # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)      # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding

        # encoding components
        self.resnet = networks.Featurizer(input_shape, hparams)
        if hparams.model in ['resnet50', 'resnet152']:
            in_features = 2048
        else:
            in_features = 512

        self.fc1 = nn.Linear(in_features, self.fc_hidden1)
        self.bn1 = nn.BatchNorm1d(self.fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.bn2 = nn.BatchNorm1d(self.fc_hidden2, momentum=0.01)
        # Latent vectors mu and sigma
        self.fc3_mu = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)      # output = CNN embedding latent variables
        self.fc3_logvar = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)  # output = CNN embedding latent variables

        # Sampling vector
        self.fc4 = nn.Linear(self.CNN_embed_dim, self.fc_hidden2)
        self.fc_bn4 = nn.BatchNorm1d(self.fc_hidden2)
        self.fc5 = nn.Linear(self.fc_hidden2, 64 * 4 * 4)
        self.fc_bn5 = nn.BatchNorm1d(64 * 4 * 4)
        self.relu = nn.ReLU(inplace=True)

        # Decoder Architecture - Progressive upsampling to reduce blur
        # Original paper uses 3 layers, but 7x interpolation (32x32->224x224) causes blur
        # Solution: Add 2 more upsampling layers to reduce interpolation to 1.75x
        # Architecture: 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128 -> interpolate 1.75x to 224x224
        self.convTrans6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32, momentum=0.01),
            nn.ReLU(inplace=True),
        )  # 4x4 -> 8x8
        
        self.convTrans7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16, momentum=0.01),
            nn.ReLU(inplace=True),
        )  # 8x8 -> 16x16
        
        self.convTrans8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8, momentum=0.01),
            nn.ReLU(inplace=True),
        )  # 16x16 -> 32x32
        
        self.convTrans9 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4, momentum=0.01),
            nn.ReLU(inplace=True),
        )  # 32x32 -> 64x64
        
        self.convTrans10 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=4, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(3, momentum=0.01),
            nn.Sigmoid()    # y = (y1, y2, y3) \in [0 ,1]^3
        )  # 64x64 -> 128x128, then minimal interpolation (1.75x) to 224x224

    def encode(self, x):
        x = self.resnet(x)  # ResNet
        x = x.view(x.size(0), -1)  # flatten output of conv

        # FC layers
        x = self.bn1(self.fc1(x))
        x = self.relu(x)
        x = self.bn2(self.fc2(x))
        x = self.relu(x)
        mu, logvar = self.fc3_mu(x), self.fc3_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        x = self.relu(self.fc_bn4(self.fc4(z)))
        x = self.relu(self.fc_bn5(self.fc5(x))).view(-1, 64, 4, 4)
        # Progressive upsampling to reduce blur (instead of 7x interpolation)
        x = self.convTrans6(x)   # 4x4 -> 8x8
        x = self.convTrans7(x)   # 8x8 -> 16x16
        x = self.convTrans8(x)   # 16x16 -> 32x32
        x = self.convTrans9(x)   # 32x32 -> 64x64
        x = self.convTrans10(x)  # 64x64 -> 128x128
        # Minimal interpolation (1.75x) instead of 7x - much less blur!
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)        
        x_reconst = self.decode(z)
        y_hat = self.qy(z)
        return x_reconst, mu, logvar, y_hat
 
    def classifier(self, x):
        with torch.no_grad():
            z_q_loc, _ = self.encode(x)
            z = z_q_loc
            logits = self.qy(z)
        return logits

    def compute_kl_weight(self, step):
        """
        Compute KL annealing weight based on step.
        Supports linear and cyclical annealing.
        """
        if self.kl_anneal_cyclical:
            # Cyclical annealing: repeats the annealing schedule
            cycle_step = step % self.kl_anneal_cycle_length
            if cycle_step < self.kl_anneal_start:
                return 0.0
            elif cycle_step >= self.kl_anneal_end:
                return 1.0
            else:
                # Linear interpolation within cycle
                progress = (cycle_step - self.kl_anneal_start) / (self.kl_anneal_end - self.kl_anneal_start)
                return min(progress, 1.0)
        else:
            # Linear annealing: one-time schedule
            if step < self.kl_anneal_start:
                return 0.0
            elif step >= self.kl_anneal_end:
                return 1.0
            else:
                # Linear interpolation
                progress = (step - self.kl_anneal_start) / (self.kl_anneal_end - self.kl_anneal_start)
                return min(progress, 1.0)

    def loss_function(self, x, y, step=0):
        """
        Compute VAE loss with KL annealing.
        
        Args:
            x: Input images (normalized with ImageNet stats)
            y: Class labels
            step: Current training step (for KL annealing)
        
        Returns:
            total_loss, recon_loss, kld_loss, y_loss
        """
        recon_x, mu, logvar, y_hat = self.forward(x)
        
        # Handle normalized inputs: unnormalize x to [0,1] range for comparison
        device = x.device
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        
        # Check if input is normalized (typical range for ImageNet normalized images)
        if x.min() < -0.5:
            # Input is normalized, unnormalize it
            x_unnorm = (x * std + mean).clamp(0, 1)
        else:
            # Input is already in [0,1] range
            x_unnorm = x.clamp(0, 1)
        
        # Reconstruction loss: BCE only (as per original VAE paper)
        # Standard VAE reconstruction loss - use 'mean' reduction for proper per-sample loss
        # This ensures loss scales correctly with batch size and dataset size
        recon_loss = F.binary_cross_entropy(recon_x, x_unnorm, reduction='mean')
        
        # KL divergence loss
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Classification loss
        CE_y = F.cross_entropy(y_hat, y, reduction='mean')
        CE_y = CE_y * y.size(0)  # Scale to match sum reduction
        
        # Compute KL annealing weight
        kl_weight = self.compute_kl_weight(step)
        effective_kl_weight = self.loss_multiplier_kl * kl_weight
        
        # Total loss
        total_loss = recon_loss + effective_kl_weight * KLD + self.loss_multiplier_y * CE_y
        
        return total_loss, recon_loss, effective_kl_weight * KLD, self.loss_multiplier_y * CE_y
        
class qy(nn.Module):
  def __init__(self, latent_dim, num_classes):
    super(qy, self).__init__()
    self.fc1 = nn.Linear(latent_dim, num_classes)
    self.relu = nn.ReLU()
  
  def forward(self, z):
    h = self.relu(z)
    loc_y = self.fc1(h)
    return loc_y
  
