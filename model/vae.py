import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl

from torch import Tensor


class VAE(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        # Image size is 64 x 64, latent dim should be 4 x 4 x 128
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256 * 2, 3, stride=2, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
        )

        self.loss_fn = nn.MSELoss()
        self.config = config
        self.kl_weight = config['kl_weight'] if 'kl_weight' in config else 1.0

    def forward(self, x: Tensor):
        z_ = self.encoder(x)
        mu, logvar = torch.chunk(z_, 2, dim=1)
        z = self.reparameterize(mu, logvar)
        y = self.decoder(z)
        return y, mu, logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        :param mu: Mean of the estimated latent Gaussian
        :param logvar: Standard deviation of the estimated latent Gaussian
        """
        unit_gaussian = torch.randn_like(mu)
        std = torch.exp(0.5 * logvar)
        return unit_gaussian * std + mu

    def compute_loss(self, inp: Tensor, rec: Tensor, mu: Tensor, logvar: Tensor) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param inp: Input image
        :param rec: Reconstructed image
        :param mu: Mean of the estimated latent Gaussian
        :param logvar: Standard deviation of the estimated latent Gaussian
        """
        recon_loss = torch.mean((inp - rec) ** 2)
        kl_loss = torch.mean(-0.5 * (1 + logvar - mu ** 2 - logvar.exp()))
        loss = recon_loss + self.kl_weight * kl_loss
        return loss, recon_loss, kl_loss

    def detect_anomaly(self, x: Tensor):
        y, mu, logvar = self(x)
        # Part 1: Reconstruction loss (64 x 64)
        residual = torch.abs(x - y)
        # Part 2: KL divergence (4 x 4)
        kl_div = torch.mean(-0.5 * (1 + logvar - mu ** 2 - logvar.exp()), dim=1, keepdim=True)
        kl_map = F.interpolate(kl_div, size=x.shape[-2:], mode='bilinear', align_corners=False)
        # Aggregate
        anomaly_map = residual + self.kl_weight * kl_map
        anomaly_score = residual.sum(dim=(1, 2, 3)) + self.kl_weight * kl_div.sum(dim=(1, 2, 3))
        return {
            'reconstruction': y,
            'anomaly_map': anomaly_map,
            'anomaly_score': anomaly_score
        }

    def training_step(self, batch: Tensor, batch_idx):
        x = batch
        y, mu, logvar = self(x)
        loss, recon_loss, kl_loss = self.compute_loss(x, y, mu, logvar)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('train_recon_loss', recon_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('train_kl_loss', kl_loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch: Tensor, batch_idx):
        x = batch
        y, mu, logvar = self(x)
        loss, recon_loss, kl_loss = self.compute_loss(x, y, mu, logvar)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('val_recon_loss', recon_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('val_kl_loss', kl_loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        print(self.config)
        return optim.Adam(self.parameters(), lr=self.config['lr'])
