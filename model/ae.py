import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from torch import Tensor


class AE(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        # Image size is 64 x 64, latent dim should be 4 x 4 x 128
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=2, padding=1, bias=False),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.Conv2d(16, 1, 3, stride=1, padding=1),
        )

        self.loss_fn = nn.MSELoss()
        self.config = config

    def forward(self, x: Tensor):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def detect_anomaly(self, x: Tensor):
        rec = self(x)
        anomaly_map = torch.abs(x - rec)
        anomaly_score = torch.sum(anomaly_map, dim=(1, 2, 3))
        return {
            'reconstruction': rec,
            'anomaly_map': anomaly_map,
            'anomaly_score': anomaly_score
        }

    def training_step(self, batch: Tensor, batch_idx):
        x = batch
        recon = self(x)
        loss = self.loss_fn(recon, x)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch: Tensor, batch_idx):
        x = batch
        recon = self(x)
        loss = self.loss_fn(recon, x)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        print(self.config)
        return optim.Adam(self.parameters(), lr=self.config['lr'])
