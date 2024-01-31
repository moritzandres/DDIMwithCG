"""
RA: Bercea, Cosmin I., et al. "Generalizing Unsupervised Anomaly Detection: Towards Unbiased Pathology Screening." Medical Imaging with Deep Learning. 2023.

Code in part from: https://github.com/taldatech/soft-intro-vae-pytorch/blob/main/soft_intro_vae/

"""

# imports
# torch and friends
import torch
import torch.optim as optim
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
# standard
import matplotlib
matplotlib.use('Agg')
import numpy as np
from skimage import exposure
from scipy.ndimage.filters import gaussian_filter
import lpips


"""
Models
"""

class EmbeddingLoss(torch.nn.Module):
    def __init__(self):
        super(EmbeddingLoss, self).__init__()
        self.criterion = torch.nn.MSELoss()
        self.similarity_loss = torch.nn.CosineSimilarity()

    def forward(self, teacher_embeddings, student_embeddings):
        # print(f'LEN {len(output_real)}')
        layer_id = 0
        # teacher_embeddings = teacher_embeddings[:-1]
        # student_embeddings = student_embeddings[3:-1]
        # print(f' Teacher: {len(teacher_embeddings)}, Student: {len(student_embeddings)}')
        for teacher_feature, student_feature in zip(teacher_embeddings, student_embeddings):
            if layer_id == 0:
                total_loss = 0.5 * self.criterion(teacher_feature, student_feature)
            else:
                total_loss += 0.5 * self.criterion(teacher_feature, student_feature)
            total_loss += torch.mean(1 - self.similarity_loss(teacher_feature.view(teacher_feature.shape[0], -1),
                                                         student_feature.view(student_feature.shape[0], -1)))
            layer_id += 1
        return total_loss
        
class ResidualBlock(nn.Module):
    """
    https://github.com/hhb072/IntroVAE
    Difference: self.bn2 on output and not on (output + identity)
    """

    def __init__(self, inc=64, outc=64, groups=1, scale=1.0):
        super(ResidualBlock, self).__init__()

        midc = int(outc * scale)

        if inc is not outc:
            self.conv_expand = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1, padding=0,
                                         groups=1, bias=False)
        else:
            self.conv_expand = None

        self.conv1 = nn.Conv2d(in_channels=inc, out_channels=midc, kernel_size=3, stride=1, padding=1, groups=groups,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(midc)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=midc, out_channels=outc, kernel_size=3, stride=1, padding=1, groups=groups,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(outc)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        if self.conv_expand is not None:
            identity_data = self.conv_expand(x)
        else:
            identity_data = x

        output = self.relu1(self.bn1(self.conv1(x)))
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu2(torch.add(output, identity_data))
        return output


class Encoder(nn.Module):
    def __init__(self, cdim=3, zdim=512, channels=(64, 128, 256, 512, 512, 512), image_size=256, conditional=False,
                 cond_dim=10):
        super(Encoder, self).__init__()
        self.zdim = zdim
        self.cdim = cdim
        self.image_size = image_size
        self.conditional = conditional
        self.cond_dim = cond_dim
        cc = channels[0]
        self.main = nn.Sequential(
            nn.Conv2d(cdim, cc, 5, 1, 2, bias=False),
            nn.BatchNorm2d(cc),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2),
        )

        sz = image_size // 2
        for ch in channels[1:]:
            self.main.add_module('res_in_{}'.format(sz), ResidualBlock(cc, ch, scale=1.0))
            self.main.add_module('down_to_{}'.format(sz // 2), nn.AvgPool2d(2))
            cc, sz = ch, sz // 2

        self.main.add_module('res_in_{}'.format(sz), ResidualBlock(cc, cc, scale=1.0))
        self.conv_output_size = self.calc_conv_output_size()
        num_fc_features = torch.zeros(self.conv_output_size).view(-1).shape[0]
        print("conv shape: ", self.conv_output_size)
        print("num fc features: ", num_fc_features)
        if self.conditional:
            self.fc = nn.Linear(num_fc_features + self.cond_dim, 2 * zdim)
        else:
            self.fc = nn.Linear(num_fc_features, 2 * zdim)

    def calc_conv_output_size(self):
        dummy_input = torch.zeros(1, self.cdim, self.image_size, self.image_size)
        dummy_input = self.main(dummy_input)
        return dummy_input[0].shape

    def forward(self, x, o_cond=None):
        y = x
        embeddings = []
        for layer in self.main:
            y = layer(y)
            if isinstance(layer, nn.AvgPool2d):
                embeddings.append(y)
        y = y.view(x.size(0), -1)
        if self.conditional and o_cond is not None:
            y = torch.cat([y, o_cond], dim=1)
        y = self.fc(y)
        mu, logvar = y.chunk(2, dim=1)
        return mu, logvar, {'embeddings': embeddings}


class Decoder(nn.Module):
    def __init__(self, cdim=3, zdim=512, channels=(64, 128, 256, 512, 512, 512), image_size=256, conditional=False,
                 conv_input_size=None, cond_dim=10):
        super(Decoder, self).__init__()
        self.cdim = cdim
        self.image_size = image_size
        self.conditional = conditional
        cc = channels[-1]
        self.conv_input_size = conv_input_size
        if conv_input_size is None:
            num_fc_features = cc * 4 * 4
        else:
            num_fc_features = torch.zeros(self.conv_input_size).view(-1).shape[0]
        self.cond_dim = cond_dim
        if self.conditional:
            self.fc = nn.Sequential(
                nn.Linear(zdim + self.cond_dim, num_fc_features),
                nn.ReLU(True),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(zdim, num_fc_features),
                nn.ReLU(True),
            )

        sz = 4

        self.main = nn.Sequential()
        for ch in channels[::-1]:
            self.main.add_module('res_in_{}'.format(sz), ResidualBlock(cc, ch, scale=1.0))
            self.main.add_module('up_to_{}'.format(sz * 2), nn.Upsample(scale_factor=2, mode='nearest'))
            cc, sz = ch, sz * 2

        self.main.add_module('res_in_{}'.format(sz), ResidualBlock(cc, cc, scale=1.0))
        self.main.add_module('predict', nn.Conv2d(cc, cdim, 5, 1, 2))
        self.main.add_module('sigmoid', nn.Sigmoid())

    def forward(self, z, y_cond=None):
        z = z.view(z.size(0), -1)
        if self.conditional and y_cond is not None:
            y_cond = y_cond.view(y_cond.size(0), -1)
            z = torch.cat([z, y_cond], dim=1)
        y = self.fc(z)
        y = y.view(z.size(0), *self.conv_input_size)
        y = self.main(y)
        return y


class RA(pl.LightningModule):
    def __init__(self, config):
        super(RA, self).__init__()
        
        self.config = config 
        self.cdim = config['cdim']
        self.zdim = config['zdim']
        self.channels = config['channels']
        self.image_size = config['image_size']
        self.conditional = False
        self.cond_dim = 10
        self.input_size = self.image_size

        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
        self.l_pips_sq = lpips.LPIPS(pretrained=True, net='squeeze', use_dropout=True, eval_mode=True, spatial=True, lpips=True).to(device)

        self.encoder = Encoder(self.cdim, self.zdim, self.channels, self.image_size, conditional=self.conditional, cond_dim=self.cond_dim)

        self.decoder = Decoder(self.cdim, self.zdim, self.channels, self.image_size, conditional=self.conditional,
                               conv_input_size=self.encoder.conv_output_size, cond_dim=self.cond_dim)
        
        self.scale = 1 / (self.input_size ** 2)  # normalize by images size (channels * height * width)
        self.gamma_r = 1e-8
        self.beta_kl = config['beta_kl']
        self.beta_rec = config['beta_rec']
        self.beta_neg = config['beta_neg']

        self.embedding_loss = EmbeddingLoss()
        self.loss_fn = nn.MSELoss()
        self.automatic_optimization = False



    def forward(self, x, o_cond=None, deterministic=False):
        if self.conditional and o_cond is not None:
            mu, logvar, embed_dict = self.encode(x, o_cond=o_cond)
            if deterministic:
                z = mu
            else:
                z = reparameterize(mu, logvar)
            y = self.decode(z, y_cond=o_cond)
        else:
            mu, logvar, embed_dict = self.encode(x)
            if deterministic:
                z = mu
            else:
                z = reparameterize(mu, logvar)
            y = self.decode(z)

        return y, {'x_rec': y, 'z_mu': mu, 'z_logvar': logvar,'z': z, 'embeddings': embed_dict['embeddings']}
        
    def detect_anomaly(self, x, o_cond=None, deterministic=False):
        x_rec, x_rec_dict = self.forward(x, o_cond, deterministic)
        anomaly_maps, anomaly_scores = self.compute_anomaly(x, x_rec)

        return {'reconstruction': x_rec, 'anomaly_map': anomaly_maps, 'anomaly_score': anomaly_scores}


    def training_step(self, batch: Tensor, batch_idx):
        x = batch
        optimizer_e, optimizer_d = self.optimizers()
        device = x.get_device()
        b, c, w, h = x.shape

        noise_batch = torch.randn(size=(b, self.zdim)).to(device)
        real_batch = x.to(device)

        # =========== Update E ================
        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.decoder.parameters():
            param.requires_grad = False

        fake = self.sample(noise_batch)
        real_mu, real_logvar, anomaly_embeddings = self.encode(real_batch)
        z = reparameterize(real_mu, real_logvar)
        rec = self.decoder(z)

        _, _, healthy_embeddings = self.encode(rec.detach())

        loss_emb = self.embedding_loss(anomaly_embeddings['embeddings'], healthy_embeddings['embeddings'])

        loss_rec = calc_reconstruction_loss(real_batch, rec, loss_type="mse", reduction="mean")
        lossE_real_kl = calc_kl(real_logvar, real_mu, reduce="mean")
        rec_rec, z_dict = self(rec.detach(), deterministic=False)
        rec_mu, rec_logvar, z_rec = z_dict['z_mu'], z_dict['z_logvar'], z_dict['z']
        rec_fake, z_dict_fake = self(fake.detach(), deterministic=False)
        fake_mu, fake_logvar, z_fake = z_dict_fake['z_mu'], z_dict_fake['z_logvar'], z_dict_fake['z']

        kl_rec = calc_kl(rec_logvar, rec_mu, reduce="none")
        kl_fake = calc_kl(fake_logvar, fake_mu, reduce="none")

        loss_rec_rec_e = calc_reconstruction_loss(rec, rec_rec, loss_type="mse", reduction='none')
        while len(loss_rec_rec_e.shape) > 1:
            loss_rec_rec_e = loss_rec_rec_e.sum(-1)
        loss_rec_fake_e = calc_reconstruction_loss(fake, rec_fake, loss_type="mse", reduction='none')
        while len(loss_rec_fake_e.shape) > 1:
            loss_rec_fake_e = loss_rec_fake_e.sum(-1)

        expelbo_rec = (-2 * self.scale * (self.beta_rec * loss_rec_rec_e + self.beta_neg * kl_rec)).exp().mean()
        expelbo_fake = (-2 * self.scale * (self.beta_rec * loss_rec_fake_e + self.beta_neg * kl_fake)).exp().mean()

        lossE_fake = 0.25 * (expelbo_rec + expelbo_fake)
        lossE_real = self.scale * (self.beta_rec * loss_rec + self.beta_kl * lossE_real_kl)

        lossE = lossE_real + lossE_fake + 0.005 * loss_emb
        optimizer_e.zero_grad()
        self.manual_backward(lossE)
        optimizer_e.step()
        

        # ========= Update D ==================
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = True

        fake = self.sample(noise_batch)
        rec = self.decoder(z.detach())
        loss_rec = calc_reconstruction_loss(real_batch, rec, loss_type="mse", reduction="mean")

        rec_mu, rec_logvar,_ = self.encode(rec)
        z_rec = reparameterize(rec_mu, rec_logvar)

        fake_mu, fake_logvar,_ = self.encode(fake)
        z_fake = reparameterize(fake_mu, fake_logvar)

        rec_rec = self.decode(z_rec.detach())
        rec_fake = self.decode(z_fake.detach())

        loss_rec_rec = calc_reconstruction_loss(rec.detach(), rec_rec, loss_type="mse", reduction="mean")
        loss_fake_rec = calc_reconstruction_loss(fake.detach(), rec_fake, loss_type="mse", reduction="mean")

        lossD_rec_kl = calc_kl(rec_logvar, rec_mu, reduce="mean")
        lossD_fake_kl = calc_kl(fake_logvar, fake_mu, reduce="mean")


        lossD = self.scale * (loss_rec * self.beta_rec + (
                lossD_rec_kl + lossD_fake_kl) * 0.5 * self.beta_kl + self.gamma_r * 0.5 * self.beta_rec * (
                                 loss_rec_rec + loss_fake_rec))

        optimizer_d.zero_grad()
        self.manual_backward(lossD)
        optimizer_d.step()

        recon,_ = self(x)
        loss = self.loss_fn(recon, x)
        self.log_dict({"e_loss": lossE, "d_loss": lossD, 'emb_loss': loss_emb}, prog_bar=True, on_epoch=True, on_step=False)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss 
        
    def validation_step(self, batch: Tensor, batch_idx):
        x = batch
        recon,_ = self(x)
        loss = self.loss_fn(recon, x)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):

        print(self.config)
        optimizer_e = optim.Adam(self.encoder.parameters(), lr=self.config['lr'])
        optimizer_d = optim.Adam(self.decoder.parameters(), lr=self.config['lr'])
        return optimizer_e, optimizer_d

    def compute_anomaly(self, x, x_rec):
        anomaly_maps = []
        for i in range(len(x)):
            x_res, saliency = self.compute_residual(x[i][0], x_rec[i][0])
            anomaly_maps.append(x_res*saliency)
        anomaly_maps = np.asarray(anomaly_maps)
        anomaly_scores = np.mean(anomaly_maps, axis=(1, 2, 3), keepdims=True)
        return anomaly_maps, anomaly_scores

    def compute_residual(self, x, x_rec):
        x = torch.clamp(x,0,1)
        x_rec = torch.clamp(x_rec, 0, 1)
        saliency = self.get_saliency(x, x_rec)
        x_rescale = exposure.equalize_adapthist(x.cpu().detach().numpy())
        x_rec_rescale = exposure.equalize_adapthist(x_rec.cpu().detach().numpy())
        saliency2 = self.get_saliency(torch.Tensor(x_rec_rescale).to(x_rec.device), torch.Tensor(x_rescale).to(x_rec.device))
        saliency = saliency * saliency2
        # x_res = np.abs(x_rec.cpu().detach().numpy() - x.cpu().detach().numpy())
        x_res = np.abs(x_rec_rescale - x_rescale)
        return x_res, saliency

    def lpips_loss(self, ph_img, anomaly_img, mode=0):
        def ano_cam(ph_img_, anomaly_img_):
            # anomaly_img_.requires_grad_(True)
            loss_lpips = self.l_pips_sq(anomaly_img_, ph_img_, normalize=True, retPerLayer=False)
            return loss_lpips.cpu().detach().numpy()

        if len(ph_img.shape) == 2:
            ph_img = torch.unsqueeze(torch.unsqueeze(ph_img, 0), 0)
            anomaly_img = torch.unsqueeze(torch.unsqueeze(anomaly_img, 0), 0)
        ano_map = ano_cam(ph_img, anomaly_img)
        return ano_map

    def get_saliency(self, x, x_rec):
        saliency = self.lpips_loss(x_rec, x)
        saliency = gaussian_filter(saliency, sigma=2)
        return saliency

    def sample(self, z, y_cond=None):
        y = self.decode(z, y_cond=y_cond)
        return y

    def sample_with_noise(self, num_samples=1, device=torch.device("cpu"), y_cond=None):
        z = torch.randn(num_samples, self.zdim).to(device)
        return self.decode(z, y_cond=y_cond)

    def encode(self, x, o_cond=None):
        if self.conditional and o_cond is not None:
            mu, logvar, embed_dict = self.encoder(x, o_cond=o_cond)
        else:
            mu, logvar, embed_dict = self.encoder(x)
        return mu, logvar, embed_dict

    def decode(self, z, y_cond=None):
        if self.conditional and y_cond is not None:
            y = self.decoder(z, y_cond=y_cond)
        else:
            y = self.decoder(z)
        return y


"""
Helpers
"""


def calc_kl(logvar, mu, mu_o=0.0, logvar_o=0.0, reduce='sum'):
    """
    Calculate kl-divergence
    :param logvar: log-variance from the encoder
    :param mu: mean from the encoder
    :param mu_o: negative mean for outliers (hyper-parameter)
    :param logvar_o: negative log-variance for outliers (hyper-parameter)
    :param reduce: type of reduce: 'sum', 'none'
    :return: kld
    """
    if not isinstance(mu_o, torch.Tensor):
        mu_o = torch.tensor(mu_o).to(mu.device)
    if not isinstance(logvar_o, torch.Tensor):
        logvar_o = torch.tensor(logvar_o).to(mu.device)
    kl = -0.5 * (1 + logvar - logvar_o - logvar.exp() / torch.exp(logvar_o) - (mu - mu_o).pow(2) / torch.exp(
        logvar_o)).sum(1)
    if reduce == 'sum':
        kl = torch.sum(kl)
    elif reduce == 'mean':
        kl = torch.mean(kl)
    return kl


def reparameterize(mu, logvar):
    """
    This function applies the reparameterization trick:
    z = mu(X) + sigma(X)^0.5 * epsilon, where epsilon ~ N(0,I)
    :param mu: mean of x
    :param logvar: log variaance of x
    :return z: the sampled latent variable
    """
    device = mu.device
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std).to(device)
    return mu + eps * std


def calc_reconstruction_loss(x, recon_x, loss_type='mse', reduction='sum'):
    """
    :param x: original inputs
    :param recon_x:  reconstruction of the VAE's input
    :param loss_type: "mse", "l1", "bce"
    :param reduction: "sum", "mean", "none"
    :return: recon_loss
    """
    if reduction not in ['sum', 'mean', 'none']:
        raise NotImplementedError
    recon_x = recon_x.view(recon_x.size(0), -1)
    x = x.view(x.size(0), -1)
    if loss_type == 'mse':
        recon_error = F.mse_loss(recon_x, x, reduction='none')
        recon_error = recon_error.sum(1)
        if reduction == 'sum':
            recon_error = recon_error.sum()
        elif reduction == 'mean':
            recon_error = recon_error.mean()
    elif loss_type == 'l1':
        recon_error = F.l1_loss(recon_x, x, reduction=reduction)
    elif loss_type == 'bce':
        recon_error = F.binary_cross_entropy(recon_x, x, reduction=reduction)
    else:
        raise NotImplementedError
    return recon_error
