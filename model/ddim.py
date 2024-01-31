
"""
 This code is based on the implementation of DDIM with Classifier-Guidance in the following repository:
 https://github.com/Project-MONAI/GenerativeModels
 tutorials/generative/anomaly_detection/anomalydetection_tutorial_classifier_guidance.py 
"""

import os
import torch
import pytorch_lightning as pl
from torch import Tensor
import torch.nn.functional as F
from monai.utils import set_determinism
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from generative.inferers import DiffusionInferer
from generative.networks.nets.diffusion_model_unet import DiffusionModelEncoder, DiffusionModelUNet
from generative.networks.schedulers.ddim import DDIMScheduler
from time import time

torch.multiprocessing.set_sharing_strategy("file_system")

# enable deterministic training
set_determinism(42)


def get_diffusion_model(config, load_weights=True):
    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        num_channels=(64, 64, 64),
        attention_levels=(False, False, True),
        num_res_blocks=1,
        num_head_channels=64,
        with_conditioning=False,
    )
    if load_weights:
        weights_path = os.path.join(config['pretrained_dir'], "best_diffusion_model.pth")
        model.load_state_dict(torch.load(weights_path))
    return model


def get_classifier(config, load_weights=True):

    model = DiffusionModelEncoder(
        spatial_dims=2,
        in_channels=1,
        out_channels=2,
        num_channels=(32, 64, 64),
        attention_levels=(False, True, True),
        num_res_blocks=(1, 1, 1),
        num_head_channels=64,
        with_conditioning=False,
    )
    if load_weights:
        weights_path = os.path.join(config['pretrained_dir'], "best_classifier_model.pth")
        model.load_state_dict(torch.load(weights_path))
    return model


class DDIMwCG:

    """ DDIM model with classifier-guidance
     this class is used for inference of anomaly detection (not training)
    """

    def __init__(self, config):

        super().__init__()
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        d_model_path = os.path.join(config['checkpoint_dir'], config["diffusion_model"])
        self.diffusion_model = DDIM.load_from_checkpoint(d_model_path)
        # self.diffusion_model = get_diffusion_model(config, load_weights=True)
        self.diffusion_model.to(self.device)

        c_model_path = os.path.join(config['checkpoint_dir'], config["classifier_model"])
        self.classifier = Classifier.load_from_checkpoint(c_model_path)
        # self.classifier = get_classifier(config, load_weights=True)
        self.classifier.to(self.device)

        self.scheduler = DDIMScheduler(num_train_timesteps=1000)
        self.inferer = DiffusionInferer(self.scheduler)

        self.L = self.config['L']      # noise level L
        self.scale = self.config['s']  # gradient-scale s

    def eval(self):
        self.diffusion_model.eval()
        self.classifier.eval()

    def detect_anomaly(self, x: Tensor):
        """ detect anomaly using diffusion model with classifier-guidance """
        t_start = time()

        x = x.to(self.device)
        rec = x.clone()
        self.scheduler.set_timesteps(num_inference_steps=1000)

        print("\nnoising process...")
        progress_bar = tqdm(range(self.L))  # go back and forth L timesteps
        for t in progress_bar:  # go through the noising process
            with autocast(enabled=False):
                with torch.no_grad():
                    model_output = self.diffusion_model(rec, timesteps=torch.Tensor((t,)).to(rec.device))
            rec, _ = self.scheduler.reversed_step(model_output, t, rec)

        print("denoising process...")
        y = torch.tensor(0)  # define the desired class label
        progress_bar = tqdm(range(self.L))  # go back and forth L timesteps
        for i in progress_bar:  # go through the denoising process
            t = self.L - i
            with autocast(enabled=True):
                with torch.no_grad():
                    model_output = self.diffusion_model(
                        rec, timesteps=torch.Tensor((t,)).to(rec.device)
                    ).detach()  # this is supposed to be epsilon

                with torch.enable_grad():
                    x_in = rec.detach().requires_grad_(True)
                    logits = self.classifier(x_in, timesteps=torch.Tensor((t,)).to(rec.device)).requires_grad_(True)
                    log_probs = F.log_softmax(logits, dim=-1).requires_grad_(True)
                    selected = log_probs[range(len(logits)), y.view(-1)].requires_grad_(True)

                    # get gradient C(x_t) regarding x_t 
                    a = torch.autograd.grad(selected.sum(), x_in)[0]
                    alpha_prod_t = self.scheduler.alphas_cumprod[t]
                    updated_noise = (
                        model_output - (1 - alpha_prod_t).sqrt() * self.scale * a
                    )  # update the predicted noise epsilon with the gradient of the classifier

            rec, _ = self.scheduler.step(updated_noise, t, rec)
            torch.cuda.empty_cache()

        # anomaly detection
        anomaly_map = torch.abs(x - rec)
        anomaly_score = torch.sum(anomaly_map, dim=(1, 2, 3))
        print(f'total inference-time: {time() - t_start:.2f}sec\n')
        return {
            'reconstruction': rec,
            'anomaly_map': anomaly_map,
            'anomaly_score': anomaly_score
        }


class DDIM(pl.LightningModule):

    """ DDIM (Denoising Diffusion Implicit Model) model 
     Class for training DDIM model.
    """

    def __init__(self, config, load_weights=False):
        super().__init__()
        self.save_hyperparameters('config')

        # use custom optimization in the training loop
        self.automatic_optimization = False

        self.config = config

        self.diffusion_model = get_diffusion_model(config, load_weights)
        self.diffusion_model.to(self.device)

        self.scheduler = DDIMScheduler(num_train_timesteps=1000)
        self.inferer = DiffusionInferer(self.scheduler)

        # diffusion training
        self.scaler = GradScaler()
        self.optimizer = self.configure_optimizers()
        self.loss_fn = F.mse_loss


    def forward(self, x: Tensor, timesteps: Tensor):
        self.diffusion_model(x, timesteps)
        return x

    def training_step(self, batch: Tensor, batch_idx):
        
        images = batch
        self.optimizer.zero_grad(set_to_none=True)
        timesteps = torch.randint(0, 1000, (len(images),)).to(self.device)  # pick a random time step t

        with autocast(enabled=True):
            # Generate random noise
            noise = torch.randn_like(images).to(self.device)

            # Get model prediction
            noise_pred = self.inferer(inputs=images, diffusion_model=self.diffusion_model, 
                                      noise=noise, timesteps=timesteps)
            loss = self.loss_fn(noise_pred.float(), noise.float())

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss

    def validation_step(self, batch: Tensor, batch_idx):

        images = batch
        timesteps = torch.randint(0, 1000, (len(images),)).to(self.device)
        with torch.no_grad():
            with autocast(enabled=True):
                noise = torch.randn_like(images).to(self.device)
                noise_pred = self.inferer(inputs=images, diffusion_model=self.diffusion_model, 
                                        noise=noise, timesteps=timesteps)
                val_loss = self.loss_fn(noise_pred.float(), noise.float())

        self.log('val_loss', val_loss, prog_bar=True, on_epoch=True, on_step=False)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config['lr'])



class Classifier(pl.LightningModule):

    """ Classifier model 
     Class for training classifier model.
    """

    def __init__(self, config, load_weights=False):
        super().__init__()
        self.save_hyperparameters('config')

        # use custom optimization in the training loop
        self.automatic_optimization = False
        
        self.config = config

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier = get_classifier(config, load_weights)
        self.classifier.to(self.device)

        self.scheduler = DDIMScheduler(num_train_timesteps=1000)

        self.optimizer = self.configure_optimizers()
        self.loss_fn = F.cross_entropy

    def forward(self, x: Tensor, timesteps: Tensor):
        pred = self.classifier(x, timesteps)
        return pred

    def training_step(self, batch: dict, batch_idx):
        images = batch["image"]
        classes = batch["slice_label"]
        # classes[classes==2]=0

        self.optimizer.zero_grad(set_to_none=True)
        timesteps = torch.randint(0, 1000, (len(images),)).to(self.device)

        with autocast(enabled=False):
            # Generate random noise
            noise = torch.randn_like(images).to(self.device)

            # Get model prediction
            noisy_img = self.scheduler.add_noise(images, noise, timesteps)  # add t steps of noise to the input image
            pred = self(noisy_img, timesteps)
            loss = self.loss_fn(pred, classes.long())

            loss.backward()
            self.optimizer.step()
            
        return loss

    def validation_step(self, batch: dict, batch_idx):
        images = batch["image"]
        classes = batch["slice_label"]
        timesteps = torch.randint(0, 1, (len(images),), device=self.device)  
        # check validation accuracy on the original images, i.e., do not add noise

        with torch.no_grad():
            with autocast(enabled=False):
                pred = self(images, timesteps)
                val_loss = self.loss_fn(pred, classes.long(), reduction="mean")

        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config['lr'])

