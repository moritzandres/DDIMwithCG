import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import yaml

from model.model import get_model
from model.ddim import Classifier, DDIM, DDIMwCG
from data_loader import TrainDataModule

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from monai.apps import DecathlonDataset
from monai import transforms
from monai.data import DataLoader


with open('./configs/ddim_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Reproducibility
pl.seed_everything(config['seed'])


# load data
train_data_module = TrainDataModule(
    split_dir=config['split_dir'],
    target_size=config['target_size'],
    batch_size=config['batch_size'])

# Reconstructions from the validation set
batch = next(iter(train_data_module.val_dataloader()))

model = DDIMwCG(config)
results = model.detect_anomaly(batch)
reconstructions = results['reconstruction'].cpu()
anomaly_map = results['anomaly_map'].cpu()

# Plot images and reconstructions
fig, ax = plt.subplots(3, 5, figsize=(15, 10))
for i in range(5):
    ax[0][i].imshow(batch[i].squeeze(), cmap='gray')
    ax[0][i].axis('off')
    ax[1][i].imshow(reconstructions[i].squeeze(), cmap='gray')
    ax[1][i].axis('off')
    ax[2][i].imshow(anomaly_map[i].squeeze(), cmap='viridis')
    ax[2][i].axis('off')
plt.tight_layout()
plt.savefig('experiments/reconstructions.png')


# from generative.networks.nets.diffusion_model_unet import DiffusionModelEncoder
# def get_classifier(config, load_weights=True):

#     model = DiffusionModelEncoder(
#         spatial_dims=2,
#         in_channels=1,
#         out_channels=2,
#         num_channels=(32, 64, 64),
#         attention_levels=(False, True, True),
#         num_res_blocks=(1, 1, 1),
#         num_head_channels=64,
#         with_conditioning=False,
#     )
#     if load_weights:
#         weights_path = os.path.join(config['pretrained_dir'], "best_classifier_model.pth")
#         model.load_state_dict(torch.load(weights_path))
#     return model


# model = get_classifier(config)
# print(model)
# model(batch, timesteps=torch.Tensor((1,)))
