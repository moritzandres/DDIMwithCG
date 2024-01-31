import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import yaml
import os
from time import time
from torch import Tensor
import torch.nn.functional as F
from monai.utils import set_determinism
from torch.cuda.amp import autocast
from tqdm import tqdm
from generative.inferers import DiffusionInferer
from generative.networks.nets.diffusion_model_unet import DiffusionModelEncoder, DiffusionModelUNet
from generative.networks.schedulers.ddim import DDIMScheduler
from PIL import Image
import numpy as np
from torchvision import transforms as tvt
import imageio

from model.ddim import DDIMwCG, DDIM, Classifier

set_determinism(42)


img_dir = '.\\data\\fastMRI\\brain_mid_png'
mask_dir = '.\\data\\fastMRI\\brain_mid_anno_pos_png'
neg_mask_dir = '.\\data\\fastMRI\\brain_mid_anno_neg_png'

data = {
    "ABSENT_SEPTUM": {
        "image": "file_brain_AXT1_202_6000392.png",
        "mask": "file_brain_AXT1_202_6000392_absent_septum_pellucidum_0.png",
        "neg_mask": "file_brain_AXT1_202_6000392.png",
    },
    "ENCEPHALOMALACIA": {
        "image": "file_brain_AXT1_202_2020377.png",
        "mask": "file_brain_AXT1_202_2020377_encephalomalacia_0.png",
        "neg_mask": "file_brain_AXT1_202_2020377.png",
    },
    "INTRAVENTRICULAR": {
        "image": "file_brain_AXT1_202_6000391.png",
        "mask": "file_brain_AXT1_202_6000391_intraventricular_substance_0.png",
        "neg_mask": "file_brain_AXT1_202_6000391.png",
    },  
    "CRANIATOMY": {
        "image": "file_brain_AXT1_202_2020486.png", 
        "mask": "file_brain_AXT1_202_2020486_craniotomy_0.png",
        "neg_mask": "file_brain_AXT1_202_2020486.png", 
    },
    "ENLARGED_VENTRICLES": {
        "image": "file_brain_AXT1_202_2020517.png",
        "mask": "file_brain_AXT1_202_2020517_enlarged_ventricles_0.png",
        "neg_mask": "file_brain_AXT1_202_2020517.png",
    },
    "NORMAL": {
        "image": "file_brain_AXT1_202_6000312.png",
        "mask": "file_brain_AXT1_202_6000391_posttreatment_change_2.png",
        "neg_mask": "file_brain_AXT1_202_6000312.png",
    },
}


with open('./configs/ddim_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Reproducibility
pl.seed_everything(config['seed'])


def compare_diseases(config, imgs, gt_imgs):

    model = DDIMwCG(config)  # change to your model
    ano_maps = model.detect_anomaly(torch.cat(imgs, dim=0))

    fig, axs = plt.subplots(len(imgs), 4, figsize=(5,6), dpi=300)
    for ax in axs.flatten():
        ax.xaxis.set_visible(False) 
        ax.tick_params(left=False, labelleft=False)

    # for i in range(len(imgs)):
    #     # axs[i][0].set_ylabel(list(data.keys())[i], fontsize=4, rotation=0, labelpad=60)
    #     axs[i][0].imshow(imgs[i].numpy().squeeze(), cmap='gray')
    #     axs[i][1].imshow(ano_maps['reconstruction'][i].cpu().numpy().squeeze(), cmap='gray')
    #     axs[i][2].imshow(ano_maps['anomaly_map'][i].cpu().numpy().squeeze())
    #     axs[i][3].imshow(gt_imgs[i], cmap='gray')
    # axs[0][0].set_title('Input', fontsize=8)
    # axs[0][1].set_title('Reconstr.', fontsize=8)
    # axs[0][2].set_title('Ano.-Map', fontsize=8)
    # axs[0][3].set_title('GT Mask', fontsize=8)

    # plt.tight_layout()
    # plt.savefig(f'experiments/compare_pathologies.png')
    # plt.close()

    # os.makedirs('experiments/pathologies', exist_ok=True)
    # for i in range(len(imgs)):
    #     plt.imshow(ano_maps['anomaly_map'][i].cpu().numpy().squeeze(), cmap='gray')
    #     plt.axis('off')
    #     plt.tight_layout()
    #     plt.savefig(f'experiments/pathologies/{list(data.keys())[i]}_anomap.png')
    #     plt.close()

    #     plt.imshow(ano_maps['reconstruction'][i].cpu().numpy().squeeze(), cmap='gray')
    #     plt.axis('off')
    #     plt.tight_layout()
    #     plt.savefig(f'experiments/pathologies/{list(data.keys())[i]}_reconstr.png')
    #     plt.close()
        
    # save combined reconstructions as single image
    im_arr = np.concatenate([ano_maps['reconstruction'][i].cpu().numpy().squeeze() for i in range(len(imgs))], axis=1)



def compare_L(config, img, gt_img, s=50):

    config['s'] = s
    Ls = [10, 50, 100, 200, 400, 900]

    res = {}
    for L in Ls:
        config['L'] = L
        model = DDIMwCG(config)
        res[L] = model.detect_anomaly(img)

    fig, axs = plt.subplots(len(Ls), 4, dpi=300)
    for ax in axs.flat:
        ax.xaxis.set_visible(False) 
        ax.tick_params(left=False, labelleft=False)

    max_val = max([res[L]['anomaly_score'].cpu().numpy()[0] for L in Ls])
    print([res[L]['anomaly_score'].cpu().numpy()[0] for L in Ls], max_val)

    for i, (L, ano_map) in enumerate(res.items()):
        axs[i][0].set_ylabel(f"L: {L}", fontsize=4, rotation=0, labelpad=20)
        axs[i][0].imshow(img.numpy().squeeze(), cmap='gray')
        axs[i][1].imshow(ano_map['reconstruction'][0].cpu().numpy().squeeze(), cmap='gray')
        axs[i][2].imshow(ano_map['anomaly_map'][0].cpu().numpy().squeeze())
        axs[i][3].imshow(gt_img, cmap='gray')
    axs[0][0].set_title('Input', fontsize=8)
    axs[0][1].set_title('Reconstr.', fontsize=8)
    axs[0][2].set_title('Ano.-Map', fontsize=8)
    axs[0][3].set_title('GT Mask', fontsize=8)

    plt.tight_layout()
    plt.savefig(f'experiments/compare_L.png')


def compare_s(config, img, gt_img, L=400):

    config['L'] = L
    Ss = [10, 50, 100, 200, 400]

    res = {}
    for s in Ss:
        config['s'] = s
        model = DDIMwCG(config)
        res[s] = model.detect_anomaly(img)

    fig, axs = plt.subplots(len(Ss), 4, dpi=300)
    for ax in axs.flat:
        ax.xaxis.set_visible(False) 
        ax.tick_params(left=False, labelleft=False)

    max_val = max([res[s]['anomaly_score'].cpu().numpy()[0] for s in Ss])
    print([res[s]['anomaly_score'].cpu().numpy()[0] for s in Ss], max_val)

    for i, (s, ano_map) in enumerate(res.items()):
        axs[i][0].set_ylabel(f"s: {s}", fontsize=4, rotation=0, labelpad=20)
        axs[i][0].imshow(img.numpy().squeeze(), cmap='gray')
        axs[i][1].imshow(ano_map['reconstruction'][0].cpu().numpy().squeeze(), cmap='gray')
        axs[i][2].imshow(ano_map['anomaly_map'][0].cpu().numpy().squeeze())
        axs[i][3].imshow(gt_img, cmap='gray')
    axs[0][0].set_title('Input', fontsize=8)
    axs[0][1].set_title('Reconstr.', fontsize=8)
    axs[0][2].set_title('Ano.-Map', fontsize=8)
    axs[0][3].set_title('GT Mask', fontsize=8)

    plt.tight_layout()
    plt.savefig(f'experiments/compare_s.png')


def visualize_gradients(config, img):

    class DDIMwCG_COPY:

        """ DDIM model with classifier-guidance
        !COPY OF DDIMwCG CLASS FROM ddim.py
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
            grads = []
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
                        logits = self.classifier(x_in, timesteps=torch.Tensor((t,)).to(rec.device))
                        log_probs = F.log_softmax(logits, dim=-1)
                        selected = log_probs[range(len(logits)), y.view(-1)]

                        # get gradient C(x_t) regarding x_t 
                        a = torch.autograd.grad(selected.sum(), x_in)[0]
                        alpha_prod_t = self.scheduler.alphas_cumprod[t]
                        updated_noise = (
                            model_output - (1 - alpha_prod_t).sqrt() * self.scale * a
                        )  # update the predicted noise epsilon with the gradient of the classifier
                        grads.append(a.detach().cpu().numpy().squeeze())

                rec, _ = self.scheduler.step(updated_noise, t, rec)
                torch.cuda.empty_cache()

            # anomaly detection
            anomaly_map = torch.abs(x - rec)
            anomaly_score = torch.sum(anomaly_map, dim=(1, 2, 3))
            print(f'total inference-time: {time() - t_start:.2f}sec\n')
            return {
                'reconstruction': rec,
                'anomaly_map': anomaly_map,
                'anomaly_score': anomaly_score,
                'gradients': grads,
            }
    
    gradients = DDIMwCG_COPY(config).detect_anomaly(img)["gradients"]

    g_dir = os.path.join('experiments', 'gradients')
    os.makedirs(g_dir, exist_ok=True)

    print('saving images of gradients...')
    for i, grad in enumerate(tqdm(gradients)):
        plt.imshow(grad)
        plt.axis('off')
        plt.title(f'gradient {i}')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(g_dir, f'gradient_{i+1:04d}.png'))
        plt.close()

    gif_path = os.path.join('experiments', 'gradients.gif')
    with imageio.get_writer(gif_path, mode='I') as writer:
        for filename in sorted(os.listdir(g_dir)):
            fpath = os.path.join(g_dir, filename)
            image = imageio.imread(fpath)
            writer.append_data(image)


if __name__ == '__main__':

    # Load images
    imgs = []
    gt_imgs = []
    for i, (disease, fnames) in enumerate(data.items()):
        img = Image.open(os.path.join(img_dir, fnames['image'])).convert('L')
        img = tvt.Pad(((img.height - img.width) // 2, 0), fill=0)(img)  # Pad to square
        img = img.resize(config["target_size"], Image.BICUBIC)  # Resize
        img = tvt.ToTensor()(img)  # Convert to tensor
        img = torch.unsqueeze(img, 0)  # shape to batch
        imgs.append(img)

        gt_img = Image.open(os.path.join(mask_dir, fnames['mask'])).convert('L')
        gt_img = tvt.Pad(((gt_img.height - gt_img.width) // 2, 0), fill=0)(gt_img)  # Pad to square
        gt_img = gt_img.resize(config["target_size"], Image.BICUBIC)  # Resize
        gt_img = np.array(gt_img)
        gt_imgs.append(gt_img)


    ### run experiments ###
    compare_diseases(config, imgs, gt_imgs)
    # compare_L(config, imgs[0], gt_imgs[0], s=50)
    # compare_s(config, imgs[0], gt_imgs[0], L=200)
    # visualize_gradients(config, imgs[0])
