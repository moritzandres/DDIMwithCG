import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import yaml
import os
from time import time
import torch.nn.functional as F
from monai.utils import set_determinism
from torch.cuda.amp import autocast
from tqdm import tqdm
from PIL import Image
import numpy as np
from torchvision import transforms as tvt

from model.ddim import DDIMwCG
from model.ddim import get_diffusion_model, get_classifier

set_determinism(42)


img_dir = './data/fastMRI/brain_mid_png'
mask_dir = './data/fastMRI/brain_mid_anno_pos_png'
neg_mask_dir = './data/fastMRI/brain_mid_anno_neg_png'

# sample files for different pathologies
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


def evaluate_ddim(config, L=1000):
    """ evaluate the image generation ability of the DDIM model (without classifier-guidance) """

    model = DDIMwCG(config)
    ddim = model.diffusion_model
    scheduler = model.scheduler
    n = L // 10  # number of images to save (10)

    # denoising process
    imgs = []
    model_outs = []
    x = torch.randn((1, 1, 64, 64)).to(model.device)
    with torch.no_grad():
        progress_bar = tqdm(range(L-1,-1,-1))  # go back and forth L timesteps
        for t in progress_bar:  # go through the denoising process
            with autocast(enabled=True):
                model_output = ddim(
                    x, timesteps=torch.Tensor((t,)).to(x.device)
                )
                x, _ = scheduler.step(model_output, t, x)
                x = torch.clamp(x, -1, 1)  # clamp 
            torch.cuda.empty_cache()

            if t % n == 0:
                imgs.append((t, x.cpu().numpy().squeeze()))
                model_outs.append((t, model_output.cpu().numpy().squeeze()))
    
    # save chain 
    chain = torch.cat([torch.from_numpy(img) for t, img in imgs], dim=-1)
    chain = np.clip(chain.cpu().numpy(), 0, 1) * 255
    image = Image.fromarray(chain.astype(np.uint8), mode='L')
    image.save("experiments/pure_ddim_denoising.png")


def compare_diseases(config, imgs, gt_imgs):

    model = DDIMwCG(config)  # change to your model
    L, s = model.L, model.scale
    ano_maps = model.detect_anomaly(torch.cat(imgs, dim=0))

    fig, axs = plt.subplots(len(imgs), 4, figsize=(4*1.5, len(imgs)*1.5), dpi=300)
    for ax in axs.flatten():
        ax.xaxis.set_visible(False) 
        ax.tick_params(left=False, labelleft=False)

    for i in range(len(imgs)):
        patho_name = list(data.keys())[i]
        axs[i][0].set_ylabel(f'{patho_name}\n(L={L},s={s})', fontsize=6, rotation=0, labelpad=40)
        axs[i][0].imshow(imgs[i].numpy().squeeze(), cmap='gray')
        axs[i][1].imshow(ano_maps['reconstruction'][i].cpu().numpy().squeeze(), cmap='gray')
        axs[i][2].imshow(ano_maps['anomaly_map'][i].cpu().numpy().squeeze(), cmap='jet')
        axs[i][3].imshow(gt_imgs[i], cmap='gray')
    axs[0][0].set_title('Input', fontsize=8)
    axs[0][1].set_title('Reconstr.', fontsize=8)
    axs[0][2].set_title('Ano.-Map', fontsize=8)
    axs[0][3].set_title('GT Mask', fontsize=8)

    plt.tight_layout()
    plt.savefig(f'experiments/compare_pathologies.png')
    plt.close()

    # # save separate images
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


def compare_L(config, imgs, gt_imgs, s=50):

    config['s'] = s
    Ls = [100, 200, 400, 800, 1000]

    res = {}
    for L in Ls:
        config['L'] = L
        model = DDIMwCG(config)
        res[L] = model.detect_anomaly(torch.cat(imgs, dim=0))

    os.makedirs('experiments/compare_L', exist_ok=True)

    for j, (img, gt_img) in enumerate(zip(imgs, gt_imgs)):

        fig, axs = plt.subplots(len(Ls), 4, figsize=(4*1.5, len(Ls)*1.5), dpi=300)
        for ax in axs.flat:
            ax.xaxis.set_visible(False) 
            ax.tick_params(left=False, labelleft=False)

        for i, (L, ano_map) in enumerate(res.items()):
            axs[i][0].set_ylabel(f"L={L}\n(s={s})", fontsize=8, rotation=0, labelpad=20)
            axs[i][0].imshow(img.numpy().squeeze(), cmap='gray')
            axs[i][1].imshow(ano_map['reconstruction'][j].cpu().numpy().squeeze(), cmap='gray')
            axs[i][2].imshow(ano_map['anomaly_map'][j].cpu().numpy().squeeze(), cmap='jet')
            axs[i][3].imshow(gt_img, cmap='gray')
        axs[0][0].set_title('Input', fontsize=8)
        axs[0][1].set_title('Reconstr.', fontsize=8)
        axs[0][2].set_title('Ano.-Map', fontsize=8)
        axs[0][3].set_title('GT Mask', fontsize=8)

        plt.tight_layout()
        patho_name = list(data.keys())[j]
        plt.savefig(f'experiments/compare_L/{patho_name}.png')
        plt.close()


def compare_s(config, imgs, gt_imgs, L=400):

    config['L'] = L
    # Ss = [10, 50, 100, 200, 400]
    Ss = [0.3, 1, 3, 9, 27]

    res = {}
    for s in Ss:
        config['s'] = s
        model = DDIMwCG(config)
        res[s] = model.detect_anomaly(torch.cat(imgs, dim=0))

    os.makedirs('experiments/compare_s', exist_ok=True)

    for j, (img, gt_img) in enumerate(zip(imgs, gt_imgs)):

        fig, axs = plt.subplots(len(Ss), 4, figsize=(4*1.5, len(Ss)*1.5), dpi=300)
        for ax in axs.flat:
            ax.xaxis.set_visible(False) 
            ax.tick_params(left=False, labelleft=False)

        for i, (s, ano_map) in enumerate(res.items()):
            axs[i][0].set_ylabel(f"s={s}\n(L={L})", fontsize=8, rotation=0, labelpad=20)
            axs[i][0].imshow(img.numpy().squeeze(), cmap='gray')
            axs[i][1].imshow(ano_map['reconstruction'][j].cpu().numpy().squeeze(), cmap='gray')
            axs[i][2].imshow(ano_map['anomaly_map'][j].cpu().numpy().squeeze(), cmap='jet')
            axs[i][3].imshow(gt_img, cmap='gray')
        axs[0][0].set_title('Input', fontsize=8)
        axs[0][1].set_title('Reconstr.', fontsize=8)
        axs[0][2].set_title('Ano.-Map', fontsize=8)
        axs[0][3].set_title('GT Mask', fontsize=8)

        plt.tight_layout()
        patho_name = list(data.keys())[j]
        plt.savefig(f'experiments/compare_s/{patho_name}.png')
        plt.close()


def visualize_flow(config, img, L=400, s=100):

    model = DDIMwCG(config)
    ddim = model.diffusion_model
    classifier = model.classifier
    scheduler = model.scheduler
    scale = s
    n = L // 10  # number of images to save (10)

    noising_imgs = []
    noising_outs = []
    cls_grads = []
    combined_outs = []
    denoising_imgs = []
    denoising_outs = []

    # noising process
    x = img.to(model.device)
    progress_bar = tqdm(range(L-1))
    for t in progress_bar:  # go through the noising process
        with autocast(enabled=False):
            with torch.no_grad():
                model_output = ddim(x, timesteps=torch.Tensor((t,)).to(x.device))
        x, _ = scheduler.reversed_step(model_output, t, x)
        x = torch.clamp(x, -1, 1)

        noising_imgs.append(x.cpu().numpy().squeeze())
        noising_outs.append(model_output.cpu().numpy().squeeze())

    # denoising process
    y = torch.tensor(0)  # define the desired class label
    with torch.no_grad():
        progress_bar = tqdm(range(L-1,-1,-1))  # go back and forth L timesteps
        for t in progress_bar:  # go through the denoising process
            with autocast(enabled=True):
                with torch.no_grad():
                    model_output = ddim(
                        x, timesteps=torch.Tensor((t,)).to(x.device)
                    ).detach()  # this is supposed to be epsilon

                with torch.enable_grad():
                    x = x.detach().requires_grad_(True)
                    logits = classifier(x, timesteps=torch.Tensor((t,)).to(x.device))
                    log_probs = F.log_softmax(logits, dim=-1)
                    selected = log_probs[range(len(logits)), y.view(-1)]

                    # get gradient C(x_t) regarding x_t 
                    a = torch.autograd.grad(selected.sum(), x)[0]
                    alpha_prod_t = scheduler.alphas_cumprod[t]
                    updated_noise = (
                        model_output - (1 - alpha_prod_t).sqrt() * scale * a
                    )  # update the predicted noise epsilon with the gradient of the classifier

            x, _ = scheduler.step(updated_noise, t, x)
            x = torch.clamp(x, -1, 1)  # clamp
            torch.cuda.empty_cache()

            denoising_imgs.append(x.cpu().numpy().squeeze())
            denoising_outs.append(model_output.cpu().numpy().squeeze())
            cls_grads.append(a.cpu().numpy().squeeze())
            combined_outs.append(updated_noise.cpu().numpy().squeeze())
        

    noise_idxs = np.unique(np.linspace(0, len(noising_imgs)-1, num=10, dtype=int))
    denoise_idxs = np.unique(np.linspace(0, len(denoising_imgs)-1, num=10, dtype=int))
    full_idxs = np.unique(np.linspace(0, len(noising_imgs)+len(denoising_imgs)-1, num=10, dtype=int))
    
    # noising reconstruction
    images = np.array(noising_imgs)
    images = np.clip(images, 0, 1) * 255
    imgs = [Image.fromarray(img) for img in images.astype(np.uint8)]
    imgs[0].save("experiments/video_noising_imgs.gif", save_all=True, append_images=imgs[1:], duration=2000 // len(imgs), loop=0)
    noise_chain = torch.cat([torch.from_numpy(img) for img in images[noise_idxs]], dim=-1).numpy()
    Image.fromarray(noise_chain.astype(np.uint8)).save("experiments/chain_noise.png")

    # gradients
    images = np.absolute(np.array(cls_grads))
    images = (images - images.min()) / (images.max() - images.min()) * 255
    imgs = [Image.fromarray(grad) for grad in images.astype(np.uint8)]
    imgs[0].save("experiments/video_gradients.gif", save_all=True, append_images=imgs[1:], duration=2000 // len(imgs), loop=0)
    grad_chain = torch.cat([torch.from_numpy(grad) for grad in images[denoise_idxs]], dim=-1).numpy()
    Image.fromarray(grad_chain.astype(np.uint8)).save("experiments/chain_gradients.png")

    # denoising reconstruction
    images = np.array(denoising_imgs)
    images = np.clip(images, 0, 1) * 255
    imgs = [Image.fromarray(img) for img in images.astype(np.uint8)]
    imgs[0].save("experiments/video_denoising_imgs.gif", save_all=True, append_images=imgs[1:], duration=2000 // len(imgs), loop=0)
    rec_chain = torch.cat([torch.from_numpy(img) for img in images[denoise_idxs]], dim=-1).numpy()
    Image.fromarray(rec_chain.astype(np.uint8)).save("experiments/chain_reconstruction.png")

    # full chain (noising + denoising)
    images = np.array(noising_imgs + denoising_imgs)
    images = np.clip(images, 0, 1) * 255
    imgs = [Image.fromarray(img) for img in images.astype(np.uint8)]
    imgs[0].save("experiments/video_full_chain.gif", save_all=True, append_images=imgs[1:], duration=4000 // len(imgs), loop=0)
    full_chain = torch.cat([torch.from_numpy(img) for img in images[full_idxs]], dim=-1).numpy()
    full_chain = Image.fromarray(full_chain.astype(np.uint8)).save("experiments/chain_full.png")


def compare_train_dataset(config, imgs, gt_imgs):
    """ compare the performance of the model trained with Decathlon dataset 
        + the model had more data for training
        - the data distribution is different
    """

    # load model trained with Decathlon dataset
    model = DDIMwCG(config)
    ddim = get_diffusion_model(config, load_weights=True).to(model.device)
    classifier = get_classifier(config, load_weights=True).to(model.device)
    model.diffusion_model = ddim
    model.classifier = classifier
    model.eval()

    ano_maps = model.detect_anomaly(torch.cat(imgs, dim=0))

    fig, axs = plt.subplots(len(imgs), 4, figsize=(5*1.5, len(imgs)*1.5), dpi=300)
    for ax in axs.flatten():
        ax.xaxis.set_visible(False) 
        ax.tick_params(left=False, labelleft=False)

    for i in range(len(imgs)):
        patho_name = list(data.keys())[i]
        axs[i][0].set_ylabel(f"{patho_name}\nL={model.L}\ns={model.scale}", fontsize=8, rotation=0, labelpad=60)
        axs[i][0].imshow(imgs[i].numpy().squeeze(), cmap='gray')
        axs[i][1].imshow(ano_maps['reconstruction'][i].cpu().numpy().squeeze(), cmap='gray')
        axs[i][2].imshow(ano_maps['anomaly_map'][i].cpu().numpy().squeeze(), cmap='jet')
        axs[i][3].imshow(gt_imgs[i], cmap='gray')
    axs[0][0].set_title('Input', fontsize=8)
    axs[0][1].set_title('Reconstr.', fontsize=8)
    axs[0][2].set_title('Ano.-Map', fontsize=8)
    axs[0][3].set_title('GT Mask', fontsize=8)

    plt.tight_layout()
    plt.savefig(f'experiments/decathlon_pathologies.png')
    plt.close()


def compare_time(config, imgs):
    """ analyze the inference time for the anomaly detection on different noise levels and cpu/ gpu"""

    # try different noise levels L
    Ls = [1000, 800, 600, 400, 200, 100]

    # on CPU
    results = {'device': [], 'L': [], 'time': []}
    for L in Ls:
        config['L'] = L
        model = DDIMwCG(config)
        model.device = torch.device('cpu')
        model.classifier = model.classifier.to(model.device)
        model.diffusion_model = model.diffusion_model.to(model.device)
        for img in imgs:
            start = time()
            model.detect_anomaly(img)
            results['device'].append('cpu')
            results['time'].append(time() - start)
            results['L'].append(L)
    
    # on GPU
    for L in Ls:
        config['L'] = L
        model = DDIMwCG(config)
        for img in imgs:
            start = time()
            model.detect_anomaly(img)
            results['device'].append('gpu')
            results['time'].append(time() - start)
            results['L'].append(L)
    
    # create two seaborn plots with variance
    import seaborn as sns
    import pandas as pd
    df = pd.DataFrame(results)

    sns.lineplot(data=df, x='L', y='time', hue='device')
    plt.title("CPU time")
    plt.savefig('experiments/inference_time.png')
    plt.close()


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
    
    os.makedirs('experiments', exist_ok=True)

    ### run experiments ###
    # evaluate_ddim(config, L=1000)
    # compare_s(config, imgs, gt_imgs, L=300)
    # compare_L(config, imgs, gt_imgs, s=3)
    # compare_diseases(config, imgs, gt_imgs)
    visualize_flow(config, imgs[2], L=200, s=3)
    # compare_train_dataset(config, imgs, gt_imgs)
    # compare_time(config, imgs)