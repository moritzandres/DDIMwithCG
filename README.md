# Master Seminar - Unsupervised Anomaly Detection for Medical Imaging

Most of the code (structure, dataloader class and evaluation) was provided by our supervisors Felix Meissen and Cosmin-Ionut Bercea. Check out their code [here](https://github.com/compai-lab/mad_seminar_s23/tree/main).

# DDIM with Classifier Guidance

This code re-implements, trains and evaluates the Denoising Diffusion Implicit Model (DDIM) with Classifier Guidance introduced in the paper "[Diffusion Models for Medical Anomaly Detection](https://doi.org/10.1007/978-3-031-16452-1_4)". Here is a brief overview of the model:

<p align="center"><img src="./DDIMwCG_sketch.png" alt="drawing" width="70%"/></p>

For both training and evaluation we use a small dataset of brain MRI scans. The dataset contains images of heathy patients and different pathologies.
Since we are going to use the normal vs. pathological scans the approach is semi-supervised. Here you can see some samples of our dataset:

| Absent Septum | Encephalomalacia | Intraventricular | Craniatomy | Enlarged Ventricles | Normal |
| :---: | :---: | :---: | :---: | :---: | :---: |
|![image](./data/fastMRI/brain_mid_png/file_brain_AXT1_202_6000392.png)|![image](./data/fastMRI/brain_mid_png/file_brain_AXT1_202_2020377.png)|![image](./data/fastMRI/brain_mid_png/file_brain_AXT1_202_6000391.png)|![image](./data/fastMRI/brain_mid_png/file_brain_AXT1_202_2020486.png)|![image](./data/fastMRI/brain_mid_png/file_brain_AXT1_202_2020517.png)|![image](./data/fastMRI/brain_mid_png/file_brain_AXT1_202_6000312.png)|

## Installation

Create (and activate) a new virtual environment (requires conda) and install the required packages
```shell
conda create --name mad python=3.9
conda activate mad
pip install -r requirements.txt
```

## Training

To train the DDIM with Classifier GUidance we need to train the DDIM and the Classifier separately.

### Train Classifier

First, we train the classifier to distinguish healthy from pathological scans.
We feed noisy images to the classifier during training since we are going to use the classifier gradients on the noisy reconstruction in the denoising process of the DDIM to shift the reconstruction towards heathy images.
We also trained the classifier on the Decathlon dataset because \
The training is done in the [train_classifier.ipynb](./train_classifier.ipynb).

### Train DDIM



### Training Curves

| DDIM | Classifier | Classifier (On Decathlon Dataset) |
| --- | --- | --- |
|![image](.\\logs\\my_logs_ddim\\metrics.png)|![image](.\\logs\\my_logs_classifier\\metrics.png)|![image](.\\logs\\my_logs_classifier_dec\\metrics.png)|



## Evaluation

To see all experiment results check-out the experiments folder

### Image generation ability of the DDIM (no Classifier Guidance)

<img src="./experiments/ddim_img_generation.png" alt="drawing" width="100%"/>

### Full Reconstruction (noising + guidaed denoising)

<img src="./experiments/chain_full.png" alt="drawing" width="100%"/>

### Gradients during Denoising Process

<img src="./experiments/chain_gradients.png" alt="drawing" width="100%"/>

### Video of Reconstruction

Noising Process | Denoising Process | Gradients during DP | Full Reconstruction
| :---: | :---: | :---: | :---: |
| ![video](./experiments/vid_noising_imgs.gif)|![video](./experiments/vid_denoising_imgs.gif)|![video](./experiments/vid_gradients.gif)|![video](./experiments/vid_full_chain.gif) |

### Reconstruction and Anomaly Maps of MRI Scans

<img src="./experiments/compare_pathologies.png" alt="drawing" width="700px"/>

## Conclusion

Looking at Anomaly maps we can clearly see that the model is not able to find and remove the anomalies in the images. This is due to the fact that the classifier is not able to learn the distribution of the healthy images. This is probably due to the fact that the classifier is trained on a relatively small dataset and the Decathlon dataset which is also reflected in the training curves of the classifier.

Looking at the image reconstruction of the DDIM without the classifier guidance we can see that the model is able to generate images that look like the input images. Due to the fact that the gradients of the classifier during denoising do not vanish as the image looks more like a healthy image, the classifier keeps adding noise to the image which results in a noisy reconstruction.

The best way to improve the model would be to use a more performant classifier that for example was trained on a larger dataset or with the use of data augmentation.
