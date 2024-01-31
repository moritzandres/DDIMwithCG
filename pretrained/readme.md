## Pretrained Models

The models in this folder where trained on the DecathlonDataset dataset used in the paper "Diffusion Models for Medical Anomaly Detection".

The diffusion model can be loaded using the `get_diffusion_model` function in the file `model.DDIM` with the flag `load_weights=True` 
or using the `DiffusionModelEncoder` class from the monai-generative package and loading the pretrained model weights as the state-dict
(analog for the classifier).