Most of the code was provided by Felix Meissen and Cosmin-Ionut Bercea. Check out their original code [here](https://github.com/compai-lab/mad_seminar_s23/tree/main).

# Master Seminar - Unsupervised Anomaly Segmentation

This repository contains the PyTorch dataloader classes and an evaluation script
to be used for the implementation of your models.
It also contains an example model and trainer, a simple Autoencoder that can be
used as a starting point for your projects.

[![Open Demo In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/compai-lab/mad_seminar_s23/blob/main/main.ipynb)

## Installation

### When on your local machine

Clone this repository
```shell
git clone https://github.com/compai-lab/mad_seminar_ws23.git
```

Create (and activate) a new virtual environment (requires conda)
```shell
conda create --name mad python=3.9
conda activate mad
```

Install the required packages
```shell
cd mad_seminar_ws23
python -m pip install -r requirements.txt
```

Download and extract the data
```shell
wget <link you got from your supervisor>
unzip data.zip
```

### When in Google Colab

Simply follow the instructions in `main.ipynb`
