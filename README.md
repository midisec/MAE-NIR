# MAE-NIR: A Masked Autoencoder that Enhances Near-Infrared Spectral Data to Predict Soil Properties

MAE-NIR is a sophisticated machine learning model tailored for the enhancement and analysis of near-infrared spectral data. The code implements the MAE-NIR framework for enhancing NIR spectral data to predict soil properties.

The code in this toolbox implements the ["MAE-NIR: A Masked Autoencoder that Enhances Near-Infrared Spectral Data to Predict Soil Properties"]("#")



The complete framework is as follows.

![Complete framework](.\Complete framework.png)

## Features

- [x] **Enhanced NIR Spectral Data**: Achieve cleaner and more relevant data by masking out noise and irrelevant spectral bands.

- [x] **Accurate Predictions**: Utilize the power of deep learning to predict various soil properties with high accuracy.

- [x] **Scalable Architecture**: Works seamlessly on both small-scale and large-scale datasets.

- [ ] More models.

Installation
---------------------

Ensure you have the following dependencies installed:

- Python 3.8+
- PyTorch
- [Additional dependencies from requirements.txt]

To install, run:

```bash
git clone https://github.com/midisec/MAE-NIR
pip3 install requirements.txt
```

## Usage

### Prepare Your Dataset

For dataset, modify the loader in the `data/load_dataset.py` script to adapt to your dataset structure.

The [LUCAS 2009 dataset](http://esdac.jrc.ec.europa.eu/) used in this work was made available by the European Commission through the European Soil Data Centre managed by the Joint Research Centre (JRC).

Additionally, We have provided the near-infrared spectroscopy dataset from Anhui for testing ([AnhuiSoilDataSet](https://github.com/midisec/Anhui-NIR-Soil-Dataset)).

### Modifying Data Loaders

Open `data/load_dataset.py` and modify the data loading function to fit the format of your datasets:

### Training The Model

Training the MAE-NIR model involves two stages. Each stage requires training with a different dataset.

#### Stage I: Using LUCAS dataset

Run:

```bash
python3 MAE_train_stage1.py
```

This script initializes the MAE model, sets up the training loop, and begins training using the LUCAS dataset.

After the training is completed, the model parameters will be saved to a file named `mae1d_1000.ckpt`. This checkpoint contains the learned parameters and is crucial for the second stage of training.

#### Stage II: Using Anhui Soil Dataset

After completing the first stage, you can continue with the second stage.

Load the pre-trained model from Stage I (`mae1d_1000.ckpt`).

Continue with the second stage of training by running:

```
python3 MAE_train_stage2.py
```

This script loads the pre-trained model from Stage I and continues training using the Anhui Soil dataset.

Once training is completed, the final model parameters will be saved. You can now use this trained model for inference on new data.

## Testing

To test the trained model on your dataset, use:

```
python3 MAE_test.py
```

## MAE Architecture

MAE-1D:

![MAE1D](.\MAE1D.png)

MAE-2D:

![MAE2D](.\MAE2D.png)



Others
----------------------

The diffusion method in this project references the implementation from the following repository: [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch).

The CycleGAN method used in this project is based on the implementation from the following repository: [CycleGAN by junyanz](https://github.com/junyanz/CycleGAN).

If you encounter the bugs while using this code, please do not hesitate to contact us.

Licensing
---------

Copyright (C) 2023 Midi Wan

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.