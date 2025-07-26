# Using ML/AI to Create New Numbers

A machine learning project that demonstrates digit classification and generation using MNIST dataset with PyTorch and creates new digits with a variational autoencoder

## Overview

This project has two main components:
1. **CNN Digit Classifier** - A convolutional neural network for digit recognition
2. **VAE (Variational Autoencoder)** - A generative model for creating new digit images
   
## Features

### CNN Digit Classifier
- **Architecture**: 2D convolutional layers with max pooling
- **Training**: 10 epochs with Adam optimizer
- **Performance**: Achieves low loss (~0.0087) on training data
- **Output**: Classifies digits 0-9 with high accuracy
<img width="2021" height="228" alt="Screenshot 2025-07-26 164622" src="https://github.com/user-attachments/assets/5970f61c-148a-49df-9e96-6c3809eec3e5" />


### VAE (Variational Autoencoder)
- **Purpose**: Generate new digit images by learning latent representations
- **Latent Space**: 2-dimensional for easy visualization
- **Architecture**: Encoder-decoder with reparameterization trick
- **Training**: 20 epochs with custom VAE loss function
- **Interactive**: Mouse-hover visualization of latent space <br>
<img width="1065" height="849" alt="Screenshot 2025-07-26 170519" src="https://github.com/user-attachments/assets/4d758b42-07e5-4a16-8323-573798ea839a" />




Look a new number! I'll call it Zleven:


<img width="236" height="247" alt="Zleven" src="https://github.com/user-attachments/assets/b141a8c8-0a04-4df2-8281-77e5aaecce3b" />


## Key Components

### Data Processing
- Loads MNIST data from IDX format
- Creates two versions: normalized (CNN) and raw [0,1] (VAE)
- Uses DataLoaders for efficient batch processing

### CNN Architecture
```python
# Features: Conv2d → ReLU → MaxPool2d (2 layers)
# Classifier: Flatten → Linear → ReLU → Linear
```

### VAE Architecture
```python
# Encoder: Flatten → Linear → ReLU → mu/logvar
# Decoder: Linear → ReLU → Linear → Sigmoid
```

## Results

- **CNN**: Successfully classifies MNIST digits with high accuracy
- **VAE**: Creates smooth latent space representation allowing generation of new digits
- **Interactive Visualization**: Mouse-hover functionality to explore the 2D latent space

## Notes

- Models are automatically saved to the `models/` directory
- GPU acceleration is used when available
- The VAE uses a 2D latent space for easy visualization and exploration 
