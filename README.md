# Creating New Numbers

A machine learning project that demonstrates digit classification and generation using MNIST dataset with PyTorch and creates new digits with a variational autoencoder

## Overview

This project has two main components:
1. **CNN Digit Classifier** - A convolutional neural network for digit recognition
2. **VAE (Variational Autoencoder)** - A generative model for creating new digit images

## Project Structure

```
Creating_New_Numbers/
├── Making_New_Numbers.ipynb    # Main Jupyter notebook with implementation
├── test.py                      # CUDA/GPU compatibility test
├── models/                      # Saved model files
│   ├── cnn_classifier.pt       # Trained CNN classifier
│   └── vae.pt                  # Trained VAE model
├── mnist_data/                 # MNIST dataset files
│   ├── train-images.idx3-ubyte
│   ├── train-labels.idx1-ubyte
│   ├── t10k-images.idx3-ubyte
│   └── t10k-labels.idx1-ubyte
└── CNN_Example_Predictions.png # Example predictions visualization
```

## Features

### CNN Digit Classifier
- **Architecture**: 2D convolutional layers with max pooling
- **Training**: 10 epochs with Adam optimizer
- **Performance**: Achieves low loss (~0.0087) on training data
- **Output**: Classifies digits 0-9 with high accuracy

### VAE (Variational Autoencoder)
- **Purpose**: Generate new digit images by learning latent representations
- **Latent Space**: 2-dimensional for easy visualization
- **Architecture**: Encoder-decoder with reparameterization trick
- **Training**: 20 epochs with custom VAE loss function
- **Interactive**: Mouse-hover visualization of latent space

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- CUDA-compatible GPU (optional, falls back to CPU)

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