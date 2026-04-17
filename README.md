# MNIST CNN Classification

## Project Description

This project implements a Convolutional Neural Network (CNN) for handwritten digit classification using the MNIST dataset.

The model is trained using PyTorch and includes evaluation metrics and visualization tools.

## Tech Stack

- Python
- PyTorch
- NumPy
- Matplotlib
- scikit-learn


## Model Architecture

The model is a simple CNN:

- Conv2D → ReLU → MaxPool
- Conv2D → ReLU → MaxPool
- Fully Connected Layer
- Output Layer (10 classes)

## Results

During training, the following metrics are generated:

- Training loss curve
- Test accuracy curve
- Confusion matrix

All plots are saved in:
outputs/plots/

## Key Features

- CNN implementation from scratch (PyTorch)
- Training loop with evaluation
- Visualization of training process
- Confusion matrix analysis
- Reproducible pipeline