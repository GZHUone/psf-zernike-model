# PSF Zernike Model

This project develops a **PyTorch-based network architecture** — **PSF Zernike**, designed for image processing tasks. The model integrates various **image preprocessing techniques** with powerful **network architectures** to enhance **denoising** and **sharpening** of images.

## Project Overview

### 1. PSF Extraction and Cropping
In the image input phase, the **PSF (Point Spread Function)** is extracted and cropped to prepare for further processing.

### 2. Preprocessing Methods
The model applies several denoising and enhancement filters:

#### **Denoising Filters**:
- **Gaussian Denoising**: Applies a Gaussian blur to reduce noise in the image.
- **Mean Denoising**: Replaces each pixel with the average of its neighboring pixels to remove noise.
- **Median Denoising**: Replaces a pixel with the median of its neighboring pixels, especially useful for salt-and-pepper noise.

#### **Sharpening Enhancement Algorithms**:
- **High-Pass Filter Sharpening**: Emphasizes high-frequency components, enhancing image details.
- **Laplacian Sharpening**: Uses the Laplacian operator to sharpen the image by detecting edges.
- **Unsharp Masking**: Enhances contrast by subtracting a blurred version of the image from the original.

**Preprocessing Example**:
![Preprocessing Input Image](https://github.com/GZHUone/psf-zernike-model/blob/main/processs.png)

### 3. Core Network Architecture
The network architecture leverages the strengths of **Convolutional Neural Networks (CNNs)** combined with **Transformer models**. Below are the main combinations used in the model:

- **CNN + Transformer**: Traditional CNNs extract local features, while the Transformer captures global dependencies.
- **ShuffleNet + Transformer**: A lightweight architecture that reduces computational complexity, paired with Transformer for enhanced global feature modeling.
- **EfficientNet + Transformer**: Optimizes the network structure for high performance with minimal computational cost, further enhanced by Transformer.
- **MobileNet + Transformer**: Designed for resource-constrained environments, MobileNet combined with Transformer boosts global feature modeling.

### 4. Network Architecture Diagram
The diagram below illustrates the overall structure of the **PSF Zernike** model, including connections between modules and the flow of data:
![Network Architecture](https://github.com/GZHUone/psf-zernike-model/blob/main/PSF%20Zernike.jpg)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/GZHUone/psf-zernike-model.git
