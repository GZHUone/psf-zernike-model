# PSF Zernike Model

This project introduces a **PyTorch-based network architecture** — **PSF Zernike**, designed specifically for high-performance **image processing** tasks. The model integrates powerful **network architectures** with advanced **image preprocessing techniques** to enhance **denoising** and **sharpening** of images.

## Project Structure

The project contains the following files:

- `psf_crop_denoise.py`: Script for cropping and denoising PSF images.
- `psf_sharpen_.ipynb`: Jupyter notebook for sharpening PSF images.
- `psf_zernike_model.py`: Script implementing the PSF Zernike model.
- `README.md`: This README file.

  
## Key Features

### 1. Network Architectures

The core of this model is built on a combination of **Convolutional Neural Networks (CNNs)** and **Transformers**, carefully selected to optimize both local and global feature extraction. The architecture includes the following key configurations:
- **CNN + Transformer**: CNNs are used for efficient local feature extraction, while Transformers capture long-range dependencies across the entire image, improving overall context understanding.
- **ShuffleNet + Transformer**: A lightweight network architecture that reduces computational complexity, while Transformer layers help in modeling global features effectively.
- **EfficientNet + Transformer**: This configuration optimizes performance and reduces computational cost, combining EfficientNet's efficiency with Transformer’s global modeling power.
- **MobileNet + Transformer**: Ideal for resource-constrained environments, this combination ensures efficient processing without sacrificing feature extraction capability.

![Network Architecture](https://github.com/GZHUone/psf-zernike-model/blob/main/PSF%20Zernike.jpg)

### 2. Image Preprocessing Techniques

#### **Denoising Filters**  
The model applies three powerful denoising techniques to improve image quality:
- **Gaussian Filter**: Smooths the image by blurring it, effectively reducing noise.
- **Mean Filter**: Each pixel is replaced with the average of its neighboring pixels, removing noise and improving overall image quality.
- **Median Filter**: Replaces each pixel with the median of its neighboring pixels, effectively removing salt-and-pepper noise while preserving edge details.

#### **Sharpening Enhancement Algorithms**  
To enhance image details and edges, the model employs the following sharpening techniques:
- **High-Pass Filter Sharpening**: Emphasizes high-frequency components, making fine details more prominent.
- **Laplacian Sharpening**: Uses the Laplacian operator to detect edges and sharpen the image.
- **Unsharp Masking**: Subtracts a blurred version of the image from the original to increase contrast and bring out fine details.

![PSF Extraction and Cropping](https://github.com/GZHUone/psf-zernike-model/blob/main/processs.png)

---

