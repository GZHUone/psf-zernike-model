# PSF Zernike Model

This project develops a PyTorch-based network architecture — PSF Zernike, designed for image processing tasks. The model combines various image preprocessing techniques and powerful network architectures to enhance denoising and sharpening of images.

## Project Overview

1. **PSF Extraction and Cropping**  
   In the image input phase, the PSF (Point Spread Function) is extracted and cropped before further processing.

2. **Preprocessing Methods**  
   The model applies three denoising filters:  
   - **Gaussian Filter**: Blurs the image to reduce noise.  
   - **Mean Filter**: Removes noise by replacing a pixel with the average of its neighboring pixels.  
   - **Median Filter**: Uses the median of the neighboring pixels to replace the target pixel, effective for salt-and-pepper noise.

   The model also applies three sharpening enhancement algorithms:  
   - **High-Pass Filter Sharpening**: Enhances image details by emphasizing high-frequency components.  
   - **Laplacian Sharpening**: Uses the Laplacian operator to sharpen the image.  
   - **Unsharp Masking**: Enhances contrast by subtracting a blurred version of the image from the original.

   **Preprocessing Example**:  
   ![Preprocessing Input Image](https://github.com/GZHUone/psf-zernike-model/blob/main/processs.png)

3. **Core Network Architecture**  
   The network architecture is based on the following combinations, leveraging the strengths of convolutional neural networks and Transformer models:
   - **CNN + Transformer**: Combines traditional CNNs for local feature extraction with Transformers for capturing global dependencies.
   - **ShuffleNet + Transformer**: ShuffleNet is a lightweight architecture that reduces computational complexity, while Transformer enhances global feature modeling.
   - **EfficientNet + Transformer**: EfficientNet optimizes the network structure for high performance with low computational cost, and Transformer further enhances feature extraction.
   - **MobileNet + Transformer**: MobileNet is an efficient architecture for resource-constrained environments, and Transformer boosts global feature modeling.

4. **Network Architecture Diagram**  
   The following diagram shows the overall structure of the PSF Zernike model, including connections between modules and data flow:  
   ![Network Architecture](https://github.com/GZHUone/psf-zernike-model/blob/main/PSF%20Zernike.jpg)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/GZHUone/psf-zernike-model.git

