# PSF Zernike Model

Welcome to **PSF Zernike** — a **PyTorch-based network architecture** engineered to revolutionize **image processing**. This groundbreaking model seamlessly blends advanced **denoising** and **sharpening** techniques with **state-of-the-art deep learning architectures**, setting new standards for image quality enhancement.

---

### 🔥 **Key Innovations**

#### 1. **PSF Extraction & Precision Cropping**
   The journey begins with **PSF (Point Spread Function)** extraction and precision cropping, ensuring that every pixel is primed for optimal enhancement. This crucial step lays the foundation for unparalleled image processing accuracy.

   ![PSF Extraction and Cropping](https://github.com/GZHUone/psf-zernike-model/blob/main/processs.png)

#### 2. **Advanced Denoising Filters for Clean Perfection**
   The model employs **three advanced denoising filters**, each designed to tackle noise with surgical precision:
   - **Gaussian Denoising**: Smoothens images by strategically blurring noise while preserving vital image features.
   - **Mean Denoising**: Removes noise by averaging neighboring pixels, leaving clean and smooth transitions.
   - **Median Denoising**: Targets salt-and-pepper noise, replacing each pixel with the median of its neighbors, maintaining sharp edges.

   **Preprocessing Example**:  
   ![Denoising Filters](https://github.com/GZHUone/psf-zernike-model/blob/main/PSF%20Zernike.jpg)

#### 3. **Unleashing Image Sharpness with Cutting-Edge Enhancement**
   Elevating image quality is a science, and the model integrates three of the most powerful **sharpening algorithms** available:
   - **High-Pass Filter Sharpening**: Highlights fine details by amplifying high-frequency components, giving clarity to every edge.
   - **Laplacian Sharpening**: Boosts sharpness by focusing on high-contrast edges, ensuring precision where it's needed most.
   - **Unsharp Masking**: Subtracts a blurred version of the image to boost contrast and fine details.

#### 4. **Next-Generation Network Architecture for Unmatched Performance**
   We’ve pushed the envelope by combining powerful **CNNs** with cutting-edge **Transformer models**. This results in a flexible, scalable architecture that can adapt to any image processing challenge:
   - **CNN + Transformer**: A perfect fusion, where CNNs extract detailed local features while Transformers capture global dependencies, enabling deep understanding of the entire image.
   - **ShuffleNet + Transformer**: A lightweight architecture that maintains computational efficiency while enhancing global feature modeling.
   - **EfficientNet + Transformer**: Optimized for both speed and accuracy, this architecture guarantees high performance with minimal resource consumption, while Transformers provide the final edge in global feature learning.
   - **MobileNet + Transformer**: Tailored for resource-constrained environments, this combination ensures rapid, high-quality processing even on edge devices.

   ![Network Architecture](https://github.com/GZHUone/psf-zernike-model/blob/main/PSF%20Zernike.jpg)

---

### 🚀 **Why PSF Zernike?**

PSF Zernike doesn’t just enhance images; it **redefines** how we approach image processing. With precision at its core, this model isn’t just a tool, but a breakthrough in leveraging both traditional and modern techniques to extract the best possible image quality. Whether you're improving clarity, removing noise, or sharpening fine details, **PSF Zernike** offers a seamless, scalable solution.

---

By incorporating powerful architectures, advanced image processing techniques, and leveraging the latest in deep learning, **PSF Zernike** is engineered for **maximum performance** and **unprecedented image quality**. It’s time to elevate your image processing workflows to the next level.

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/GZHUone/psf-zernike-model.git
