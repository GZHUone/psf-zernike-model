# psf-zernike-model
本项目开发了一个基于PyTorch的网络架构——PSF Zernike。在图像输入阶段，首先进行PSF提取与裁剪，然后应用三种滤波去噪方法（高斯滤波、均值滤波和中值滤波）以及三种锐化增强算法（高通滤波锐化、拉普拉斯锐化和反锐化掩蔽）。此外，网络架构结合了多种模型组合，包括：CNN+Transformer、ShuffleNet+Transformer、EfficientNet+Transformer 和 MobileNet+Transformer，以增强网络的性能和图像处理能力。
