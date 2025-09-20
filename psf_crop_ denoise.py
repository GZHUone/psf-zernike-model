#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成随机PSF样本并进行中心定位提取、滤波去噪（高斯、均值、中值滤波）、锐化增强的预处理脚本
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def generate_large_psf_image(size=(800, 800), psf_center=None, psf_radius=60, 
                           noise_level=0.1, blur_level=5):
    """
    生成包含单个PSF的大图像，并返回原始图像
    """
    print(f"生成大图像: 尺寸{size}, PSF半径{psf_radius}")
    
    # 创建黑色背景
    image = np.zeros(size, dtype=np.uint8)
    
    # 随机生成PSF中心位置
    if psf_center is None:
        margin = psf_radius + 50
        psf_center = (
            np.random.randint(margin, size[1] - margin),
            np.random.randint(margin, size[0] - margin)
        )
    
    print(f"PSF中心位置: {psf_center}")
    
    # 创建坐标网格
    y, x = np.ogrid[:size[0], :size[1]]
    
    # 计算到PSF中心的距离
    distance = np.sqrt((x - psf_center[0])**2 + (y - psf_center[1])**2)
    
    # 创建高斯PSF
    sigma = psf_radius / 3  # 3σ原则
    psf = np.exp(-(distance**2) / (2 * sigma**2))
    
    # 添加噪声
    noise = np.random.normal(0, noise_level, psf.shape)
    psf_noisy = np.clip(psf + noise, 0, 1)
    
    # 转换为8位图像
    psf_image = (psf_noisy * 255).astype(np.uint8)
    
    # 添加模糊
    if blur_level > 0:
        psf_image = cv2.GaussianBlur(psf_image, (blur_level*2+1, blur_level*2+1), 0)
    
    return psf_image, psf_center


def center_crop_image(image, center, crop_radius=100):
    """
    根据PSF的中心位置裁剪图像（放大后的区域）
    """
    x, y = center
    cropped_image = image[max(y - crop_radius, 0):y + crop_radius, max(x - crop_radius, 0):x + crop_radius]
    return cropped_image


def denoise_image(image, method='gaussian'):
    """
    对图像进行去噪处理，包括高斯滤波、均值滤波、中值滤波
    """
    if method == 'gaussian':
        denoised_image = cv2.GaussianBlur(image, (5, 5), 0)
    elif method == 'mean':
        denoised_image = cv2.blur(image, (5, 5))
    elif method == 'median':
        denoised_image = cv2.medianBlur(image, 5)
    else:
        raise ValueError("Invalid denoising method. Choose 'gaussian', 'mean' or 'median'.")
    
    return denoised_image


def sharpen_image(image):
    """
    锐化图像以增强细节
    """
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image


def resize_image(image, size=(256, 256)):
    """
    将图像resize为指定大小
    """
    resized_image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    return resized_image


def save_image(image, filename):
    """
    保存图像到文件
    """
    cv2.imwrite(filename, image)
    print(f"已保存图像: {filename}")


def main():
    """
    主测试函数
    """
    print("PSF预处理脚本测试")
    print("=" * 50)
    
    try:
        # 1. 生成单个PSF并进行处理
        test_image, psf_center = generate_large_psf_image(
            size=(600, 600),
            psf_center=(300, 250),  # 固定位置便于测试
            psf_radius=80,
            noise_level=0.08,
            blur_level=3
        )
        
        # 保存原始图像
        save_image(test_image, 'single_psf_original.png')
        
        # 中心定位提取并裁剪图像
        cropped_image = center_crop_image(test_image, psf_center, crop_radius=100)
        save_image(cropped_image, 'single_psf_cropped.png')
        
        # 对图像分别进行去噪处理：高斯滤波、均值滤波、中值滤波
        gaussian_denoised = denoise_image(cropped_image, method='gaussian')
        mean_denoised = denoise_image(cropped_image, method='mean')
        median_denoised = denoise_image(cropped_image, method='median')
        
        # 将每个去噪后的图像resize为256x256
        gaussian_resized = resize_image(gaussian_denoised)
        mean_resized = resize_image(mean_denoised)
        median_resized = resize_image(median_denoised)
        
        # 保存每个处理结果
        save_image(gaussian_resized, 'single_psf_gaussian_denoised_resized.png')
        save_image(mean_resized, 'single_psf_mean_denoised_resized.png')
        save_image(median_resized, 'single_psf_median_denoised_resized.png')
        
        # 锐化增强处理
        sharpened_image = sharpen_image(gaussian_resized)  # 用高斯去噪后的图像进行锐化
        save_image(sharpened_image, 'single_psf_sharpened.png')
        
        print("单个PSF处理完成，图像已保存")

    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
