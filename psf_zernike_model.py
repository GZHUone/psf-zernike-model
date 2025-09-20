import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn import MultiheadAttention
from typing import Tuple, List
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class PreprocessingModule(nn.Module):
    """预处理模块：针对5张图像concat后的输入"""
    
    def __init__(self, input_size: int = 256, num_images: int = 5, channels_per_image: int = 1):
        super(PreprocessingModule, self).__init__()
        self.input_size = input_size
        self.num_images = num_images
        self.channels_per_image = channels_per_image
        self.total_channels = num_images * channels_per_image
        
        # 对于多通道输入的预处理
        self.normalization = nn.BatchNorm2d(self.total_channels)
        
        # 通道注意力机制
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.total_channels, self.total_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.total_channels // 4, self.total_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 批量归一化
        x = self.normalization(x)
        
        # 通道注意力
        attention_weights = self.channel_attention(x)
        x = x * attention_weights
        
        return x


class ShuffleNetV2(nn.Module):
    """ShuffleNetV2特征提取"""
    def __init__(self, inp, oup, stride):  # inp: 输入通道数, oup: 输出通道数, stride: 步长
        super().__init__()

        self.stride = stride

        # 计算每个分支的通道数
        branch_features = oup // 2
        # 确保步长为1时输入通道数是分支通道数的两倍
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride == 2:
            # 定义 branch1，当步长为2时
            self.branch1 = nn.Sequential(
                # 深度卷积，输入通道数等于输出通道数，步长为2
                nn.Conv2d(inp, inp, kernel_size=3, stride=self.stride, padding=1, groups=inp),
                nn.BatchNorm2d(inp),
                # 1x1 卷积，输出通道数等于 branch_features
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True))
        else:
            # 步长为1时，branch1 为空
            self.branch1 = nn.Sequential()

        # 定义 branch2
        self.branch2 = nn.Sequential(
            # 1x1 卷积，步长为1，输出通道数等于 branch_features
            nn.Conv2d(inp if (self.stride == 2) else branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            # 深度卷积，步长为 stride，输出通道数等于 branch_features
            nn.Conv2d(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1, groups=branch_features),
            nn.BatchNorm2d(branch_features),
            # 另一个 1x1 卷积，步长为1
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        if self.stride == 1:
            # 当步长为1时，将输入在通道维度上分成两部分
            x1, x2 = x.chunk(2, dim=1)
            # 连接 x1 和 branch2 处理后的 x2
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            # 当步长为2时，连接 branch1 和 branch2 的输出
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        # 进行通道混洗
        out = self.channel_shuffle(out, 2)

        return out

    def channel_shuffle(self, x, groups):
        # 获取输入张量的形状信息
        N, C, H, W = x.size()
        # 调整张量的形状，并交换通道维度
        out = x.view(N, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)
        return out


class MultiImageShuffleNetExtractor(nn.Module):
    """使用ShuffleNet特征提取器"""
    
    def __init__(self, total_input_channels: int, num_images: int = 5):
        super(MultiImageShuffleNetExtractor, self).__init__()
        self.total_input_channels = total_input_channels
        self.num_images = num_images
        
        # 初始卷积层
        self.initial_conv = nn.Sequential(
            nn.Conv2d(total_input_channels, 24, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )
        
        # ShuffleNet特征提取层
        self.feature_layers = nn.Sequential(
            ShuffleNetV2(inp=24, oup=48, stride=2),
            ShuffleNetV2(inp=48, oup=96, stride=2),
            ShuffleNetV2(inp=96, oup=192, stride=2),
        )
        
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 特征重组：将192维特征转换为num_images个图像特征
        self.feature_reorganizer = nn.Linear(192, num_images * 128)  # 每个图像128维特征
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # 初始卷积
        x = self.initial_conv(x)  # [B, 24, H/2, W/2]
        
        # 特征提取
        x = self.feature_layers(x)  # [B, 192, H/16, W/16]
        
        # 全局平均池化
        x = self.global_avg_pool(x)  # [B, 192, 1, 1]
        x = x.flatten(1)  # [B, 192]
        
        # 特征重组为图像序列
        features = self.feature_reorganizer(x)  # [B, num_images * 128]
        features = features.view(batch_size, self.num_images, 128)  # [B, num_images, 128]
        
        return features


class MultiImageCNNExtractor(nn.Module):
    """专门处理concat后多图像输入的CNN特征提取器"""
    
    def __init__(self, total_input_channels: int, num_images: int = 5):
        super(MultiImageCNNExtractor, self).__init__()
        self.total_input_channels = total_input_channels
        self.num_images = num_images
        
        # 初始卷积层
        self.initial_conv = nn.Sequential(
            nn.Conv2d(total_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # 特征提取层
        self.feature_layers = nn.Sequential(
            # 第一组卷积块
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 第二组卷积块
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 第三组卷积块
            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 特征重组：将512维特征转换为num_images个图像特征
        self.feature_reorganizer = nn.Linear(512, num_images * 128)  # 每个图像128维特征
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # 初始卷积
        x = self.initial_conv(x)  # [B, 64, H/4, W/4]
        
        # 特征提取
        x = self.feature_layers(x)  # [B, 512, H/32, W/32]
        
        # 全局平均池化
        x = self.global_avg_pool(x)  # [B, 512, 1, 1]
        x = x.flatten(1)  # [B, 512]
        
        # 特征重组为图像序列
        features = self.feature_reorganizer(x)  # [B, num_images * 128]
        features = features.view(batch_size, self.num_images, 128)  # [B, num_images, 128]
        
        return features


class MultiImageEfficientNetExtractor(nn.Module):
    """使用EfficientNet作为特征提取器"""

    def __init__(self, total_input_channels: int, num_images: int = 5):
        super(MultiImageEfficientNetExtractor, self).__init__()
        self.total_input_channels = total_input_channels
        self.num_images = num_images

        # 加载预训练的 EfficientNet 模型
        self.efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # 替换输入通道数（默认 EfficientNet 输入通道为 3）
        if total_input_channels != 3:
            self.efficientnet.features[0][0] = nn.Conv2d(
                total_input_channels, 32, kernel_size=3, stride=2, padding=1, bias=False
            )

        # 获取 EfficientNet 的输出特征维度
        self.feature_dim = self.efficientnet.classifier[1].in_features

        # 替换分类层
        self.efficientnet.classifier = nn.Identity()  # 去掉分类层

        # 特征重组：将 EfficientNet 的输出特征转换为 num_images 个图像特征
        self.feature_reorganizer = nn.Linear(self.feature_dim, num_images * 128)  # 每个图像 128 维特征

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        # 使用 EfficientNet 提取特征
        x = self.efficientnet(x)  # [B, feature_dim]

        # 特征重组为图像序列
        features = self.feature_reorganizer(x)  # [B, num_images * 128]
        features = features.view(batch_size, self.num_images, 128)  # [B, num_images, 128]

        return features


class MultiImageMobileNetExtractor(nn.Module):
    """使用MobileNet作为特征提取器"""

    def __init__(self, total_input_channels: int, num_images: int = 5):
        super(MultiImageMobileNetExtractor, self).__init__()
        self.total_input_channels = total_input_channels
        self.num_images = num_images

        # 加载预训练的 MobileNet 模型
        self.mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

        # 替换输入通道数（默认 MobileNet 输入通道为 3）
        if total_input_channels != 3:
            self.mobilenet.features[0][0] = nn.Conv2d(
                total_input_channels, 32, kernel_size=3, stride=2, padding=1, bias=False
            )

        # 获取 MobileNet 的输出特征维度
        self.feature_dim = self.mobilenet.last_channel

        # 替换分类层
        self.mobilenet.classifier = nn.Identity()  # 去掉分类层

        # 特征重组：将 MobileNet 的输出特征转换为 num_images 个图像特征
        self.feature_reorganizer = nn.Linear(self.feature_dim, num_images * 128)  # 每个图像 128 维特征

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        # 使用 MobileNet 提取特征
        x = self.mobilenet(x)  # [B, feature_dim]

        # 特征重组为图像序列
        features = self.feature_reorganizer(x)  # [B, num_images * 128]
        features = features.view(batch_size, self.num_images, 128)  # [B, num_images, 128]

        return features


class ImageSequenceTransformer(nn.Module):
    """处理图像序列的Transformer"""
    
    def __init__(self, d_model: int = 128, nhead: int = 8, num_layers: int = 4, num_images: int = 5):
        super(ImageSequenceTransformer, self).__init__()
        
        self.d_model = d_model
        self.num_images = num_images
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, num_images, d_model))
        
        # 输入投影到更高维度
        self.input_projection = nn.Linear(d_model, 256)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出投影回原维度
        self.output_projection = nn.Linear(256, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 添加位置编码
        x = x + self.pos_encoding
        
        # 投影到高维
        x = self.input_projection(x)
        
        # Transformer处理
        x = self.transformer(x)
        
        # 投影回原维度
        x = self.output_projection(x)
        
        return x


# class PSFZernikeModel(nn.Module):
#     """完整的PSF到Zernike系数的模型 - 批处理版本 cnn + transformer"""
    
#     def __init__(self, input_size: int = 256, num_images: int = 5, 
#                  channels_per_image: int = 1, num_coefficients: int = 36):
#         super(PSFZernikeModel, self).__init__()
        
#         self.input_size = input_size
#         self.num_images = num_images
#         self.channels_per_image = channels_per_image
#         self.num_coefficients = num_coefficients
#         self.total_channels = num_images * channels_per_image
        
#         # 预处理模块
#         self.preprocessing = PreprocessingModule(input_size, num_images, channels_per_image)
        
#         # CNN特征提取器
#         self.cnn_extractor = MultiImageCNNExtractor(self.total_channels, num_images)
        
#         # Transformer处理器
#         self.transformer = ImageSequenceTransformer(d_model=128, nhead=8, num_layers=4, num_images=num_images)
        
#         # 输出层：为每个图像生成Zernike系数
#         self.output_heads = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(128, 64),
#                 nn.ReLU(inplace=True),
#                 nn.Dropout(0.1),
#                 nn.Linear(64, num_coefficients)
#             )
#             for _ in range(num_images)
#         ])
        
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         前向传播
        
#         Args:
#             x: 输入图像
#                - 灰度模式: [B, 5, 256, 256]
#                - RGB模式: [B, 15, 256, 256]
               
#         Returns:
#             output: [B, 5, 36] - 每个样本的5组Zernike系数
#         """
#         batch_size = x.size(0)
        
#         # 验证输入通道数
#         if x.size(1) != self.total_channels:
#             raise ValueError(f"Expected {self.total_channels} channels, got {x.size(1)}")
        
#         # 预处理
#         x = self.preprocessing(x)
        
#         # CNN特征提取 -> [B, num_images, 128]
#         image_features = self.cnn_extractor(x)
        
#         # Transformer处理图像序列
#         enhanced_features = self.transformer(image_features)  # [B, num_images, 128]
        
#         # 为每个图像生成Zernike系数
#         outputs = []
#         for i, head in enumerate(self.output_heads):
#             # 取出第i个图像的特征
#             image_feature = enhanced_features[:, i, :]  # [B, 128]
#             # 生成系数
#             coeffs = head(image_feature)  # [B, 36]
#             outputs.append(coeffs)
        
#         # 堆叠所有输出 [B, num_images, num_coefficients]
#         output = torch.stack(outputs, dim=1)  # [B, 5, 36]
        
#         return output


# class PSFZernikeModel(nn.Module):
#     """完整的PSF到Zernike系数的模型 - 批处理版本 efficientnet + transformer"""

#     def __init__(self, input_size: int = 256, num_images: int = 5,
#                  channels_per_image: int = 1, num_coefficients: int = 36):
#         super(PSFZernikeModel, self).__init__()

#         self.input_size = input_size
#         self.num_images = num_images
#         self.channels_per_image = channels_per_image
#         self.num_coefficients = num_coefficients
#         self.total_channels = num_images * channels_per_image

#         # 预处理模块
#         self.preprocessing = PreprocessingModule(input_size, num_images, channels_per_image)

#         # 使用 EfficientNet 特征提取器
#         self.cnn_extractor = MultiImageEfficientNetExtractor(self.total_channels, num_images)

#         # Transformer 处理器
#         self.transformer = ImageSequenceTransformer(d_model=128, nhead=8, num_layers=4, num_images=num_images)

#         # 输出层：为每个图像生成 Zernike 系数
#         self.output_heads = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(128, 64),
#                 nn.ReLU(inplace=True),
#                 nn.Dropout(0.1),
#                 nn.Linear(64, num_coefficients)
#             )
#             for _ in range(num_images)
#         ])

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         前向传播

#         Args:
#             x: 输入图像
#                - 灰度模式: [B, 5, 256, 256]
#                - RGB模式: [B, 15, 256, 256]

#         Returns:
#             output: [B, 5, 36] - 每个样本的 5 组 Zernike 系数
#         """
#         batch_size = x.size(0)

#         # 验证输入通道数
#         if x.size(1) != self.total_channels:
#             raise ValueError(f"Expected {self.total_channels} channels, got {x.size(1)}")

#         # 预处理
#         x = self.preprocessing(x)

#         # EfficientNet 特征提取 -> [B, num_images, 128]
#         image_features = self.cnn_extractor(x)

#         # Transformer 处理图像序列
#         enhanced_features = self.transformer(image_features)  # [B, num_images, 128]

#         # 为每个图像生成 Zernike 系数
#         outputs = []
#         for i, head in enumerate(self.output_heads):
#             # 取出第 i 个图像的特征
#             image_feature = enhanced_features[:, i, :]  # [B, 128]
#             # 生成系数
#             coeffs = head(image_feature)  # [B, 36]
#             outputs.append(coeffs)

#         # 堆叠所有输出 [B, num_images, num_coefficients]
#         output = torch.stack(outputs, dim=1)  # [B, 5, 36]

#         return output


# class PSFZernikeModel(nn.Module):
#     """完整的PSF到Zernike系数的模型 - 批处理版本 shufflenet + transformer"""

#     def __init__(self, input_size: int = 256, num_images: int = 5,
#                  channels_per_image: int = 1, num_coefficients: int = 36):
#         super(PSFZernikeModel, self).__init__()

#         self.input_size = input_size
#         self.num_images = num_images
#         self.channels_per_image = channels_per_image
#         self.num_coefficients = num_coefficients
#         self.total_channels = num_images * channels_per_image

#         # 预处理模块
#         self.preprocessing = PreprocessingModule(input_size, num_images, channels_per_image)

#         # 使用ShuffleNet特征提取器
#         self.cnn_extractor = MultiImageShuffleNetExtractor(self.total_channels, num_images)

#         # Transformer处理器
#         self.transformer = ImageSequenceTransformer(d_model=128, nhead=8, num_layers=4, num_images=num_images)

#         # 输出层：为每个图像生成Zernike系数
#         self.output_heads = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(128, 64),
#                 nn.ReLU(inplace=True),
#                 nn.Dropout(0.1),
#                 nn.Linear(64, num_coefficients)
#             )
#             for _ in range(num_images)
#         ])

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         前向传播

#         Args:
#             x: 输入图像
#                - 灰度模式: [B, 5, 256, 256]
#                - RGB模式: [B, 15, 256, 256]

#         Returns:
#             output: [B, 5, 36] - 每个样本的5组Zernike系数
#         """
#         batch_size = x.size(0)

#         # 验证输入通道数
#         if x.size(1) != self.total_channels:
#             raise ValueError(f"Expected {self.total_channels} channels, got {x.size(1)}")

#         # 预处理
#         x = self.preprocessing(x)

#         # CNN特征提取 -> [B, num_images, 128]
#         image_features = self.cnn_extractor(x)

#         # Transformer处理图像序列
#         enhanced_features = self.transformer(image_features)  # [B, num_images, 128]

#         # 为每个图像生成Zernike系数
#         outputs = []
#         for i, head in enumerate(self.output_heads):
#             # 取出第i个图像的特征
#             image_feature = enhanced_features[:, i, :]  # [B, 128]
#             # 生成系数
#             coeffs = head(image_feature)  # [B, 36]
#             outputs.append(coeffs)

#         # 堆叠所有输出 [B, num_images, num_coefficients]
#         output = torch.stack(outputs, dim=1)  # [B, 5, 36]

#         return output


class PSFZernikeModel(nn.Module):
    """完整的PSF到Zernike系数的模型 - 批处理版本 MobileNet + Transformer"""

    def __init__(self, input_size: int = 256, num_images: int = 5,
                 channels_per_image: int = 1, num_coefficients: int = 36):
        super(PSFZernikeModel, self).__init__()

        self.input_size = input_size
        self.num_images = num_images
        self.channels_per_image = channels_per_image
        self.num_coefficients = num_coefficients
        self.total_channels = num_images * channels_per_image

        # 预处理模块
        self.preprocessing = PreprocessingModule(input_size, num_images, channels_per_image)

        # 使用 MobileNet 特征提取器
        self.cnn_extractor = MultiImageMobileNetExtractor(self.total_channels, num_images)

        # Transformer 处理器
        self.transformer = ImageSequenceTransformer(d_model=128, nhead=8, num_layers=4, num_images=num_images)

        # 输出层：为每个图像生成 Zernike 系数
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(64, num_coefficients)
            )
            for _ in range(num_images)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入图像
               - 灰度模式: [B, 5, 256, 256]
               - RGB模式: [B, 15, 256, 256]

        Returns:
            output: [B, 5, 36] - 每个样本的 5 组 Zernike 系数
        """
        batch_size = x.size(0)

        # 验证输入通道数
        if x.size(1) != self.total_channels:
            raise ValueError(f"Expected {self.total_channels} channels, got {x.size(1)}")

        # 预处理
        x = self.preprocessing(x)

        # MobileNet 特征提取 -> [B, num_images, 128]
        image_features = self.cnn_extractor(x)

        # Transformer 处理图像序列
        enhanced_features = self.transformer(image_features)  # [B, num_images, 128]

        # 为每个图像生成 Zernike 系数
        outputs = []
        for i, head in enumerate(self.output_heads):
            # 取出第 i 个图像的特征
            image_feature = enhanced_features[:, i, :]  # [B, 128]
            # 生成系数
            coeffs = head(image_feature)  # [B, 36]
            outputs.append(coeffs)

        # 堆叠所有输出 [B, num_images, num_coefficients]
        output = torch.stack(outputs, dim=1)  # [B, 5, 36]

        return output


def create_model(input_size: int = 256, num_images: int = 5, 
                channels_per_image: int = 1, num_coefficients: int = 36) -> PSFZernikeModel:
    """
    创建模型实例
    
    Args:
        input_size: 输入图像尺寸
        num_images: 输入图像数量
        channels_per_image: 每张图像通道数 (1=灰度, 3=RGB)
        num_coefficients: 每张图像的Zernike系数数量
        
    Returns:
        PSFZernikeModel实例
    """
    return PSFZernikeModel(
        input_size=input_size,
        num_images=num_images,
        channels_per_image=channels_per_image,
        num_coefficients=num_coefficients
    )


def create_data_batch(psf_images_list: List[torch.Tensor], mode: str = 'grayscale') -> torch.Tensor:
    """
    将5张单独的PSF图像组合成批处理格式
    
    Args:
        psf_images_list: 包含5张PSF图像的列表，每张图像形状为[H, W] or [C, H, W]
        mode: 'grayscale' 或 'rgb'
        
    Returns:
        tensor: [1, total_channels, H, W]
    """
    if len(psf_images_list) != 5:
        raise ValueError("需要恰好5张PSF图像")
    
    processed_images = []
    for img in psf_images_list:
        if img.dim() == 2:  # [H, W]
            img = img.unsqueeze(0)  # [1, H, W]
        elif img.dim() == 3 and mode == 'grayscale':  # [C, H, W] -> [1, H, W]
            if img.size(0) == 3:  # RGB转灰度
                img = torch.mean(img, dim=0, keepdim=True)
        processed_images.append(img)
    
    # 在通道维度concat
    batch_data = torch.cat(processed_images, dim=0)  # [5 or 15, H, W]
    batch_data = batch_data.unsqueeze(0)  # [1, 5 or 15, H, W]
    
    return batch_data


if __name__ == "__main__":
    # 测试灰度模式
    print("=== 灰度模式测试 ===")
    model_gray = create_model(channels_per_image=1)
    x_gray = torch.randn(2, 5, 256, 256)  # 2个样本，每个5张灰度图像(5*1=5通道)
    
    print(f"模型参数量: {sum(p.numel() for p in model_gray.parameters()):,}")
    print(f"输入形状: {x_gray.shape}")
    
    with torch.no_grad():
        output_gray = model_gray(x_gray)
        print(f"输出形状: {output_gray.shape}")
        print(f"输出统计: min={output_gray.min():.4f}, max={output_gray.max():.4f}")
    
    # 测试RGB模式
    print("\n=== RGB模式测试 ===")
    model_rgb = create_model(channels_per_image=3)
    x_rgb = torch.randn(2, 15, 256, 256)  # 2个样本，每个5张RGB图像(5*3=15通道)
    
    print(f"输入形状: {x_rgb.shape}")
    
    with torch.no_grad():
        output_rgb = model_rgb(x_rgb)
        print(f"输出形状: {output_rgb.shape}")
        print(f"输出统计: min={output_rgb.min():.4f}, max={output_rgb.max():.4f}")
    
    # 测试数据预处理函数
    print("\n=== 数据预处理测试 ===")
    sample_images = [torch.randn(256, 256) for _ in range(5)]  # 5张灰度图像
    batch_data = create_data_batch(sample_images, mode='grayscale')
    print(f"预处理后形状: {batch_data.shape}")
    
    with torch.no_grad():
        result = model_gray(batch_data)
        print(f"预测结果形状: {result.shape}")
        print("每个图像的前3个系数:")
        for i in range(5):
            coeffs = result[0, i, :3].tolist()
            print(f"  图像{i+1}: {coeffs}")
