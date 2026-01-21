# src/model/dynamic_head.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicMaskHead(nn.Module):
    def __init__(self, in_channels=96, text_dim=512, mask_channels=64):
        """
        in_channels: Số kênh của Feature Map ảnh (F1 của ConvNeXt là 96)
        text_dim: Số kênh của Vector text (512)
        mask_channels: Số kênh trung gian của mask
        """
        super().__init__()
        
        # 1. Giảm chiều feature map ảnh cho nhẹ
        self.vis_project = nn.Sequential(
            nn.Conv2d(in_channels, mask_channels, kernel_size=1),
            nn.GroupNorm(8, mask_channels),
            nn.ReLU()
        ) # [Batch, 96, H, W] -> [Batch, 64, H, W]
        
        # 2. CONTROLLER: Sinh weights từ text
        # Chúng ta cần sinh weights cho 1 lớp Conv 1x1: (mask_channels -> 1)
        # Số tham số cần sinh = (Kernel_Size) + Bias 
        # Weights shape: [1, mask_channels, 1, 1] -> Số phần tử: mask_channels
        # Bias shape: [1]
        self.controller = nn.Linear(text_dim, mask_channels + 1) 

    def forward(self, vision_features, text_embedding):
        """
        vision_features: [Batch, C, H, W] (Lấy feature map to nhất F1)
        text_embedding: [Batch, Text_Dim]
        """
        # A. Chuẩn bị ảnh
        x = self.vis_project(vision_features) # [B, 64, H, W]
        B, C, H, W = x.shape
        
        # B. Sinh weights động
        params = self.controller(text_embedding) # [B, 65]
        
        # Tách weights và bias
        dynamic_weight = params[:, :C] # [B, 64]
        dynamic_bias = params[:, C]    # [B]
        
        # C. Thực hiện Dynamic Convolution
        # Vì mỗi sample trong batch dùng một bộ weight khác nhau,
        # ta reshape để dùng chức năng Group Conv của PyTorch cho nhanh.
        
        # Reshape input: [1, B*C, H, W]
        x_reshaped = x.view(1, B * C, H, W)
        
        # Reshape weights: [B, C, 1, 1] -> [B, C, 1, 1] (Group=B)
        weight_reshaped = dynamic_weight.view(B, C, 1, 1)
        
        # Conv2d với groups=B
        # Output: [1, B, H, W]
        mask_logits = F.conv2d(x_reshaped, weight_reshaped, bias=None, groups=B)
        mask_logits = mask_logits.view(B, 1, H, W)
        
        # Cộng bias
        mask_logits = mask_logits + dynamic_bias.view(B, 1, 1, 1)
        
        # Upsample về kích thước gốc (vì F1 đang nhỏ hơn ảnh gốc 4 lần)
        mask_out = F.interpolate(mask_logits, scale_factor=4, mode='bilinear', align_corners=False)
        
        return mask_out # [Batch, 1, Origin_H, Origin_W]