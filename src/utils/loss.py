# src/utils/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Focal Loss thay thế BCE
class SigmoidFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: Logits, targets: 0/1
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

# 2. Dice Loss (Giữ nguyên)
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1 - dice

# 3. Edge Loss (Giữ nguyên)
class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.l1_loss = nn.L1Loss()

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        pred_grad_x = F.conv2d(probs, self.sobel_x, padding=1)
        pred_grad_y = F.conv2d(probs, self.sobel_y, padding=1)
        pred_edge = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-6)
        
        target_grad_x = F.conv2d(targets, self.sobel_x, padding=1)
        target_grad_y = F.conv2d(targets, self.sobel_y, padding=1)
        target_edge = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-6)
        
        return self.l1_loss(pred_edge, target_edge)

# 4. OmniLoss Tổng hợp
class OmniLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.weights = cfg['train']['loss_weights']
        
        # --- THAY ĐỔI LỚN ---
        self.focal_loss = SigmoidFocalLoss(alpha=0.25, gamma=2.0) # Thay BCE
        self.dice_loss = DiceLoss()
        self.edge_loss = EdgeLoss()
        
        self.cap_loss = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, outputs, batch):
        task_type = batch['task_type'][0]
        total_loss = 0.0
        logs = {}

        if task_type == "segment":
            pred = outputs['mask_logits']
            target = batch['target_mask'].to(pred.device)
            
            if pred.shape[-1] != target.shape[-1]:
                target = F.interpolate(target, size=pred.shape[-2:], mode='nearest')
            
            # Loss Component 1: Focal Loss (Pixel Balance)
            l_focal = self.focal_loss(pred, target)
            
            # Loss Component 2: Dice Loss (Region Balance)
            l_dice = self.dice_loss(pred, target)
            
            # Loss Component 3: Edge Loss (Boundary Sharpening)
            l_edge = self.edge_loss(pred, target)
            
            # Tổng hợp: Tăng trọng số cho Dice vì nó quan trọng nhất
            seg_loss_total = l_focal + 2.0 * l_dice + 0.5 * l_edge
            
            total_loss += seg_loss_total * self.weights['segment']
            
            logs['seg_loss'] = seg_loss_total.item()
            logs['seg_focal'] = l_focal.item()
            logs['seg_dice'] = l_dice.item()

        elif task_type == "caption":
            logits = outputs['caption_logits']
            targets = batch['target_ids'].to(logits.device)[:, 1:]
            loss = self.cap_loss(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            total_loss += loss * self.weights['caption']
            logs['cap_loss'] = loss.item()

        return total_loss, logs