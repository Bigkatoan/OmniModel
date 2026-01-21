# src/utils/visualization.py

import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def denormalize(img_tensor):
    """
    Chuyển Tensor [C,H,W] về lại ảnh numpy [H,W,C] 0-255.
    FIX: Thêm np.ascontiguousarray để tránh lỗi OpenCV layout.
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # 1. Chuyển về CPU và numpy
    img = img_tensor.permute(1, 2, 0).cpu().detach().numpy()
    
    # 2. Denormalize
    img = img * std + mean
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    
    # 3. QUAN TRỌNG: Ép layout bộ nhớ liên tục cho OpenCV
    img = np.ascontiguousarray(img)
    
    return img

def save_loss_plots(history, output_dir):
    """Vẽ và lưu 4 biểu đồ Loss"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    def moving_average(data, window_size=100):
        if len(data) < window_size: return data
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    keys = ['total', 'caption', 'segment', 'keypoint']
    
    for key in keys:
        if key not in history or len(history[key]) == 0: continue
            
        plt.figure(figsize=(10, 6))
        
        # Data gốc (mờ)
        raw_data = history[key]
        plt.plot(raw_data, alpha=0.3, color='gray', label='Raw Loss')
        
        # Data trung bình (đậm)
        ma_data = moving_average(raw_data)
        if len(ma_data) > 0:
            x_axis = range(len(raw_data) - len(ma_data), len(raw_data))
            plt.plot(x_axis, ma_data, color='blue', linewidth=2, label='Mean (100 steps)')
        
        plt.title(f"Loss History: {key.upper()}")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        save_path = os.path.join(output_dir, f"chart_{key}.png")
        plt.savefig(save_path)
        plt.close()

def save_batch_visualization(model_out, batch, tokenizer, output_path):
    """
    Visualize sample đầu tiên trong batch.
    """
    # Lấy sample đầu tiên
    img_tensor = batch['image'][0]
    task_type = batch['task_type'][0]
    
    # Ảnh nền
    vis_img = denormalize(img_tensor)
    H, W, _ = vis_img.shape
    
    # Canvas cho Ground Truth và Prediction
    target_vis = np.zeros_like(vis_img)
    pred_vis = np.zeros_like(vis_img)
    
    # Thông tin header
    text_info = f"Task: {task_type.upper()}"

    # --- LOGIC VẼ TỪNG TASK ---
    if task_type == 'segment':
        # Target Mask
        t_mask = batch['target_mask'][0].cpu().detach().numpy().squeeze()
        target_vis[t_mask > 0] = [0, 255, 0] # Xanh lá
        
        # Pred Mask (Sigmoid > 0.5)
        p_mask = torch.sigmoid(model_out['mask_logits'][0]).cpu().detach().numpy().squeeze()
        pred_vis[p_mask > 0.5] = [0, 0, 255] # Đỏ
        
        # Prompt text
        prompt_text = tokenizer.decode(batch['prompt_ids'][0])
        text_info += f" | {prompt_text[:30]}"

    elif task_type == 'keypoint':
        # Target Heatmap
        t_map = batch['target_mask'][0].cpu().detach().numpy().squeeze()
        # Chuẩn hóa về 0-255 để vẽ màu
        t_map_norm = np.clip(t_map * 255, 0, 255).astype(np.uint8)
        target_vis = cv2.applyColorMap(t_map_norm, cv2.COLORMAP_JET)
        
        # Pred Heatmap
        p_map = torch.sigmoid(model_out['mask_logits'][0]).cpu().detach().numpy().squeeze()
        p_map_norm = np.clip(p_map * 255, 0, 255).astype(np.uint8)
        pred_vis = cv2.applyColorMap(p_map_norm, cv2.COLORMAP_JET)
        
        text_info += " | Heatmap Regression"

    elif task_type == 'caption':
        # Text GT
        target_text = tokenizer.decode(batch['target_ids'][0])
        
        # Text Pred
        logits = model_out['caption_logits'][0]
        pred_ids = torch.argmax(logits, dim=1)
        pred_text = tokenizer.decode(pred_ids)
        
        # Viết chữ lên ảnh
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(target_vis, "GT:", (10, 30), font, 0.6, (0, 255, 0), 2)
        cv2.putText(target_vis, target_text[:40], (10, 60), font, 0.5, (255, 255, 255), 1)
        cv2.putText(target_vis, target_text[40:80], (10, 80), font, 0.5, (255, 255, 255), 1)

        cv2.putText(pred_vis, "Pred:", (10, 30), font, 0.6, (0, 0, 255), 2)
        cv2.putText(pred_vis, pred_text[:40], (10, 60), font, 0.5, (255, 255, 255), 1)
        cv2.putText(pred_vis, pred_text[40:80], (10, 80), font, 0.5, (255, 255, 255), 1)

    # Ghép 3 ảnh ngang
    final_img = np.hstack([vis_img, target_vis, pred_vis])
    
    # Thêm Header trắng ở trên
    header = np.zeros((40, final_img.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, text_info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    final_result = np.vstack([header, final_img])
    
    cv2.imwrite(output_path, final_result)