# train_clip.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import os
from transformers import CLIPProcessor, CLIPModel # <--- Import Teacher

from src.data.caption_dataset import CaptionDataset
from src.model.clip_model import OmniCLIP
from src.utils.plotter import LogPlotter
from src.utils.distillation import DistillationLoss # <--- Import Loss mới

# ... (Hàm contrastive_loss giữ nguyên) ...
def contrastive_loss(img_emb, text_emb, logit_scale):
    logits_per_image = logit_scale * img_emb @ text_emb.t()
    logits_per_text = logits_per_image.t()
    
    batch_size = img_emb.shape[0]
    labels = torch.arange(batch_size, device=img_emb.device)
    
    loss_i = nn.CrossEntropyLoss()(logits_per_image, labels)
    loss_t = nn.CrossEntropyLoss()(logits_per_text, labels)
    
    return (loss_i + loss_t) / 2, logits_per_image # Trả thêm logits để distill

def main():
    with open("configs/text_pretrain.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device(cfg['system']['device'])
    output_dir = cfg['system']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    plotter = LogPlotter(output_dir)
    
    # --- 1. SETUP TEACHER (OPENAI CLIP) ---
    print(">>> Loading Teacher Model (OpenAI CLIP)...")
    teacher_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    teacher_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    teacher_model.eval() # Đóng băng Teacher
    for param in teacher_model.parameters():
        param.requires_grad = False
        
    distill_criterion = DistillationLoss(temperature=2.0)
    DISTILL_WEIGHT = 5.0 # Trọng số loss phụ trợ (Teacher quan trọng hơn)

    # --- 2. SETUP STUDENT (YOUR MODEL) ---
    dataset = CaptionDataset(cfg)
    dataloader = DataLoader(dataset, batch_size=cfg['data']['batch_size'], 
                            shuffle=True, num_workers=4, drop_last=True)
    
    model = OmniCLIP(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['train']['epochs'])
    
    ACCUM_STEPS = 4
    print(f">>> START DISTILLATION TRAINING...")
    
    best_loss = float('inf')
    
    for epoch in range(cfg['train']['epochs']):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        for i, batch in enumerate(pbar):
            # A. Prepare Student Inputs
            img = batch['image'].to(device)
            txt = batch['input_ids'].to(device)
            mask = batch['mask'].to(device)
            raw_texts = batch['raw_text'] # List of strings
            
            # B. Forward Student
            s_img_emb, s_txt_emb, s_scale = model(img, txt, mask)
            # Loss gốc (Contrastive)
            loss_contrast, s_logits = contrastive_loss(s_img_emb, s_txt_emb, s_scale)
            
            # C. Forward Teacher (No Grad)
            with torch.no_grad():
                # Teacher tự xử lý text input
                t_inputs = teacher_processor(text=raw_texts, return_tensors="pt", padding=True, truncation=True).to(device)
                
                # Input ảnh cho Teacher:
                # CLIP gốc dùng normalization khác, nhưng dùng chung img tensor cũng tạm ổn
                # Hoặc tốt nhất là resize 224 đúng chuẩn CLIP (dataset ta đã làm 224 rồi)
                
                # Teacher Forward
                # Lưu ý: Teacher cần pixel_values (ảnh) và input_ids (text)
                # Nhưng ở đây ta dùng ảnh từ dataloader (đã normalize ImageNet).
                # CLIP normalize hơi khác 1 chút nhưng chấp nhận được để distill.
                t_outputs = teacher_model(pixel_values=img, input_ids=t_inputs['input_ids'], attention_mask=t_inputs['attention_mask'])
                
                t_logits = t_outputs.logits_per_image # [Batch, Batch]
            
            # D. Distillation Loss
            # So sánh Logits của Student và Logits của Teacher
            loss_distill = distill_criterion(s_logits, t_logits)
            
            # E. Total Loss
            total_loss = loss_contrast + (DISTILL_WEIGHT * loss_distill)
            
            # F. Backward & Optimize
            total_loss = total_loss / ACCUM_STEPS
            total_loss.backward()
            
            if (i + 1) % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            # Logging
            current_loss = total_loss.item() * ACCUM_STEPS
            epoch_loss += current_loss
            pbar.set_postfix(loss=current_loss, dist=loss_distill.item())
        
        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
        plotter.update(epoch + 1, avg_loss, scheduler.get_last_lr()[0])
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.vision.state_dict(), os.path.join(output_dir, "vision_encoder_best.pth"))
            torch.save(model.text.state_dict(), os.path.join(output_dir, "text_encoder_best.pth"))
            torch.save(model.vis_proj.state_dict(), os.path.join(output_dir, "vision_proj_best.pth"))

if __name__ == "__main__":
    main()