import os
import matplotlib
# QUAN TRỌNG: Chế độ không màn hình cho Server
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import yaml
import glob
from PIL import Image

# Import các module đã build
from src.model.backbone import VisionEncoder
from src.model.prompt_encoder import PromptEncoder
from src.utils.tokenizer import BPE_Tokenizer
import albumentations as A
from albumentations.pytorch import ToTensorV2

def load_components(cfg, device):
    print(">>> Đang tái tạo model và load weights...")
    embed_dim = cfg['model']['text']['embed_dim']
    weights_dir = cfg['system']['output_dir']

    # 1. Vision Encoder (Đôi mắt)
    vision_cfg = cfg['model']['vision']
    vision = VisionEncoder(depths=vision_cfg['depths'], dims=vision_cfg['dims'])
    vision_path = os.path.join(weights_dir, "vision_encoder_best.pth")
    vision.load_state_dict(torch.load(vision_path, map_location=device))
    
    # 2. Vision Projection (Cầu nối)
    vis_proj = nn.Linear(vision_cfg['dims'][-1], embed_dim)
    proj_path = os.path.join(weights_dir, "vision_proj_best.pth")
    vis_proj.load_state_dict(torch.load(proj_path, map_location=device))
    
    # 3. Text Encoder (Bộ não)
    text_cfg = cfg['model']['text']
    text_model = PromptEncoder(vocab_size=cfg['data']['vocab_size'], 
                               embed_dim=embed_dim, 
                               depth=text_cfg['depth'], 
                               heads=text_cfg['heads'])
    text_path = os.path.join(weights_dir, "text_encoder_best.pth")
    text_model.load_state_dict(torch.load(text_path, map_location=device))

    # Chuyển sang device & Eval mode
    vision.to(device).eval()
    vis_proj.to(device).eval()
    text_model.to(device).eval()

    return vision, vis_proj, text_model

def get_transforms(img_size):
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def main():
    # 1. Load Config
    config_path = "configs/text_pretrain.yaml"
    if not os.path.exists(config_path):
        print("❌ Không tìm thấy file config!")
        return

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg['system']['device'] if torch.cuda.is_available() else "cpu")
    print(f">>> Running on: {device}")

    # 2. Tạo folder lưu kết quả
    result_dir = os.path.join(cfg['system']['output_dir'], "retrieval_results")
    os.makedirs(result_dir, exist_ok=True)
    
    # 3. Load Model
    vision, vis_proj, text_model = load_components(cfg, device)
    tokenizer = BPE_Tokenizer(cfg['data']['tokenizer_path'], cfg['data']['max_seq_len'])
    transform = get_transforms(cfg['data']['img_size'])

    # 4. Tạo Gallery (Kho ảnh để tìm kiếm)
    print(">>> Đang khởi tạo Gallery (Kho ảnh)...")
    # Ưu tiên lấy val2017, nếu không có thì lấy train2017
    root_dir = cfg['data']['root_dir']
    img_dir = os.path.join(root_dir, "val2017")
    if not os.path.exists(img_dir) or len(os.listdir(img_dir)) == 0:
        print("Warning: Không thấy val2017, chuyển sang train2017.")
        img_dir = os.path.join(root_dir, cfg['data']['train_img_dir'])

    # Lấy danh sách ảnh jpg
    all_images = glob.glob(os.path.join(img_dir, "*.jpg"))
    if len(all_images) == 0:
        print("❌ Lỗi: Folder ảnh rỗng!")
        return

    # Lấy ngẫu nhiên 50 ảnh để test cho nhanh
    num_samples = min(1000, len(all_images))
    gallery_paths = np.random.choice(all_images, num_samples, replace=False)
    
    gallery_embeds = []
    valid_paths = []

    print(f"--> Đang mã hóa {len(gallery_paths)} ảnh...")
    with torch.no_grad():
        for path in gallery_paths:
            # Đọc ảnh
            img_bgr = cv2.imread(path)
            if img_bgr is None: continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # Preprocess
            img_tensor = transform(image=img_rgb)['image'].unsqueeze(0).to(device)
            
            # Vision Forward
            feats = vision(img_tensor)[-1]      # [1, 768, 7, 7]
            feats = feats.mean(dim=[-2, -1])    # Pooling -> [1, 768]
            embed = vis_proj(feats)             # Project -> [1, 512]
            embed = F.normalize(embed, dim=-1)  # Normalize
            
            gallery_embeds.append(embed)
            valid_paths.append(path)

    if not gallery_embeds:
        print("❌ Không mã hóa được ảnh nào.")
        return

    gallery_tensor = torch.cat(gallery_embeds, dim=0) # [N, 512]
    print(f"✅ Gallery sẵn sàng! Shape: {gallery_tensor.shape}")

    # 5. Vòng lặp Test
    print("\n" + "="*40)
    print(" CHẾ ĐỘ TEST RETRIEVAL (SERVER)")
    print(" Gõ 'exit' để thoát.")
    print(" Kết quả sẽ lưu tại:", result_dir)
    print("="*40)

    query_count = 0
    while True:
        query = input("\n>>> Nhập mô tả (tiếng Anh): ").strip()
        if query.lower() in ['exit', 'quit']: break
        if not query: continue

        query_count += 1
        
        # A. Encode Text
        input_ids, mask = tokenizer.encode(query)
        input_ids = input_ids.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)

        with torch.no_grad():
            txt_embed = text_model(input_ids, mask) # [1, 512]
            txt_embed = F.normalize(txt_embed, dim=-1)

            # B. Tính độ tương đồng (Cosine Similarity)
            # Dot product của 2 vector đã normalize chính là Cosine Similarity
            sims = (txt_embed @ gallery_tensor.t()).squeeze() # [N]
            
            # C. Lấy Top 3
            topk = torch.topk(sims, k=min(3, len(valid_paths)))
        
        # D. Vẽ và Lưu ảnh
        fig = plt.figure(figsize=(12, 5))
        fig.suptitle(f"Query: '{query}'", fontsize=14, fontweight='bold')

        for i, idx in enumerate(topk.indices):
            idx = idx.item()
            score = topk.values[i].item()
            path = valid_paths[idx]
            
            try:
                img_pil = Image.open(path).convert("RGB")
                ax = plt.subplot(1, 3, i+1)
                ax.imshow(img_pil)
                ax.set_title(f"Rank {i+1}\nScore: {score:.4f}", color='green' if i==0 else 'black')
                ax.axis('off')
            except Exception as e:
                print(f"Lỗi load ảnh kết quả: {e}")

        # Lưu file
        safe_name = "".join([c if c.isalnum() else "_" for c in query])[:30]
        save_path = os.path.join(result_dir, f"test_{query_count}_{safe_name}.jpg")
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig) # Giải phóng RAM

        print(f"--> Đã lưu kết quả: {save_path}")
        print(f"--> Scores: {[round(x.item(), 3) for x in topk.values]}")

if __name__ == "__main__":
    main()