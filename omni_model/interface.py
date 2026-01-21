import os
import cv2
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from huggingface_hub import hf_hub_download  # <--- THƯ VIỆN QUAN TRỌNG

from .src.model.backbone import VisionEncoder
from .src.model.prompt_encoder import PromptEncoder
from .src.utils.tokenizer import BPE_Tokenizer
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Đặt ID Repo trên Hugging Face của bạn
HF_REPO_ID = "Bigkatoan/OmniModel" 

class OmniModel:
    def __init__(self, device=None):
        # 1. Setup Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f">>> OmniModel initializing on {self.device}...")

        # 2. Load Config (Đi kèm trong gói pip)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config/model_config.yaml")
        
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        # 3. Load Weights (Tự động tải từ HuggingFace)
        self._load_weights_from_hf()
        self._setup_transforms()
        print(">>> Model loaded successfully!")

    def _load_weights_from_hf(self):
        embed_dim = self.cfg['model']['text']['embed_dim']
        
        # Hàm tiện ích để download hoặc lấy từ cache
        def get_weight_path(filename):
            print(f"--> Checking/Downloading {filename} from HuggingFace...")
            return hf_hub_download(repo_id=HF_REPO_ID, filename=filename)

        # --- Vision Encoder ---
        vis_cfg = self.cfg['model']['vision']
        self.vision = VisionEncoder(depths=vis_cfg['depths'], dims=vis_cfg['dims'])
        vis_path = get_weight_path("vision_encoder.pth")
        self.vision.load_state_dict(torch.load(vis_path, map_location=self.device))

        # --- Projection ---
        self.vis_proj = nn.Linear(vis_cfg['dims'][-1], embed_dim)
        proj_path = get_weight_path("vision_proj.pth")
        self.vis_proj.load_state_dict(torch.load(proj_path, map_location=self.device))

        # --- Text Encoder ---
        txt_cfg = self.cfg['model']['text']
        self.text = PromptEncoder(vocab_size=self.cfg['data']['vocab_size'], 
                                  embed_dim=embed_dim, 
                                  depth=txt_cfg['depth'], 
                                  heads=txt_cfg['heads'])
        txt_path = get_weight_path("text_encoder.pth")
        self.text.load_state_dict(torch.load(txt_path, map_location=self.device))

        # --- Tokenizer (Load từ file config đi kèm package) ---
        current_dir = os.path.dirname(os.path.abspath(__file__))
        tok_path = os.path.join(current_dir, "config/tokenizer.json")
        self.tokenizer = BPE_Tokenizer(tok_path, self.cfg['data']['max_seq_len'])

        # Eval Mode
        self.vision.to(self.device).eval()
        self.vis_proj.to(self.device).eval()
        self.text.to(self.device).eval()

    # ... (Giữ nguyên các hàm encode_image, encode_text, setup_transforms) ...
    def _setup_transforms(self):
        size = self.cfg['data']['img_size']
        self.transform = A.Compose([
            A.Resize(height=size, width=size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def encode_image(self, image_input):
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_input
        img_tensor = self.transform(image=image)['image'].unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.vision(img_tensor)[-1].mean(dim=[-2, -1])
            embed = self.vis_proj(feat)
            embed = F.normalize(embed, dim=-1)
        return embed

    def encode_text(self, text):
        input_ids, mask = self.tokenizer.encode(text)
        input_ids = input_ids.unsqueeze(0).to(self.device)
        mask = mask.unsqueeze(0).to(self.device)
        with torch.no_grad():
            embed = self.text(input_ids, mask)
            embed = F.normalize(embed, dim=-1)
        return embed
    
    def get_backbone(self):
        return self.vision