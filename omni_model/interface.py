import os
import cv2
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Import các module nội bộ
from src.model.backbone import VisionEncoder
from src.model.prompt_encoder import PromptEncoder
from src.utils.tokenizer import BPE_Tokenizer
import albumentations as A
from albumentations.pytorch import ToTensorV2

class OmniModel:
    def __init__(self, model_path=None, device=None):
        """
        Khởi tạo OmniModel SDK.
        
        Args:
            model_path (str): Đường dẫn đến thư mục 'omni_model'. 
                              Mặc định sẽ lấy thư mục chứa file này.
            device (str): 'cuda' hoặc 'cpu'.
        """
        # Tự động lấy đường dẫn của thư mục chứa file interface.py này
        if model_path is None:
            self.root = os.path.dirname(os.path.abspath(__file__))
        else:
            self.root = model_path

        # Setup Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f">>> OmniModel initialized on: {self.device}")

        # Load Config
        config_path = os.path.join(self.root, "config/model_config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Missing config: {config_path}")
            
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        # Load Weights
        self._load_weights()
        self._setup_transforms()

    def _load_weights(self):
        embed_dim = self.cfg['model']['text']['embed_dim']
        weights_dir = os.path.join(self.root, "weights")

        # 1. Vision Encoder
        vis_cfg = self.cfg['model']['vision']
        self.vision = VisionEncoder(depths=vis_cfg['depths'], dims=vis_cfg['dims'])
        
        path = os.path.join(weights_dir, "vision_encoder.pth")
        if os.path.exists(path):
            self.vision.load_state_dict(torch.load(path, map_location=self.device))
        else:
            print(f"[WARN] Missing {path}, using random weights.")

        # 2. Projection
        self.vis_proj = nn.Linear(vis_cfg['dims'][-1], embed_dim)
        path = os.path.join(weights_dir, "vision_proj.pth")
        if os.path.exists(path):
            self.vis_proj.load_state_dict(torch.load(path, map_location=self.device))

        # 3. Text Encoder
        txt_cfg = self.cfg['model']['text']
        self.text = PromptEncoder(vocab_size=self.cfg['data']['vocab_size'], 
                                  embed_dim=embed_dim, 
                                  depth=txt_cfg['depth'], 
                                  heads=txt_cfg['heads'])
        path = os.path.join(weights_dir, "text_encoder.pth")
        if os.path.exists(path):
            self.text.load_state_dict(torch.load(path, map_location=self.device))

        # 4. Tokenizer
        self.tokenizer = BPE_Tokenizer(os.path.join(self.root, "config/tokenizer.json"), 
                                       self.cfg['data']['max_seq_len'])
        
        # Eval mode
        self.vision.to(self.device).eval()
        self.vis_proj.to(self.device).eval()
        self.text.to(self.device).eval()

    def _setup_transforms(self):
        size = self.cfg['data']['img_size']
        self.transform = A.Compose([
            A.Resize(height=size, width=size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def encode_image(self, image_input):
        """Encode image (Path or Array) -> Vector [1, 512]"""
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
        """Encode text -> Vector [1, 512]"""
        input_ids, mask = self.tokenizer.encode(text)
        input_ids = input_ids.unsqueeze(0).to(self.device)
        mask = mask.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embed = self.text(input_ids, mask)
            embed = F.normalize(embed, dim=-1)
        return embed

    def get_backbone(self):
        return self.vision