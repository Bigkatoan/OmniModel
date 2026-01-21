import os
import cv2
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from huggingface_hub import hf_hub_download

# Lưu ý dấu chấm (.) ở trước src -> Relative import cho package
from .src.model.backbone import VisionEncoder
from .src.model.prompt_encoder import PromptEncoder
from .src.utils.tokenizer import BPE_Tokenizer
import albumentations as A
from albumentations.pytorch import ToTensorV2

HF_REPO_ID = "bigkatoan/OmniModel" # Thay bằng ID thật của bạn sau này

class OmniModel:
    def __init__(self, device=None):
        # 1. Setup Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f">>> OmniModel initializing on {self.device}...")

        # 2. Load Config
        self.root_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(self.root_dir, "config/model_config.yaml")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Missing config at {config_path}")

        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        # 3. Load Weights (Hybrid Mode)
        self._load_weights()
        self._setup_transforms()
        print(">>> OmniModel is ready!")

    def _get_file_path(self, filename):
        """
        Logic thông minh:
        1. Tìm trong folder weights/ nội bộ trước (cho dev/test local).
        2. Nếu không thấy, tự động tải từ HuggingFace Hub (cho user pip).
        """
        # Check Local
        local_path = os.path.join(self.root_dir, "weights", filename)
        if os.path.exists(local_path):
            print(f"--> Found local weight: {filename}")
            return local_path
        
        # Check HuggingFace
        print(f"--> Local not found. Downloading {filename} from HuggingFace...")
        try:
            return hf_hub_download(repo_id=HF_REPO_ID, filename=filename)
        except Exception as e:
            raise RuntimeError(f"Could not download {filename} from HF. Error: {e}")

    def _load_weights(self):
        embed_dim = self.cfg['model']['text']['embed_dim']
        
        # --- Vision ---
        vis_cfg = self.cfg['model']['vision']
        self.vision = VisionEncoder(depths=vis_cfg['depths'], dims=vis_cfg['dims'])
        path = self._get_file_path("vision_encoder.pth")
        self.vision.load_state_dict(torch.load(path, map_location=self.device))

        # --- Projection ---
        self.vis_proj = nn.Linear(vis_cfg['dims'][-1], embed_dim)
        path = self._get_file_path("vision_proj.pth")
        self.vis_proj.load_state_dict(torch.load(path, map_location=self.device))

        # --- Text ---
        txt_cfg = self.cfg['model']['text']
        self.text = PromptEncoder(vocab_size=self.cfg['data']['vocab_size'], 
                                  embed_dim=embed_dim, 
                                  depth=txt_cfg['depth'], 
                                  heads=txt_cfg['heads'])
        path = self._get_file_path("text_encoder.pth")
        self.text.load_state_dict(torch.load(path, map_location=self.device))

        # --- Tokenizer ---
        tok_path = os.path.join(self.root_dir, "config/tokenizer.json")
        self.tokenizer = BPE_Tokenizer(tok_path, self.cfg['data']['max_seq_len'])

        # Eval Mode
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