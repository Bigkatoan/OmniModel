# src/model/clip_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.model.backbone import VisionEncoder
from src.model.prompt_encoder import PromptEncoder

class OmniCLIP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        embed_dim = cfg['model']['text']['embed_dim']
        
        # Vision Tower
        vision_cfg = cfg['model']['vision']
        self.vision = VisionEncoder(depths=vision_cfg['depths'], dims=vision_cfg['dims'])
        self.vis_proj = nn.Linear(vision_cfg['dims'][-1], embed_dim)
        
        # Text Tower
        text_cfg = cfg['model']['text']
        self.text = PromptEncoder(
            vocab_size=cfg['data']['vocab_size'],
            embed_dim=embed_dim,
            depth=text_cfg['depth'],
            heads=text_cfg['heads']
        )
        
        # Learnable Temperature
        # Init log(1/0.07) ~ 2.65
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / cfg['train']['temperature']))

    def forward(self, image, text_ids, text_mask):
        # 1. Vision
        img_feats = self.vision(image)[-1]
        img_embed = img_feats.mean(dim=[-2, -1]) 
        img_embed = self.vis_proj(img_embed)
        
        # 2. Text
        text_embed = self.text(text_ids, text_mask)
        
        # 3. Normalize
        img_embed = F.normalize(img_embed, dim=-1)
        text_embed = F.normalize(text_embed, dim=-1)
        
        # [FIX] Clamp logit_scale để tránh Instability (Max ~100)
        # logit_scale quá lớn sẽ làm gradient biến mất
        logit_scale = self.logit_scale.clamp(max=4.605) 
        
        return img_embed, text_embed, logit_scale.exp()