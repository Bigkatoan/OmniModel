# src/model/omninet.py

import torch
import torch.nn as nn
from src.model.backbone import VisionEncoder
from src.model.prompt_encoder import PromptEncoder
from src.model.dynamic_head import DynamicMaskHead

class OmniNet60M(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 1. Vision Backbone (ConvNeXt-Tiny style)
        vision_cfg = cfg['model']['vision']
        self.vision_encoder = VisionEncoder(
            depths=vision_cfg['depths'], 
            dims=vision_cfg['dims']
        )
        
        # 2. Prompt Encoder (Transformer)
        text_cfg = cfg['model']['text']
        self.prompt_encoder = PromptEncoder(
            vocab_size=cfg['data']['vocab_size'],
            embed_dim=text_cfg['embed_dim'],
            depth=text_cfg['depth'],
            heads=text_cfg['heads']
        )
        
        # 3. Dynamic Head (Cho Segmentation & Keypoint)
        # Input features từ Vision Backbone F1 (96 channels)
        self.dynamic_head = DynamicMaskHead(
            in_channels=vision_cfg['dims'][0], # 96
            text_dim=text_cfg['embed_dim'],    # 512
            mask_channels=cfg['model']['heads']['mask_channels']
        )
        
        # 4. Caption Decoder (Transformer Decoder)
        # Dùng để sinh text từ ảnh
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=text_cfg['embed_dim'], 
            nhead=4, 
            dim_feedforward=1024,
            batch_first=True
        )
        self.caption_decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)
        
        # Project Vision Features (768) -> Text Dim (512) để đưa vào Decoder
        self.vis_proj = nn.Linear(vision_cfg['dims'][-1], text_cfg['embed_dim'])
        
        # Output layer cho caption (Vocab prediction)
        self.caption_out = nn.Linear(text_cfg['embed_dim'], cfg['data']['vocab_size'])
        
        # Positional Encoding cho Decoder (Học được)
        self.tgt_pos_embed = nn.Parameter(torch.zeros(1, cfg['data']['max_seq_len'], text_cfg['embed_dim']))

    def forward(self, image, prompt_ids, prompt_mask, task_type, target_ids=None):
        """
        Input:
            image: [B, 3, H, W]
            prompt_ids: [B, Seq_Len] (Prompt lệnh: "Find the cat")
            task_type: list of str ["caption", "segment", ...]
        """
        # A. Vision Path
        # feats = [F1(1/4), F2(1/8), F3(1/16), F4(1/32)]
        img_feats = self.vision_encoder(image)
        
        # B. Prompt Path
        # Lấy vector đại diện câu lệnh (Pooling)
        prompt_embed = self.prompt_encoder(prompt_ids, prompt_mask) # [B, 512]
        
        outputs = {}

        # --- LOGIC XỬ LÝ THEO BATCH ---
        # Lưu ý: Code này giả định cả batch làm cùng 1 task hoặc dataloader mix task.
        # Để đơn giản cho training loop, ta xử lý dựa trên task_type đầu tiên của batch
        # (Trong thực tế nên group batch theo task).
        
        current_task = task_type[0] 

        if current_task == "segment" or current_task == "keypoint":
            # --- TASK: SEGMENTATION / KEYPOINT ---
            # Dùng F1 (High Res) và Prompt Embedding để sinh Mask
            mask_logits = self.dynamic_head(img_feats[0], prompt_embed)
            outputs["mask_logits"] = mask_logits
            
        elif current_task == "caption":
            # --- TASK: CAPTIONING ---
            # Dùng F4 (Semantic) làm Memory cho Decoder
            # F4: [B, 768, H/32, W/32] -> Flatten -> [B, Seq_Vis, 768]
            B, C, H, W = img_feats[-1].shape
            memory = img_feats[-1].flatten(2).permute(0, 2, 1) # [B, HW, C]
            memory = self.vis_proj(memory) # [B, HW, 512]
            
            if target_ids is not None:
                # Training Mode: Teacher Forcing
                # Target: [BOS] a cat [EOS] -> Input: [BOS] a cat
                tgt_input = target_ids[:, :-1] # Bỏ token cuối
                
                # Embed target text
                tgt_emb = self.prompt_encoder.embedding(tgt_input)
                seq_len = tgt_emb.shape[1]
                tgt_emb = tgt_emb + self.tgt_pos_embed[:, :seq_len, :]
                
                # Causal Mask (Che tương lai)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(image.device)
                
                # Decode
                out_dec = self.caption_decoder(tgt_emb, memory, tgt_mask=tgt_mask)
                logits = self.caption_out(out_dec) # [B, Seq-1, Vocab]
                
                outputs["caption_logits"] = logits
            
        return outputs