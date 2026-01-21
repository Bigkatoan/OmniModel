# src/model/prompt_encoder.py

import torch
import torch.nn as nn

class PromptEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, depth=6, heads=8):
        super().__init__()
        # 1. Word Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # 2. Positional Encoding (Learnable)
        # Giả định max_len=128 (dư sức cho lệnh robot)
        self.pos_embed = nn.Parameter(torch.zeros(1, 128, embed_dim))
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=heads, 
            dim_feedforward=embed_dim*4,
            batch_first=True,
            norm_first=True
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # 4. Final Norm
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        # x: [Batch, Seq_Len]
        # mask: [Batch, Seq_Len] (1=not pad, 0=pad)
        
        B, L = x.shape
        x = self.embedding(x)
        
        # Add Positional Encoding
        x = x + self.pos_embed[:, :L, :]
        
        # Tạo mask cho Transformer (PyTorch yêu cầu True=Ignored/Pad)
        if mask is not None:
            # Đảo ngược mask: 0 -> True (Ignore), 1 -> False (Keep)
            src_key_padding_mask = (mask == 0)
        else:
            src_key_padding_mask = None

        x = self.blocks(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)
        
        # Global Pooling: Lấy trung bình cộng của các token (Mean Pooling)
        # Để lấy ra 1 vector duy nhất đại diện cho câu lệnh
        if mask is not None:
            # Chỉ tính trung bình các token thật (bỏ padding)
            mask_expanded = mask.unsqueeze(-1).expand(x.size()).float()
            sum_embeddings = torch.sum(x * mask_expanded, 1)
            sum_mask = mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            return sum_embeddings / sum_mask
        else:
            return x.mean(dim=1) # [Batch, Embed_Dim]