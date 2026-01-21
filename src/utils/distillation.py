# src/utils/distillation.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.T = temperature
        # KLDivLoss yêu cầu input là log-probabilities
        self.kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)

    def forward(self, student_logits, teacher_logits):
        """
        student_logits: [Batch, Batch] - Ma trận tương đồng của model bạn
        teacher_logits: [Batch, Batch] - Ma trận tương đồng của CLIP xịn
        """
        # 1. Softmax với Temperature (làm mềm phân phối để dễ học hơn)
        # Student cần Log-Softmax
        s_log_probs = F.log_softmax(student_logits / self.T, dim=-1)
        
        # Teacher cần Log-Softmax (vì log_target=True giúp ổn định số học hơn)
        t_log_probs = F.log_softmax(teacher_logits / self.T, dim=-1)
        
        return self.kl_loss(s_log_probs, t_log_probs)