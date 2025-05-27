import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class DistillationLoss(nn.Module):
    def __init__(self, temperature=3.0):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature

    def forward(self, teacher_logits,student_logits):
        student_logits_temp = student_logits / self.temperature
        teacher_logits_temp = teacher_logits / self.temperature

        soft_loss = F.kl_div(
            F.log_softmax(teacher_logits_temp, dim=1),
            F.softmax(student_logits_temp, dim=1),
            reduction='batchmean'
        ) 

        return soft_loss*(self.temperature ** 2)  
