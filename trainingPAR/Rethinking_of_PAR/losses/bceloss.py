import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import LOSSES
from tools.function import ratio2weight
import numpy as np

@LOSSES.register("bceloss")
class BCELoss(nn.Module):

    def __init__(self, sample_weight=None, size_sum=True, scale=None, tb_writer=None):
        super(BCELoss, self).__init__()

        self.sample_weight = sample_weight
        self.size_sum = size_sum
        self.hyper = 0.8
        self.smoothing = None

    def forward(self, logits, targets):
        logits = logits[0]
    
        # then apply the mask to the loss
        if self.smoothing is not None:
            targets = (1 - self.smoothing) * targets + self.smoothing * (1 - targets)

        targets_cpu_np = targets.detach().cpu().numpy()
        pos_weight = np.ma.MaskedArray(targets_cpu_np, mask=(np.ones_like(targets_cpu_np)*(targets_cpu_np[:,:]!=-1))).mask.astype(int)
        pos_weight_tensor = torch.from_numpy(pos_weight)
        pos_weight_tensor = pos_weight_tensor.to(device='cuda')
        """
        print("logits")
        print(logits)
        print("targets")
        print(targets)
        print("pos_weight_tensor")
        """
        #print(pos_weight_tensor)
        loss_m = F.binary_cross_entropy_with_logits(logits, targets, reduction='none', pos_weight=pos_weight_tensor)
        #loss_m = F.binary_cross_entropy_with_logits(logits, targets, reduction='none', weight=pos_weight_tensor)
        #loss_m = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        """
        print("loss_m")
        print(loss_m)
        """

        targets_mask = torch.where(targets.detach().cpu() > 0.5, torch.ones(1), torch.zeros(1))
        if self.sample_weight is not None:
            sample_weight = ratio2weight(targets_mask, self.sample_weight)
            #print("sample_weight")
            #print(sample_weight)
            loss_m = (loss_m * sample_weight.cuda())

        # losses = loss_m.sum(1).mean() if self.size_sum else loss_m.mean()
        loss = loss_m.sum(1).mean() if self.size_sum else loss_m.sum()

        return [loss], [loss_m]
    
