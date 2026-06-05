from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.function import ratio2weight


class CEL_Sigmoid(nn.Module):
    def __init__(self, sample_weight=None, size_average=True, attr_idx=None):
        super(CEL_Sigmoid, self).__init__()

        self.sample_weight = sample_weight
        self.size_average = size_average
        self.attr_idx = attr_idx

    def forward(self, logits, targets):
        batch_size = logits.shape[0]

        # 1. Crear una máscara para ignorar los -1
        # valid_mask será 1 donde hay datos (0 o 1) y 0 donde hay -1
        valid_mask = (targets != -1).float()

        # 2. Calcular la pérdida base
        # Usamos reduction='none' para filtrar después
        loss = F.binary_cross_entropy_with_logits(logits, targets.clamp(min=0), reduction='none')

        # 3. Aplicar la máscara de validez (aquí "borramos" la pérdida de los -1)
        loss = loss * valid_mask

        # 4. Manejo de pesos (tu lógica original adaptada)
        # Solo calculamos pesos sobre lo que no es -1 y es positivo (>0.5)
        targets_mask = torch.where(targets > 0.5, torch.ones_like(targets), torch.zeros_like(targets))
        
        if self.sample_weight is not None:
            if self.attr_idx is not None and targets_mask.shape[1] != self.sample_weight.shape[0]:
                weight = ratio2weight(targets_mask[:, self.attr_idx], self.sample_weight)
                loss = loss[:, self.attr_idx]
            else:
                weight = ratio2weight(targets_mask, self.sample_weight)
            
            # Aplicamos el peso (asegurando que el peso también respete el dispositivo)
            loss = (loss * weight.to(logits.device))

        # 5. Normalización final
        # Es mejor normalizar por el número de elementos válidos, no solo por el batch_size
        if self.size_average:
            # Sumamos toda la pérdida y dividimos por los elementos que NO eran -1
            return loss.sum() / valid_mask.sum().clamp(min=1)
        else:
            return loss.sum()


class FocalLoss(nn.Module):
    def __init__(self, sample_weight=None, size_average=True, attr_idx=None, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()

        self.sample_weight = sample_weight
        self.size_average = size_average
        self.attr_idx = attr_idx
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        batch_size = logits.shape[0]

        # Compute the binary cross entropy loss with logits
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # Apply sigmoid to get the predicted probabilities
        probs = torch.sigmoid(logits)
        targets_mask = torch.where(targets.detach().cpu() > 0.5, torch.ones(1), torch.zeros(1))

        # Compute the focal loss components
        p_t = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - p_t) ** self.gamma

        # Compute the focal loss
        focal_loss = self.alpha * focal_weight * bce_loss

        if self.sample_weight is not None:
            if self.attr_idx is not None and targets_mask.shape[1] != self.sample_weight.shape[0]:
                weight = ratio2weight(targets_mask[:, self.attr_idx], self.sample_weight)
                focal_loss = focal_loss[:, self.attr_idx]
            else:
                weight = ratio2weight(targets_mask, self.sample_weight)
            focal_loss = focal_loss * weight.cuda()

        loss = focal_loss.sum() / batch_size if self.size_average else focal_loss.sum()

        return loss