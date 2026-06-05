from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.function import ratio2weight

class NLL_class_weight_loss(nn.Module):
    def __init__(self, sample_weight=None, size_average=True, attr_idx=None):
        super(NLL_class_weight_loss, self).__init__()

        self.sample_weight = sample_weight
        self.size_average = size_average
        self.attr_idx = attr_idx

    def forward(self, logits, targets, gt_label):
        # 1. Crear una máscara: 1 donde el target es válido, 0 donde es -1 (ignorar)
        valid_mask = (targets != -1).float()
        
        # 2. Copia segura de los targets: cambiamos los -1 por 0 para que 'gather' no explote en CUDA
        safe_targets = targets.clone()
        safe_targets[targets == -1] = 0

        # 3. Hacer el gather con los índices seguros
        # (Si había un -1, ahora leerá el índice 0, pero lo anularemos en el paso 5)
        selected_log_probs = torch.gather(logits, dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)
        
        targets_mask = torch.where(gt_label.detach().cpu() > 0.5, torch.ones(1), torch.zeros(1))
        
        if self.sample_weight is not None:
            weights = ratio2weight(targets_mask, self.sample_weight)       
            
        # 4. Calcular la pérdida ponderada original
        weighted_loss = -selected_log_probs * weights.contiguous().view(-1).cuda()
        
        # 5. ¡APLICAR LA MÁSCARA! Multiplicamos por 0 la pérdida de los atributos desconocidos (-1)
        weighted_loss = weighted_loss * valid_mask.view(-1).cuda()
        
        # 6. Calcular el promedio real (dividiendo solo entre los elementos válidos, no entre todos)
        # Evitamos dividir por 0 usando clamp
        loss = torch.sum(weighted_loss) / torch.clamp(valid_mask.sum(), min=1e-5)
        
        return loss