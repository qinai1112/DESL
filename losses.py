import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiTaskLoss(nn.Module):
    def __init__(self, alpha_cls=1.0, class_weights=None, label_smoothing=0.05):
        super().__init__()
        self.alpha_cls = alpha_cls
        self.class_weights = class_weights
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits, y, sample_weight=None):
        # logits: [B,2], y: [B]
        y = y.view(-1).long()
        ce = F.cross_entropy(
            logits, y,
            reduction="none",
            weight=self.class_weights,
            label_smoothing=self.label_smoothing
        )
        if sample_weight is not None:
            w = sample_weight.view(-1).to(ce.device).float()
            loss = (ce * w).sum() / (w.sum() + 1e-12)
        else:
            loss = ce.mean()
        return {"cls": loss, "total": loss}


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, label_smoothing=0.05):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits, y, sample_weight=None):
        y = y.view(-1).long()
        ce = F.cross_entropy(
            logits, y,
            reduction="none",
            weight=self.alpha,
            label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce)
        fl = ((1 - pt) ** self.gamma) * ce

        if sample_weight is not None:
            w = sample_weight.view(-1).to(fl.device).float()
            loss = (fl * w).sum() / (w.sum() + 1e-12)
        else:
            loss = fl.mean()
        return {"cls": loss, "total": loss}


class ClassBalancedLoss(nn.Module):
    def __init__(self, beta=0.9999, label_smoothing=0.05):
        super().__init__()
        self.beta = float(beta)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits, y, sample_weight=None):
        y = y.view(-1).long()
        num_classes = logits.size(1)

        labels_np = y.detach().cpu().numpy()
        class_count = np.array([(labels_np == i).sum() for i in range(num_classes)], dtype=np.float32)

        effective_num = 1.0 - np.power(self.beta, class_count)
        weights = (1.0 - self.beta) / (effective_num + 1e-12)
        weights = weights / weights.sum() * num_classes
        weights = torch.tensor(weights, dtype=torch.float, device=logits.device)

        ce = F.cross_entropy(
            logits, y,
            reduction="none",
            weight=weights,
            label_smoothing=self.label_smoothing
        )

        if sample_weight is not None:
            w = sample_weight.view(-1).to(ce.device).float()
            loss = (ce * w).sum() / (w.sum() + 1e-12)
        else:
            loss = ce.mean()

        return {"cls": loss, "total": loss}
