# labels.py
import torch
import math
import numpy as np

class LabelBuilder:
    def __init__(self, alpha=4.0, tau=180.0):
        self.alpha = alpha
        self.tau = tau

    def build(self, patient_graphs, device="cpu"):
        B = len(patient_graphs)
        max_visits = max(g.x.shape[0] for g in patient_graphs)

        y_cls = torch.full((B, max_visits), -1, dtype=torch.long)
        mask = torch.zeros((B, max_visits), dtype=torch.bool)
        sample_weight = torch.ones((B, max_visits), dtype=torch.float32)

        for b, g in enumerate(patient_graphs):
            if hasattr(g, "graph_label"):
                y_cls[b, 0] = int(g.graph_label)
                mask[b, 0] = 1
                sample_weight[b, 0] = 1.0
                continue
            n = g.x.shape[0]
            mask[b, :n] = 1
            y_cls[b, :n] = g.y  # visit-level SIC 标签

            # onset = 第一次 SIC=1 的日期，如果没有就用最后一次
            onset_date = None
            for i, sic in enumerate(g.y.tolist()):
                if sic == 1:
                    onset_date = np.datetime64(g.visit_dates[i])
                    break
            if onset_date is None:
                onset_date = np.datetime64(g.visit_dates[-1])

            # 根据到 onset 的时间差生成 sample_weight
            for i, d in enumerate(g.visit_dates):
                delta = abs((np.datetime64(d) - onset_date).astype(int))
                w = 1.0 + self.alpha * math.exp(-delta / self.tau)
                sample_weight[b, i] = min(w, 1.0 + self.alpha)

        return {
            "y_cls": y_cls.to(device),
            "mask": mask.to(device),
            "sample_weight": sample_weight.to(device),
        }
