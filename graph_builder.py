# graph_builder.py
import torch
import numpy as np
from torch_geometric.data import Data

class GraphBuilder:
    def __init__(self, tau=180.0):
        self.tau = float(tau)

    def build_fixed12m_samples(
        self,
        patient_id,
        feats,
        tabular_feats,
        dates,
        labels,
        horizon_days=365,
        tolerance_days=120,
        index_requires_negative=True,
        prefer_later=True,
    ):
        """
        Build rolling-origin samples for fixed 12m timepoint status prediction.
        Input uses visits[0..j], target is closest follow-up to t0+horizon within tolerance.
        Returns (samples, censored_count, total_index_count).
        """
        dates_np = np.array(dates).astype("datetime64[D]")
        order = np.argsort(dates_np, kind="mergesort")
        feats = feats[order]
        tabular_feats = tabular_feats[order]
        dates_np = dates_np[order]
        labels = np.array(labels)[order]

        n = feats.shape[0]
        if n < 2:
            return [], 0, 0

        samples = []
        censored = 0
        total = 0
        tol = np.timedelta64(int(tolerance_days), "D")
        horizon = np.timedelta64(int(horizon_days), "D")

        for j in range(n - 1):
            # 🚨【新增】：如果当前基准节点 (index) 的标签缺失 (SIC=-1)，跳过该节点
            if int(labels[j]) < 0:
                continue
            if index_requires_negative and int(labels[j]) != 0:
                continue
            total += 1
            t0 = dates_np[j]
            target_time = t0 + horizon
            future_dates = dates_np[j + 1 :]
            if len(future_dates) == 0:
                censored += 1
                continue
            deltas = np.abs(future_dates - target_time)
            within = deltas <= tol
            if not within.any():
                censored += 1
                continue
            cand_idx = np.where(within)[0]
            min_delta = deltas[cand_idx].min()
            best = cand_idx[deltas[cand_idx] == min_delta]
            if len(best) > 1:
                if prefer_later:
                    k_rel = best[-1]
                else:
                    k_rel = best[0]
            else:
                k_rel = best[0]
            k = j + 1 + int(k_rel)

            # 🚨【新增】：如果找到的 target 节点标签缺失 (SIC=-1)，按删失处理
            if int(labels[k]) < 0:
                censored += 1
                continue

            prefix_len = j + 1
            x_np = np.concatenate([feats[:prefix_len], tabular_feats[:prefix_len]], axis=1)
            x = torch.tensor(x_np, dtype=torch.float)
            y = torch.tensor(int(labels[k]), dtype=torch.long)

            edge_index = []
            edge_attr = []
            for i in range(prefix_len - 1):
                delta_days = int((dates_np[i + 1] - dates_np[i]).astype(int))
                w = float(np.exp(-delta_days / self.tau))
                edge_index.append([i, i + 1]); edge_attr.append([delta_days, w])
                edge_index.append([i + 1, i]); edge_attr.append([delta_days, w])

            if len(edge_index) == 0:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                edge_attr = torch.zeros((0, 2), dtype=torch.float)
            else:
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_attr, dtype=torch.float)

            data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
            data.patient_id = patient_id
            data.index_date = str(t0)
            data.target_date = str(dates_np[k])
            data.index_pos = int(j)
            data.target_pos = int(k)
            data.prefix_len = prefix_len

            assert data.x.size(0) == prefix_len
            assert data.y.dim() == 0
            samples.append(data)

        return samples, censored, total

    def build_patient_graph(self, patient_id, feats, tabular_feats, dates, labels):
        """
        Build ONE sample per patient:
          input  = prefix visits (1..n-1)
          target = last visit label (n)
        Args:
            feats: (N, D_img)
            tabular_feats: (N, D_tab)  (can be (N,0))
            dates: (N,)
            labels: (N,)
        Returns:
            Data or None (if N < 2)
        """
        # 1) sort by date
        dates_np = np.array(dates).astype("datetime64[D]")
        order = np.argsort(dates_np)
        feats = feats[order]
        tabular_feats = tabular_feats[order]
        dates_np = dates_np[order]
        labels = np.array(labels)[order]

        n = feats.shape[0]
        if n < 2:
            return None

        # 🚨【新增】：如果最后一次visit的目标标签缺失 (SIC=-1)，直接安全丢弃该样本
        target_label = int(labels[-1])
        if target_label < 0:
            return None

        prefix_len = n - 1
        target_date = dates_np[-1]

        # 2) prefix node features only
        x_np = np.concatenate([feats[:prefix_len], tabular_feats[:prefix_len]], axis=1)
        x = torch.tensor(x_np, dtype=torch.float)

        # 3) scalar label = last visit
        y = torch.tensor(target_label, dtype=torch.long)  # scalar

        # 4) edges within prefix only
        edge_index = []
        edge_attr = []
        for i in range(prefix_len - 1):
            delta_days = int((dates_np[i + 1] - dates_np[i]).astype(int))
            w = float(np.exp(-delta_days / self.tau))
            # i -> i+1
            edge_index.append([i, i + 1]); edge_attr.append([delta_days, w])
            # i+1 -> i  (keep bidirectional; no leakage since target node is excluded)
            edge_index.append([i + 1, i]); edge_attr.append([delta_days, w])

        if len(edge_index) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 2), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
        data.patient_id = patient_id
        data.visit_dates = [str(d) for d in dates_np[:prefix_len]]
        data.target_date = str(target_date)
        data.prefix_len = prefix_len

        # --- anti-leakage asserts ---
        assert data.x.size(0) == prefix_len
        assert data.y.dim() == 0  # scalar
        return data