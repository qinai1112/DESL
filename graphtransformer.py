# graphtransformer.py
import math
from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv
from tabular_extractor import TabularFeatureExtractor

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B,T,D)
        T = x.size(1)
        return x + self.pe[:, :T, :]


def _make_mlp(input_dim: int, output_dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.3) -> nn.Sequential:
    hidden_dim = hidden_dim or output_dim
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
        nn.ReLU(inplace=True),
    )


class ConcatenationFusion(nn.Module):
    """Aligned image/tabular projections followed by concatenation and MLP_F."""
    def __init__(self, image_dim, tab_dim, fusion_dim=128, output_dim=128, dropout=0.3):
        super().__init__()
        self.img_align_mlp = _make_mlp(image_dim, fusion_dim, dropout=dropout)
        self.tab_align_mlp = _make_mlp(tab_dim, fusion_dim, dropout=dropout)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * fusion_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, img_feat, tab_feat):
        img_z = self.img_align_mlp(img_feat)
        tab_z = self.tab_align_mlp(tab_feat)
        return self.fusion_mlp(torch.cat([img_z, tab_z], dim=-1))


class ReliabilityAwareCrossModalFusion(nn.Module):
    """
    Reliability-aware cross-modal alignment fusion.

    Inputs:
        img_feat: (..., image_dim)
        tab_feat: (..., tab_dim), expected as [tabular values, raw missing mask]
        missing_mask: optional (..., missing_mask_dim). If None, it is read from
            the last missing_mask_dim columns of tab_feat.

    Output:
        h: (..., output_dim), the visit/node embedding consumed by the graph encoder.
    """
    def __init__(
        self,
        image_dim: int,
        tab_dim: int,
        fusion_dim: int = 128,
        output_dim: int = 128,
        missing_mask_dim: Optional[int] = None,
        dropout: float = 0.3,
        use_multiplicative_interaction: bool = True,
    ):
        super().__init__()
        if tab_dim <= 0:
            raise ValueError("tab_dim must be positive for reliability-aware fusion.")
        self.image_dim = int(image_dim)
        self.tab_dim = int(tab_dim)
        self.fusion_dim = int(fusion_dim)
        self.output_dim = int(output_dim)
        self.missing_mask_dim = int(missing_mask_dim or max(1, tab_dim // 2))
        self.use_multiplicative_interaction = bool(use_multiplicative_interaction)

        self.img_align_mlp = _make_mlp(image_dim, fusion_dim, dropout=dropout)
        self.tab_align_mlp = _make_mlp(tab_dim, fusion_dim, dropout=dropout)
        self.gate_layer = nn.Linear(2 * fusion_dim + self.missing_mask_dim, 2)
        fusion_input_dim = (3 if self.use_multiplicative_interaction else 2) * fusion_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def _get_missing_mask(self, tab_feat: torch.Tensor, missing_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if missing_mask is None:
            if tab_feat.size(-1) < self.missing_mask_dim:
                raise ValueError(
                    f"tab_feat last dim ({tab_feat.size(-1)}) is smaller than missing_mask_dim ({self.missing_mask_dim})."
                )
            missing_mask = tab_feat[..., -self.missing_mask_dim:]
        if missing_mask.size(-1) != self.missing_mask_dim:
            raise ValueError(
                f"missing_mask last dim must be {self.missing_mask_dim}, got {missing_mask.size(-1)}."
            )
        return missing_mask.to(dtype=tab_feat.dtype, device=tab_feat.device)

    def forward(
        self,
        img_feat: torch.Tensor,
        tab_feat: torch.Tensor,
        missing_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        img_z = self.img_align_mlp(img_feat)
        tab_z = self.tab_align_mlp(tab_feat)
        missing_mask = self._get_missing_mask(tab_feat, missing_mask)

        gate_input = torch.cat([img_z, tab_z, missing_mask], dim=-1)
        gate = torch.softmax(self.gate_layer(gate_input), dim=-1)
        gamma_i = gate[..., 0:1]
        gamma_t = gate[..., 1:2]

        weighted_modalities = [gamma_i * img_z, gamma_t * tab_z]
        if self.use_multiplicative_interaction:
            weighted_modalities.append(img_z * tab_z)
        fusion_input = torch.cat(weighted_modalities, dim=-1)
        return self.fusion_mlp(fusion_input)

class GraphTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        transformer_d_model: int = 128,
        hidden_dim: int = 256,
        num_classes: int = 2,
        transformer_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
        use_gnn: bool = True,
        edge_dim: int = 2,
        tab_dim: int = 0,  # ← 设为 10 就会启用表格分支
        fusion_type: str = "reliability_gate",
        fusion_dim: int = 128,
    ):
        super().__init__()
        self.use_gnn = use_gnn
        self.edge_dim = edge_dim
        self.tab_dim = tab_dim
        self.fusion_type = fusion_type
        self.fusion_dim = fusion_dim

        # ---- 特征融合：ablation concat 或 reliability-aware cross-modal alignment ----
        if tab_dim > 0:
            img_dim = input_dim - tab_dim
            assert img_dim > 0, "input_dim 必须大于 tab_dim"
            if fusion_type == "concat":
                img_out_dim = transformer_d_model // 2
                tab_out_dim = transformer_d_model - img_out_dim
                # Keep legacy attribute names so old concat checkpoints can still load.
                self.img_proj = nn.Sequential(
                    nn.Linear(img_dim, img_out_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                )
                self.tab_mlp = TabularFeatureExtractor(
                    input_dim=tab_dim,
                    hidden_dim=64,
                    out_dim=tab_out_dim,
                    dropout=dropout,
                )
                self.concat_out_proj = nn.Identity()
                self.concat_fusion = None
                self.reliability_fusion = None
            elif fusion_type == "concat_mlp":
                self.img_proj = None
                self.tab_mlp = None
                self.concat_out_proj = None
                self.concat_fusion = ConcatenationFusion(
                    image_dim=img_dim,
                    tab_dim=tab_dim,
                    fusion_dim=fusion_dim,
                    output_dim=transformer_d_model,
                    dropout=dropout,
                )
                self.reliability_fusion = None
            elif fusion_type in ("reliability_gate", "reliability_gate_no_interaction"):
                self.img_proj = None
                self.tab_mlp = None
                self.concat_out_proj = None
                self.concat_fusion = None
                self.reliability_fusion = ReliabilityAwareCrossModalFusion(
                    image_dim=img_dim,
                    tab_dim=tab_dim,
                    fusion_dim=fusion_dim,
                    output_dim=transformer_d_model,
                    missing_mask_dim=tab_dim // 2,
                    dropout=dropout,
                    use_multiplicative_interaction=(fusion_type == "reliability_gate"),
                )
            else:
                raise ValueError(f"Unknown fusion_type: {fusion_type}")
            fused_dim = transformer_d_model
        else:
            self.img_proj = None
            self.tab_mlp = None
            self.concat_out_proj = None
            self.concat_fusion = None
            self.reliability_fusion = None
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, transformer_d_model),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            fused_dim = transformer_d_model

        # ---- 可选 GNN（注意 edge_dim）----
        if use_gnn:
            self.gnn = GATConv(
                in_channels=fused_dim,
                out_channels=transformer_d_model,
                heads=heads,
                concat=False,
                edge_dim=edge_dim,
                dropout=dropout,
                add_self_loops=False,
            )
        else:
            self.gnn = None

        # ---- Transformer 编码器 ----
        enc = nn.TransformerEncoderLayer(
            d_model=transformer_d_model,
            nhead=4,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=transformer_layers)
        self.pos_encoder = PositionalEncoding(transformer_d_model, max_len=2048)

        # ---- 分类头 ----
        self.head = nn.Sequential(
            nn.LayerNorm(transformer_d_model),
            nn.Linear(transformer_d_model, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def _fuse_features(self, x: torch.Tensor, missing_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (N_nodes, input_dim)
        if self.tab_dim > 0:
            img_x = x[..., :-self.tab_dim]
            tab_x = x[..., -self.tab_dim:]
            if self.fusion_type == "concat":
                img_h = self.img_proj(img_x)
                tab_h = self.tab_mlp(tab_x)
                return self.concat_out_proj(torch.cat([img_h, tab_h], dim=-1))
            if self.fusion_type == "concat_mlp":
                return self.concat_fusion(img_x, tab_x)
            return self.reliability_fusion(img_x, tab_x, missing_mask=missing_mask)
        else:
            return self.input_proj(x)

    def _to_padded_sequence(self, x: torch.Tensor, batch: torch.Tensor):
        if isinstance(batch, Batch):
            batch_vec = batch.batch
        else:
            batch_vec = batch
        num_graphs = int(batch_vec.max().item()) + 1
        counts = torch.bincount(batch_vec, minlength=num_graphs)
        T_max = int(counts.max().item())
        D = x.size(-1)
        device = x.device

        x_pad = x.new_zeros((num_graphs, T_max, D))
        pad_mask = torch.ones((num_graphs, T_max), dtype=torch.bool, device=device)

        start = 0
        for g, n in enumerate(counts.tolist()):
            if n == 0: 
                continue
            sl = slice(start, start + n)
            x_pad[g, :n] = x[sl]
            pad_mask[g, :n] = False
            start += n
        return x_pad, pad_mask

    def forward(self, data: Batch, return_gat_attention: bool = False):
        # 1) 融合节点特征
        x = self._fuse_features(data.x, missing_mask=getattr(data, "missing_mask", None))

        # 2) ★ 关键修复：把 edge_attr 喂给 GATConv
        if self.gnn is not None:
            if getattr(data, "edge_attr", None) is None:
                raise ValueError("缺少 edge_attr（形状应为 [E, edge_dim]），GraphBuilder 需在 Data 里设置。")
            if return_gat_attention:
                x, gat_attention = self.gnn(
                    x,
                    data.edge_index,
                    edge_attr=data.edge_attr,
                    return_attention_weights=True,
                )
            else:
                x = self.gnn(x, data.edge_index, edge_attr=data.edge_attr)
                gat_attention = None
        else:
            gat_attention = None

        # 3) 还原为 (B,T,D) + padding mask
        x_pad, pad_mask = self._to_padded_sequence(x, data)

        # 4) PE + Transformer（mask 忽略补齐位）
        x_pad = self.pos_encoder(x_pad)
        x_trans = self.transformer(x_pad, src_key_padding_mask=pad_mask)

        # 5) masked mean pooling → 图级表征
        valid = (~pad_mask).float()
        g_repr = (x_trans * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1.0)

        # 6) 分类头
        logits = self.head(g_repr)
        if return_gat_attention:
            return logits, gat_attention
        return logits
