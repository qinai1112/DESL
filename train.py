import os
import argparse
import subprocess
import json
import numpy as np
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch_geometric.data import Batch
from sklearn.metrics import precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, confusion_matrix

from graphtransformer import GraphTransformer
from graph_builder import GraphBuilder
from losses import MultiTaskLoss, FocalLoss, ClassBalancedLoss
import matplotlib.pyplot as plt
import pandas as pd
import math

# ======================
# 配置开关
# ======================
END2END = False   # True=端到端训练，False=用预提取特征

# ======================
# 默认配置
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.environ.get("HT_DATA_DIR", os.path.join(BASE_DIR, "data"))
RESULTS_DIR = os.environ.get("HT_RESULTS_DIR", os.path.join(BASE_DIR, "results"))
MODEL_TRAIN_DIR = os.path.join(BASE_DIR, "models_train")

DEFAULT_MODE = "train"
DEFAULT_FINETUNE = True
DEFAULT_ROOT_DIR = os.path.join(DATASET_DIR, "ALL")
DEFAULT_LABEL_PATH = os.path.join(DATASET_DIR, "FiveIndicators.xlsx")
DEFAULT_MODEL_PATH = os.path.join(MODEL_TRAIN_DIR, "graph_transformer.pt")
DEFAULT_FEATURE_IMPROVED = os.path.join(RESULTS_DIR, "Features_improved.npz")
DEFAULT_FEATURE_FINETUNE = os.path.join(RESULTS_DIR, "Features_finetune.npz")
log_file_path = os.path.join(RESULTS_DIR, "Training_log.csv")

DEFAULT_EPOCHS = 200
DEFAULT_LR = 1e-4
DEFAULT_BATCH_SIZE = 8
DEFAULT_VAL_RATIO = 0.2
DEFAULT_PATIENCE = 20
DEFAULT_SEED = 42

# 新增配置参数：Dropout 和 Weight Decay
DEFAULT_DROPOUT = 0.4  # 默认的 Dropout 值
DEFAULT_WEIGHT_DECAY = 1e-3  # 默认的 Weight Decay 值

DEFAULT_PREDICT_PATIENTS = []  # patient IDs to score in --mode predict

# 损失函数选择
default_loss_type = "ce"  # "ce", "focal", "cbloss"

MODEL_HIDDEN = 128
MODEL_LAYERS = 2  # not used: the model has one GATConv layer (see graphtransformer.py)
MODEL_NUM_CLASSES = 2

USE_GNN = True
POS_ENC_DIM = 64
TRANSFORMER_D_MODEL = 128
TRANSFORMER_LAYERS = 3
EDGE_DIM = 2
TAU = 180
IMAGE_FEATURE_DIM = 2048
NUM_INDICATORS = 7  # sex, age, FT3, FT4, TSH, TgAb, TPOAb
TAB_DIM = NUM_INDICATORS * 2  # 输入给模型的 tabular 维度 (值 + mask)
DEFAULT_FUSION_TYPE = "reliability_gate"
DEFAULT_FUSION_DIM = 128

# ======================
# 工具函数
# ======================
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_tab_dim_from_input_dim(in_dim, default_tab_dim=TAB_DIM):
    """
    Infer the actual tabular width from the image feature width so slicing
    stays correct if a feature file was generated with a different tabular
    configuration.
    """
    if in_dim is None:
        return default_tab_dim
    if in_dim > IMAGE_FEATURE_DIM:
        inferred = int(in_dim - IMAGE_FEATURE_DIM)
        if inferred > 0:
            return inferred
    return 0


def infer_tab_dim_from_state_dict(state_dict, fallback_tab_dim):
    for key in ("tab_mlp.net.0.weight", "reliability_fusion.tab_align_mlp.0.weight"):
        if key in state_dict:
            return int(state_dict[key].shape[1])
    for key, value in state_dict.items():
        if key.endswith("tab_align_mlp.0.weight") and hasattr(value, "shape"):
            return int(value.shape[1])
    return int(fallback_tab_dim)


def infer_fusion_type_from_state_dict(state_dict, default_fusion_type=DEFAULT_FUSION_TYPE):
    if any(k.startswith("concat_fusion.") for k in state_dict):
        return "concat_mlp"
    if any(k.startswith("reliability_fusion.") for k in state_dict):
        return "reliability_gate"
    if any(k.startswith("img_proj.") or k.startswith("tab_mlp.") for k in state_dict):
        return "concat"
    return default_fusion_type


def prepare_features(finetune, root_dir, label_path, mode="train"):
    """
    训练/预测通用的特征获取函数：
    - 预测模式优先复用已存在的 .npz；
    - 如需生成，统一用函数导入的方式调用，不再使用 subprocess。
    """
    # 优先选择（存在就直接返回）
    end2end_file = os.path.join(RESULTS_DIR, "Features_end2end.npz")
    finetune_file = DEFAULT_FEATURE_FINETUNE
    improved_file = DEFAULT_FEATURE_IMPROVED

    # 预测模式：尽量避免重算，按优先级复用
    if mode == "predict":
        for cand in [end2end_file, finetune_file, improved_file]:
            if os.path.exists(cand):
                return cand
        # 都不存在则生成一个（遵循 finetune 开关）
        out_file = finetune_file if finetune else improved_file
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        from feature_extractor_finetune import extract_and_save
        extract_and_save(root_dir, label_path, out_file, finetune=finetune)
        return out_file

    # 训练模式
    if END2END:
        os.makedirs(os.path.dirname(end2end_file), exist_ok=True)
        from feature_extractor_finetune import extract_and_save
        extract_and_save(root_dir, label_path, end2end_file, finetune=finetune)
        return end2end_file
    else:
        out_file = finetune_file if finetune else improved_file
        if not os.path.exists(out_file):
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            from feature_extractor_finetune import extract_and_save
            extract_and_save(root_dir, label_path, out_file, finetune=finetune)
        return out_file

def load_patient_graphs(data_path: str):
    """
    兼容 .npz 中含 / 不含 tabular_feats 的两种情形。
    如果 tabular_feats 存在，会优先传入 GraphBuilder.build_patient_graph 的新版接口。
    若 GraphBuilder 不支持新版接口（TypeError），会把 tabular 与 image 先 concat，再调用旧版接口。
    """
    npz = np.load(data_path, allow_pickle=True)
    feats = npz["features"] if "features" in npz.files else npz["feats"]
    labels = npz["labels"]
    patients = npz["patients"]
    dates = npz["dates"]
    tabular_feats = npz["tabular_feats"] if "tabular_feats" in npz.files else np.zeros((feats.shape[0], 0), dtype=feats.dtype)

    builder = GraphBuilder()
    graphs = []
    patient_ids = np.unique(patients)

    for pid in patient_ids:
        idx = np.where(patients == pid)[0]
        f = feats[idx]
        tf = tabular_feats[idx]
        d = dates[idx]
        l = labels[idx]

        g = builder.build_patient_graph(pid, f, tf, d, l)
        if g is None:
            continue
        g.patient_id = pid
        graphs.append(g)

    return graphs


def convert_to_next_visit_prediction(graph_list):
    raise NotImplementedError("Next-Visit conversion is handled in load_patient_graphs().")


def load_fixed12m_samples(
    data_path: str,
    horizon_days=365,
    tolerance_days=120,
    index_requires_negative=True,
):
    npz = np.load(data_path, allow_pickle=True)
    feats = npz["features"] if "features" in npz.files else npz["feats"]
    labels = npz["labels"]
    patients = npz["patients"]
    dates = npz["dates"]
    tabular_feats = npz["tabular_feats"] if "tabular_feats" in npz.files else np.zeros((feats.shape[0], 0), dtype=feats.dtype)

    builder = GraphBuilder()
    samples = []
    censored = 0
    total = 0
    patient_ids = np.unique(patients)

    for pid in patient_ids:
        idx = np.where(patients == pid)[0]
        f = feats[idx]
        tf = tabular_feats[idx]
        d = dates[idx]
        l = labels[idx]
        cur_samples, cur_censored, cur_total = builder.build_fixed12m_samples(
            pid,
            f,
            tf,
            d,
            l,
            horizon_days=horizon_days,
            tolerance_days=tolerance_days,
            index_requires_negative=index_requires_negative,
        )
        samples.extend(cur_samples)
        censored += cur_censored
        total += cur_total

    stats = {
        "total_index": int(total),
        "censored": int(censored),
        "kept": int(len(samples)),
        "pos_rate": float(np.mean([int(s.y.item()) for s in samples])) if samples else float("nan"),
    }
    return samples, stats


# --- replace split_train_val with a stratified version ---
def split_train_val(graphs, val_ratio=0.2, seed=42):
    rng = np.random.RandomState(seed)
    # 用病人级标签（是否曾经为 1）做分层
    labels = np.array([int(g.y.item()) for g in graphs], dtype=int)
    idx_pos = np.where(labels == 1)[0]
    idx_neg = np.where(labels == 0)[0]
    rng.shuffle(idx_pos); rng.shuffle(idx_neg)
    n_val_pos = int(len(idx_pos) * val_ratio)
    n_val_neg = int(len(idx_neg) * val_ratio)
    val_idx = np.concatenate([idx_pos[:n_val_pos], idx_neg[:n_val_neg]])
    train_idx = np.concatenate([idx_pos[n_val_pos:], idx_neg[n_val_neg:]])
    rng.shuffle(train_idx); rng.shuffle(val_idx)
    return [graphs[i] for i in train_idx], [graphs[i] for i in val_idx]


def split_train_val_by_patient(samples, val_ratio=0.2, seed=42):
    patient_ids = np.array(sorted({s.patient_id for s in samples}))
    patient_labels = []
    for pid in patient_ids:
        ys = [int(s.y.item()) for s in samples if s.patient_id == pid]
        patient_labels.append(max(ys) if ys else 0)
    patient_labels = np.array(patient_labels, dtype=int)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(splitter.split(patient_ids, patient_labels))
    train_patients = set(patient_ids[train_idx])
    val_patients = set(patient_ids[val_idx])

    train_samples = [s for s in samples if s.patient_id in train_patients]
    val_samples = [s for s in samples if s.patient_id in val_patients]

    assert train_patients.isdisjoint(val_patients)
    return train_samples, val_samples



def get_model(
    device,
    in_dim=None,
    tab_dim=TAB_DIM,
    fusion_type=DEFAULT_FUSION_TYPE,
    fusion_dim=DEFAULT_FUSION_DIM,
):
    """
    in_dim: 节点特征维度（影像 + 表格）
    tab_dim: 节点特征最后多少维是表格（用于 TabularFeatureExtractor）
    """
    if in_dim is None:
        in_dim = 2048  # 兜底

    model = GraphTransformer(
        # 注意：如果你已按我之前给的 graphtransformer.py 替换过，
        # 这里的参数名是 input_dim / edge_dim / tab_dim
        input_dim=in_dim,
        transformer_d_model=TRANSFORMER_D_MODEL,
        hidden_dim=MODEL_HIDDEN * 2,
        num_classes=MODEL_NUM_CLASSES,
        transformer_layers=TRANSFORMER_LAYERS,
        heads=4,
        dropout=DEFAULT_DROPOUT,
        use_gnn=USE_GNN,
        edge_dim=EDGE_DIM,
        tab_dim=tab_dim,     # ★ 新增：启用表格分支
        fusion_type=fusion_type,
        fusion_dim=fusion_dim,
    ).to(device)
    return model

# train.py 新增函数

def impute_missing_values(train_graphs, val_graphs, tab_dim_raw=NUM_INDICATORS):
    """
    仅利用 train_graphs 计算中位数，然后填充 train 和 val。
    假设 tabular 特征在 graph.x 的最后 2*tab_dim_raw 维度中：
    [...ResNet(2048)..., raw_ft3, raw_ft4..., mask_ft3, mask_ft4...]
    """
    print("正在进行抗泄露的数据填充 (Imputing)...")
    if tab_dim_raw <= 0:
        print("未检测到 tabular 特征，跳过填充。")
        return
    
    # 1. 收集所有训练集的 tabular 数据 (只取原始值部分，不取 mask)
    # graph.x shape: [Nodes, 2048 + 2*K]
    # 我们需要的是倒数 2K 到倒数 K 列（即原始指标）
    all_train_values = []
    for g in train_graphs:
        # 取出 raw features (排除 ResNet 和 Mask)
        # 假设 tab_dim=2K (K个值+K个mask)，则原始值在 -2K:-K
        raw_part = g.x[:, -2*tab_dim_raw : -tab_dim_raw]
        all_train_values.append(raw_part)
    
    all_train_values = torch.cat(all_train_values, dim=0) # (Total_Nodes, 5)

    # 2. 计算训练集的中位数 (忽略 NaN)
    # 注意：torch.nanmedian 在旧版本可能不稳定，如果报错可以用 numpy
    train_medians = torch.nanmedian(all_train_values, dim=0).values
    
    # 打印一下计算出的中位数，确保看起来正常
    print(f"训练集计算出的中位数: {train_medians.cpu().numpy()}")

    # 3. 定义填充函数
    def fill_graph_nan(graphs, medians):
        for g in graphs:
            # 找到 raw features 的位置
            start_idx = g.x.shape[1] - 2 * tab_dim_raw
            end_idx = g.x.shape[1] - tab_dim_raw
            
            raw_data = g.x[:, start_idx:end_idx]
            
            # 找到 NaN 的位置
            is_nan = torch.isnan(raw_data)
            
            # 用 medians 填充
            # 扩展 medians 以匹配节点数
            medians_expanded = medians.unsqueeze(0).expand_as(raw_data)
            
            # 填充
            raw_data[is_nan] = medians_expanded[is_nan]
            
            # 写回 graph.x
            g.x[:, start_idx:end_idx] = raw_data
            
            # 再次检查是否还有 NaN (防止全空列)
            if torch.isnan(g.x).any():
                g.x = torch.nan_to_num(g.x, nan=0.0) # 兜底：如果整列都空，填0

    # 4. 应用填充
    fill_graph_nan(train_graphs, train_medians)
    fill_graph_nan(val_graphs, train_medians)
    
    print("数据填充完成。")

def get_loss_fn(loss_type="ce", class_weights=None, label_smoothing=0.05):
    if loss_type == "ce":
        return MultiTaskLoss(class_weights=class_weights, label_smoothing=label_smoothing)
    if loss_type == "focal":
        return FocalLoss(gamma=2.0, alpha=class_weights, label_smoothing=label_smoothing)
    if loss_type == "cbloss":
        return ClassBalancedLoss(beta=0.9999, label_smoothing=label_smoothing)
    raise ValueError(f"Unknown loss type: {loss_type}")


def load_interval_map(csv_path):
    df = pd.read_csv(csv_path)
    for col in ["patient_id", "target_date", "dt_next_days"]:
        if col not in df.columns:
            raise ValueError(f"Missing column in interval CSV: {col}")
    df["target_date"] = pd.to_datetime(df["target_date"], errors="coerce").dt.date.astype(str)
    df["patient_id"] = df["patient_id"].astype(str)
    interval_map = {}
    for _, row in df.iterrows():
        key = (row["patient_id"], row["target_date"])
        interval_map[key] = float(row["dt_next_days"])
    return interval_map


def compute_bin_metrics(records, bins):
    rows = []
    for label, lo, hi in bins:
        subset = [r for r in records if (r["dt"] >= lo) and (r["dt"] < hi)]
        if not subset:
            rows.append({
                "bin": label,
                "n": 0,
                "pos_rate": float("nan"),
                "roc_auc": float("nan"),
                "pr_auc": float("nan"),
                "brier": float("nan"),
                "ece": float("nan"),
                "prec_0.5": float("nan"),
                "rec_0.5": float("nan"),
                "f1_0.5": float("nan"),
                "best_thr": float("nan"),
                "prec_best": float("nan"),
                "rec_best": float("nan"),
                "f1_best": float("nan"),
            })
            continue
        y_true = np.array([r["y_true"] for r in subset], dtype=int)
        y_prob = np.array([r["y_prob"] for r in subset], dtype=float)
        pos_rate = float(y_true.mean()) if len(y_true) else float("nan")
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            roc_auc = float("nan")
        try:
            pr_auc = float(average_precision_score(y_true, y_prob))
        except Exception:
            pr_auc = float("nan")
        brier = float(mean_squared_error(y_true, y_prob))
        ece = compute_ece(y_true, y_prob, n_bins=10)
        y_pred_05 = (y_prob >= 0.5).astype(int)
        prec_05 = precision_score(y_true, y_pred_05, zero_division=0)
        rec_05 = recall_score(y_true, y_pred_05, zero_division=0)
        f1_05 = f1_score(y_true, y_pred_05, zero_division=0)
        best_thr, _ = find_best_threshold(y_true, y_prob, metric="f1")
        y_pred_best = (y_prob >= best_thr).astype(int)
        prec_best = precision_score(y_true, y_pred_best, zero_division=0)
        rec_best = recall_score(y_true, y_pred_best, zero_division=0)
        f1_best = f1_score(y_true, y_pred_best, zero_division=0)

        rows.append({
            "bin": label,
            "n": int(len(y_true)),
            "pos_rate": pos_rate,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "brier": brier,
            "ece": ece,
            "prec_0.5": prec_05,
            "rec_0.5": rec_05,
            "f1_0.5": f1_05,
            "best_thr": best_thr,
            "prec_best": prec_best,
            "rec_best": rec_best,
            "f1_best": f1_best,
        })
    return pd.DataFrame(rows)


# ======================
# 阈值搜索函数
# ======================
def find_best_threshold(y_true, y_prob, metric="f1"):
    best_t, best_score = 0.5, -1.0
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    for t in np.linspace(0.01, 0.99, 99):
        y_pred = (y_prob >= t).astype(int)
        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "balanced_acc":
            tp = ((y_true == 1) & (y_pred == 1)).sum()
            tn = ((y_true == 0) & (y_pred == 0)).sum()
            fp = ((y_true == 0) & (y_pred == 1)).sum()
            fn = ((y_true == 1) & (y_pred == 0)).sum()
            tpr = tp / (tp + fn + 1e-12)
            tnr = tn / (tn + fp + 1e-12)
            score = 0.5 * (tpr + tnr)
        else:
            score = (y_pred == y_true).mean()
        if score > best_score:
            best_score, best_t = score, t
    return best_t, best_score


def compute_ece(y_true, y_prob, n_bins=10):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lo, bin_hi = bins[i], bins[i + 1]
        mask = (y_prob > bin_lo) & (y_prob <= bin_hi)
        if not mask.any():
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += (mask.mean()) * abs(bin_acc - bin_conf)
    return float(ece)


# ======================
# 评估函数
# ======================
def _as_list(x, batch_size):
    if isinstance(x, (list, tuple)):
        return list(x)
    if torch.is_tensor(x):
        x = x.cpu().tolist()
        if isinstance(x, list):
            return x
        return [x] * batch_size
    return [x] * batch_size


def evaluate(model, loader, device, threshold_metric="f1"):
    model.eval()
    y_true, y_prob = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)  # [B,2]
            prob = torch.softmax(out, dim=1)[:, 1]

            y = batch.y.view(-1).long()
            y_true.extend(y.cpu().numpy().tolist())
            y_prob.extend(prob.cpu().numpy().tolist())

    # AUC/PR-AUC（注意：若验证集只有单一类别会报错）
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        pr_auc = float("nan")

    # 阈值搜索 + 用同一阈值算全部指标
    threshold, _ = find_best_threshold(y_true, y_prob, metric=threshold_metric)
    y_pred = (np.array(y_prob) >= threshold).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = (y_pred == np.array(y_true)).mean()
    tp = ((np.array(y_true) == 1) & (y_pred == 1)).sum()
    tn = ((np.array(y_true) == 0) & (y_pred == 0)).sum()
    fp = ((np.array(y_true) == 0) & (y_pred == 1)).sum()
    fn = ((np.array(y_true) == 1) & (y_pred == 0)).sum()
    tpr = tp / (tp + fn + 1e-12)
    tnr = tn / (tn + fp + 1e-12)
    bal_acc = 0.5 * (tpr + tnr)
    mae = mean_absolute_error(y_true, y_prob)
    mse = mean_squared_error(y_true, y_prob)
    brier = mean_squared_error(y_true, y_prob)
    ece = compute_ece(y_true, y_prob, n_bins=10)
    cm = confusion_matrix(y_true, y_pred)

    print(f"Confusion Matrix:\n{cm}")
    return auc, pr_auc, acc, bal_acc, precision, recall, f1, mae, mse, brier, ece, cm, threshold


def evaluate_collect(model, loader, device):
    model.eval()
    y_true, y_prob, pids, tdates = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            prob = torch.softmax(out, dim=1)[:, 1]
            y = batch.y.view(-1).long()
            y_true.extend(y.cpu().numpy().tolist())
            y_prob.extend(prob.cpu().numpy().tolist())
            bs = y.shape[0]
            pid_list = _as_list(getattr(batch, "patient_id", None), bs)
            td_list = _as_list(getattr(batch, "target_date", None), bs)
            pids.extend([str(p) for p in pid_list])
            tdates.extend([str(t) for t in td_list])
    return y_true, y_prob, pids, tdates

# ======================
# EMA（仅用于验证）的权重跟踪
# ======================
# === FIXED EMA: track ONLY floating-point tensors ===

class EMA:
    def __init__(self, model, decay: float = 0.999, device=None):
        self.decay = decay
        self.device = device
        self.shadow = {}
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
                    t = v.detach().clone()
                    if device is not None:
                        t = t.to(device)
                    self.shadow[k] = t  # only float tensors are tracked

    @torch.no_grad()
    def swap_in(self, model):
        """
        将 EMA 的 shadow 权重临时加载进模型，用于验证；
        返回当前模型浮点权重的备份，验证后可 restore。
        只处理浮点条目（与 shadow 对齐）。
        """
        backup = {}
        msd = model.state_dict()
        for k, v_shadow in self.shadow.items():
            backup[k] = msd[k].detach().clone()
            msd[k].copy_(v_shadow)
        return backup

    @torch.no_grad()
    def restore(self, model, backup):
        """把 swap_in 前备份的浮点权重恢复回模型。"""
        msd = model.state_dict()
        for k, v in backup.items():
            msd[k].copy_(v)


    @torch.no_grad()
    def update(self, model):
        d = self.decay
        msd = model.state_dict()
        for k, v_shadow in self.shadow.items():  # only float keys
            v_model = msd[k].detach()
            # make sure dtype/device match shadow
            if v_model.dtype != v_shadow.dtype or v_model.device != v_shadow.device:
                v_model = v_model.to(dtype=v_shadow.dtype, device=v_shadow.device)
            v_shadow.mul_(d).add_(v_model, alpha=1.0 - d)

    @torch.no_grad()
    def apply_to(self, model):
        """Copy EMA weights into model (for eval)."""
        msd = model.state_dict()
        for k, v_shadow in self.shadow.items():
            msd[k].copy_(v_shadow)

    @torch.no_grad()
    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    @torch.no_grad()
    def load_state_dict(self, state):
        self.decay = state["decay"]
        self.shadow = state["shadow"]


# ======================
# 训练函数
# ======================
def train(
    data_path,
    model_path,
    epochs,
    lr,
    batch_size,
    val_ratio,
    patience,
    seed,
    task="next_visit",
    best_by="val_pr_auc",
    threshold_metric="f1",
    interval_csv=None,
    bin_csv_out=None,
    bin_edges=(180, 300, 450),
    fixed12m_horizon=365,
    fixed12m_tolerance=120,
    fixed12m_index_requires_negative=True,
    loss_type=default_loss_type,
    label_smoothing=0.0,
    use_class_weights=True,
    weight_decay=DEFAULT_WEIGHT_DECAY,
    tab_dim=TAB_DIM,
    fusion_type=DEFAULT_FUSION_TYPE,
    fusion_dim=DEFAULT_FUSION_DIM,
):
    set_seed(seed)

    # 创建空的 DataFrame 来保存日志
    columns = [
        "Epoch",
        "Train ACC",
        "Val AUC",
        "Val PR-AUC",
        "Val ACC",
        "Val Bal ACC",
        "Precision",
        "Recall",
        "F1",
        "MAE",
        "MSE",
        "Brier",
        "ECE",
        "Thr",
    ]
    training_log = pd.DataFrame(columns=columns)

    if task == "fixed12m":
        graphs, fixed12m_stats = load_fixed12m_samples(
            data_path,
            horizon_days=fixed12m_horizon,
            tolerance_days=fixed12m_tolerance,
            index_requires_negative=fixed12m_index_requires_negative,
        )
        print(f"[fixed12m] stats: {fixed12m_stats}")
    else:
        graphs = load_patient_graphs(data_path)
        fixed12m_stats = None
    if len(graphs) == 0:
        raise RuntimeError("No graphs built from data_path")

    # 根据第一个 graph 的节点特征维度推断 in_dim（影像 + tabular 的融合后维度）
    sample_in_dim = graphs[0].x.shape[1]
    print(f"Detected node feature dim = {sample_in_dim} (will be used as model input dim)")
    tab_dim_for_data = infer_tab_dim_from_input_dim(sample_in_dim, default_tab_dim=tab_dim)
    if tab_dim_for_data % 2 != 0:
        raise ValueError(f"tab_dim must be value+mask and therefore even, got {tab_dim_for_data}")
    tab_dim_raw_for_data = tab_dim_for_data // 2
    print(
        f"Detected tabular dim = {tab_dim_for_data} "
        f"(raw indicators={tab_dim_raw_for_data}, fusion_type={fusion_type}, fusion_dim={fusion_dim})"
    )

    if task == "fixed12m":
        train_graphs, val_graphs = split_train_val_by_patient(graphs, val_ratio=val_ratio, seed=seed)
    else:
        train_graphs, val_graphs = split_train_val(graphs, val_ratio=val_ratio, seed=seed)
    impute_missing_values(train_graphs, val_graphs, tab_dim_raw=tab_dim_raw_for_data)

    print(f"数据集: Train={len(train_graphs)}  Val={len(val_graphs)}")
    print(f"使用的损失函数: {loss_type}")

    labels_arr = np.array([int(g.y.item()) for g in train_graphs], dtype=int)
    class_counts = np.bincount(labels_arr, minlength=2)
    if use_class_weights:
        class_weights = torch.tensor([1.0/c if c > 0 else 0.0 for c in class_counts], dtype=torch.float)
        class_weights = class_weights / class_weights.sum() * len(class_counts)
    else:
        class_weights = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if class_weights is not None:
        class_weights = class_weights.to(device)

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=1, shuffle=False)

    # === 新增：用一个 batch 来获取节点特征维度 in_dim ===
    sample_batch = next(iter(train_loader))
    # PyG 的 Batch 对象：节点特征在 sample_batch.x，形状 [N_nodes_in_batch, in_dim]
    in_dim_from_batch = sample_batch.x.size(-1)
    print(f"[train] detected node feature dim from batch = {in_dim_from_batch}")

    # （可选）校验边特征维度，避免忘了把 edge_attr 放进 Data 对象
    assert hasattr(sample_batch, "edge_attr") and sample_batch.edge_attr.size(-1) == EDGE_DIM, \
        f"edge_attr last dim must be {EDGE_DIM}, got {getattr(sample_batch,'edge_attr', None)}"

    # === 修改：把维度和 tab_dim 一起传入 get_model ===
    tab_dim_for_model = infer_tab_dim_from_input_dim(in_dim_from_batch, default_tab_dim=tab_dim_for_data)
    model = get_model(
        device,
        in_dim=in_dim_from_batch,
        tab_dim=tab_dim_for_model,
        fusion_type=fusion_type,
        fusion_dim=fusion_dim,
    )
    loss_fn = get_loss_fn(loss_type, class_weights=class_weights, label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    # === 新增：EMA 仅用于验证 ===
    ema = EMA(model, decay=0.999)

    best_score, best_epoch, wait = -1.0, 0, 0
    best_metrics = {}
    train_accs, val_accs = [], []
    best_threshold = 0.5  # 初始阈值

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        y_true_train, y_pred_train = [], []

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            y = batch.y.view(-1).long()
            losses = loss_fn(out, y)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            # === 更新 EMA（训练仍使用原权重）===
            ema.update(model)

            total_loss += losses["total"].item()

            prob = torch.softmax(out, dim=1)[:, 1].detach().cpu().numpy()
            pred = (prob >= 0.5).astype(int)
            y_true_train.extend(y.detach().cpu().numpy().tolist())
            y_pred_train.extend(pred.tolist())

        train_acc = (np.array(y_true_train) == np.array(y_pred_train)).mean()
        train_accs.append(train_acc)

        # === 1. 切入 EMA 权重进行验证与评估 ===
        _backup = ema.swap_in(model)
        auc, pr_auc, val_acc, val_bal_acc, precision, recall, f1, mae, mse, brier, ece, cm, val_thr = evaluate(
            model,
            val_loader,
            device,
            threshold_metric=threshold_metric,
        )
        val_accs.append(val_acc)

        if best_by == "val_auc":
            val_score = auc
        elif best_by == "val_bal_acc":
            val_score = val_bal_acc
        else:
            val_score = pr_auc
        scheduler.step(val_score)

        # 在训练日志中记录当前 epoch 的数据
        epoch_data = {
            "Epoch": epoch + 1,
            "Train ACC": train_acc,
            "Val AUC": auc,
            "Val PR-AUC": pr_auc,
            "Val ACC": val_acc,
            "Val Bal ACC": val_bal_acc,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "MAE": mae,
            "MSE": mse,
            "Brier": brier,
            "ECE": ece,
            "Thr": val_thr,
        }

        # 使用 pd.concat 来追加数据
        epoch_df = pd.DataFrame([epoch_data])
        training_log = pd.concat([training_log, epoch_df], ignore_index=True)

        # 每个epoch只输出ACC和AUC
        print(
            f"Epoch {epoch + 1}/{epochs}, Train ACC={train_acc:.4f}, Val AUC={auc:.4f}, "
            f"Val PR-AUC={pr_auc:.4f}, Val ACC={val_acc:.4f}, Val Bal ACC={val_bal_acc:.4f}, Thr={val_thr:.2f}"
        )

        # === 2. 继续保存最优模型（此时 model.state_dict() 获取的将是 EMA 权重）===
        if val_score > best_score:
            best_score, best_epoch, wait = val_score, epoch + 1, 0
            best_threshold = float(val_thr)
            best_metrics = {
                "epoch": best_epoch,
                "val_auc": float(auc),
                "val_pr_auc": float(pr_auc),
                "val_acc": float(val_acc),
                "val_bal_acc": float(val_bal_acc),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "brier": float(brier),
                "ece": float(ece),
                "threshold": float(val_thr),
            }
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "threshold": best_threshold,
                "best_by": best_by,
                "best_score": float(val_score),
                "tab_dim": int(tab_dim_for_model),
                "fusion_type": fusion_type,
                "fusion_dim": int(fusion_dim),
            }, model_path)
            print(
                f"保存最佳模型 (Epoch {best_epoch}, best_by={best_by}, score={val_score:.4f}, Thr={best_threshold:.2f}) -> {model_path}")
        else:
            wait += 1
            
        # === 3. 评估和保存都完成后，恢复原始训练权重 ===
        ema.restore(model, _backup)

        # === 4. 早停判断 ===
        if wait >= patience:
            print(f"提前停止: 最佳在 Epoch {best_epoch}, best_by={best_by}, score={best_score:.4f}")
            break

    # 保存训练日志到CSV文件
    out_log = log_file_path if task != "fixed12m" else os.path.join(RESULTS_DIR, "training_log_fixed12m.csv")
    os.makedirs(os.path.dirname(out_log), exist_ok=True)
    training_log.to_csv(out_log, index=False)
    print(f"训练日志已保存到 {out_log}")

    if task == "fixed12m" and fixed12m_stats is not None:
        summary_path = os.path.join(RESULTS_DIR, "fixed12m_eval_summary.json")
        summary = {
            "sample_stats": fixed12m_stats,
            "best_by": best_by,
            "best_score": float(best_score),
            "best_metrics": best_metrics,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[fixed12m] summary saved to {summary_path}")

    if task == "next_visit" and interval_csv:
        interval_map = load_interval_map(interval_csv)
        y_true, y_prob, pids, tdates = evaluate_collect(model, val_loader, device)
        records = []
        for yt, yp, pid, tdate in zip(y_true, y_prob, pids, tdates):
            if tdate in (None, "None", "nan"):
                continue
            key = (str(pid), str(pd.to_datetime(tdate, errors="coerce").date()))
            dt = interval_map.get(key)
            if dt is None:
                continue
            records.append({"y_true": int(yt), "y_prob": float(yp), "dt": float(dt)})
        if records:
            edges = list(bin_edges)
            bins = [
                (f"<{edges[0]}", -1e9, edges[0]),
                (f"{edges[0]}-{edges[1]}", edges[0], edges[1]),
                (f"{edges[1]}-{edges[2]}", edges[1], edges[2]),
                (f">{edges[2]}", edges[2], 1e9),
            ]
            bin_df = compute_bin_metrics(records, bins)
            out_bin = bin_csv_out or os.path.join(RESULTS_DIR, "bin_metrics_next_visit.csv")
            os.makedirs(os.path.dirname(out_bin), exist_ok=True)
            bin_df.to_csv(out_bin, index=False)
            print(f"bin metrics saved to {out_bin}")

    # 绘图（只保留 ACC 曲线）
    epochs_range = range(1, len(val_accs) + 1)
    plt.figure(figsize=(6, 5))
    plt.plot(epochs_range, train_accs, label="Train ACC")
    plt.plot(epochs_range, val_accs, label="Val ACC")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train vs Validation ACC")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    curve_path = os.path.join(RESULTS_DIR, "train_val_curves.png")
    os.makedirs(os.path.dirname(curve_path), exist_ok=True)
    plt.savefig(curve_path)
    plt.close()


# ======================
# 预测函数
# ======================
def predict(
    data_path,
    model_path,
    patient_ids,
    horizon_years=5,
    plot=True,
    tab_dim=TAB_DIM,
    fusion_type=DEFAULT_FUSION_TYPE,
    fusion_dim=DEFAULT_FUSION_DIM,
):
    graphs = load_patient_graphs(data_path)
    if len(graphs) == 0:
        raise RuntimeError("No graphs built from data_path")
    patient_map = {g.patient_id: g for g in graphs}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) 根据数据推断输入维度
    in_dim = graphs[0].x.shape[1]

    # 2) 读取 checkpoint & state_dict
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    sd = ckpt["model_state"] if "model_state" in ckpt else ckpt
    keys = list(sd.keys())

    data_tab_dim = infer_tab_dim_from_input_dim(in_dim, default_tab_dim=tab_dim)
    ckpt_fusion_type = ckpt.get("fusion_type") if isinstance(ckpt, dict) else None
    fusion_type_for_model = ckpt_fusion_type or infer_fusion_type_from_state_dict(sd, fusion_type)
    fusion_dim_for_model = int(ckpt.get("fusion_dim", fusion_dim)) if isinstance(ckpt, dict) else fusion_dim
    tab_dim_for_model = int(ckpt.get("tab_dim", infer_tab_dim_from_state_dict(sd, data_tab_dim))) if isinstance(ckpt, dict) else data_tab_dim

    # 4) 从权重形状里反推 d_model（Transformer的隐藏维）与 d_ff（前馈层维度）
    #    线索：transformer.layers.0.linear1.weight 的形状是 [d_ff, d_model]
    lin1_key = None
    for k in keys:
        if "transformer.layers.0.linear1.weight" in k:
            lin1_key = k
            break
    if lin1_key is not None:
        d_ff, d_model = sd[lin1_key].shape
    else:
        d_model = TRANSFORMER_D_MODEL
        d_ff = MODEL_HIDDEN * 2

    # 5) 统计层数
    layer_indices = set()
    for k in keys:
        if k.startswith("transformer.layers."):
            parts = k.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                layer_indices.add(int(parts[2]))
    num_layers = (max(layer_indices) + 1) if layer_indices else TRANSFORMER_LAYERS

    # 6) 构建与 checkpoint 对齐的模型（注意：tab_dim 根据是否存在 tab 分支来设）
    model = GraphTransformer(
        input_dim=in_dim,
        transformer_d_model=d_model,
        hidden_dim=d_ff,                 # dim_feedforward 与 ckpt 对齐
        num_classes=MODEL_NUM_CLASSES,
        transformer_layers=num_layers,
        heads=4,
        dropout=DEFAULT_DROPOUT,
        use_gnn=USE_GNN,
        edge_dim=EDGE_DIM,
        tab_dim=tab_dim_for_model,
        fusion_type=fusion_type_for_model,
        fusion_dim=fusion_dim_for_model,
    ).to(device)

    # 7) 加载参数（strict=False 以忽略 pos_encoder.pe 等非持久 buffer 差异）
    res = model.load_state_dict(sd, strict=False)
    print(
        f"[predict] ckpt适配: fusion_type={fusion_type_for_model}, tab_dim={tab_dim_for_model}, "
        f"fusion_dim={fusion_dim_for_model}, d_model={d_model}, d_ff={d_ff}, layers={num_layers}"
    )
    print(f"[predict] missing_keys={res.missing_keys}, unexpected_keys={res.unexpected_keys}")

    threshold = ckpt.get("threshold", 0.5)
    model.eval()

    # 8) 推断
    results = {}
    with torch.no_grad():
        for pid in patient_ids:
            if pid not in patient_map:
                results[pid] = "No data"
                continue
            data = patient_map[pid]
            batch = Batch.from_data_list([data]).to(device)
            out = model(batch)
            prob = torch.softmax(out, dim=1)[0, 1].item()
            pred = int(prob >= threshold)
            results[pid] = {"pred_class": pred, "prob_sick": prob, "threshold": threshold}
            print(f"Patient {pid}: class={pred}, prob={prob:.3f}, thr={threshold:.2f}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tab-dim", type=int, default=TAB_DIM)
    parser.add_argument("--mode", type=str, default=DEFAULT_MODE, choices=["train", "predict"])
    parser.add_argument("--finetune", type=lambda x: str(x).lower() == "true", default=DEFAULT_FINETUNE)
    parser.add_argument("--root_dir", type=str, default=DEFAULT_ROOT_DIR)
    parser.add_argument("--label_path", type=str, default=DEFAULT_LABEL_PATH)
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--val_ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--predict_patients", nargs="*", default=DEFAULT_PREDICT_PATIENTS)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--use_class_weights", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument(
        "--fusion_type",
        type=str,
        default=DEFAULT_FUSION_TYPE,
        choices=["reliability_gate", "reliability_gate_no_interaction", "concat_mlp", "concat"],
    )
    parser.add_argument("--fusion_dim", type=int, default=DEFAULT_FUSION_DIM)
    parser.add_argument("--task", type=str, default="next_visit", choices=["next_visit", "fixed12m"])
    parser.add_argument("--best_by", type=str, default="val_pr_auc", choices=["val_auc", "val_pr_auc", "val_bal_acc"])
    parser.add_argument("--threshold_metric", type=str, default="f1", choices=["f1", "balanced_acc"])
    parser.add_argument("--interval_csv", type=str, default="", help="CSV with dt_next_days for bin metrics")
    parser.add_argument("--bin_csv_out", type=str, default="", help="Output CSV for bin metrics")
    parser.add_argument("--bin_edges", type=str, default="180,300,450", help="Comma-separated bin edges")
    parser.add_argument("--fixed12m_horizon", type=int, default=365)
    parser.add_argument("--fixed12m_tolerance", type=int, default=120)
    parser.add_argument("--fixed12m_index_requires_negative", type=lambda x: str(x).lower() == "true", default=True)
    args = parser.parse_args()

    feature_file = prepare_features(args.finetune, args.root_dir, args.label_path, mode=args.mode)
    print(f"使用特征文件: {feature_file} (finetune={args.finetune})")

    if args.mode == "train":
        train(
            data_path=feature_file,
            model_path=args.model_path,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            val_ratio=args.val_ratio,
            patience=args.patience,
            seed=args.seed,
            task=args.task,
            best_by=args.best_by,
            threshold_metric=args.threshold_metric,
            interval_csv=args.interval_csv or None,
            bin_csv_out=args.bin_csv_out or None,
            bin_edges=tuple(int(x) for x in args.bin_edges.split(",")),
            fixed12m_horizon=args.fixed12m_horizon,
            fixed12m_tolerance=args.fixed12m_tolerance,
            fixed12m_index_requires_negative=args.fixed12m_index_requires_negative,
            loss_type=default_loss_type,
            label_smoothing=args.label_smoothing,
            use_class_weights=args.use_class_weights,
            weight_decay=args.weight_decay,
            tab_dim=args.tab_dim,
            fusion_type=args.fusion_type,
            fusion_dim=args.fusion_dim,
        )
    elif args.mode == "predict":
        predict(
            data_path=feature_file,
            model_path=args.model_path,
            patient_ids=args.predict_patients,
            horizon_years=5,
            plot=True,
            tab_dim=args.tab_dim,
            fusion_type=args.fusion_type,
            fusion_dim=args.fusion_dim,
        )


if __name__ == "__main__":
    main()
