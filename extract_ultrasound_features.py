import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
import argparse
from torchvision.models import resnet50, ResNet50_Weights
from tabular_extractor import preprocess_tabular   # 🔥 依赖你的 tabular_extractor

DEFAULT_BATCH_SIZE = 1
DEFAULT_MAX_IMAGES = 5

# ======================
# Dataset
# ======================
class UltrasoundDataset(Dataset):
    def __init__(self, root_dir, label_path, transform=None, max_images=5, seed=42):
        self.root_dir = root_dir
        self.transform = transform
        self.max_images = max_images

        # 1. 读取 Excel
        self.label_df = pd.read_excel(label_path)

        # 2. 筛选出本地确实存在文件夹的患者记录
        self.patient_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.label_df = self.label_df[self.label_df["patient_Id"].astype(str).isin(self.patient_dirs)]
        
        # 🚨【核心修改1】重置索引，确保遍历时与 tabular_data 的行严格对应
        self.label_df = self.label_df.reset_index(drop=True)

        # 指标字段
        self.rng = np.random.RandomState(seed)  # 随机选图，固定种子保证可复现
        self.indicator_cols = ["sex", "age", "FT3", "FT4", "TSH", "TgAb", "TPOAb"]

        # 预处理指标
        self.tabular_data, self.medians = preprocess_tabular(self.label_df, self.indicator_cols)

        self.samples = []
        
        # 🚨【核心修改2】以 Excel 表格记录为主导，而非本地文件夹
        grouped = self.label_df.groupby("patient_Id")
        for pid, group in grouped:
            pid_str = str(pid)
            patient_path = os.path.join(root_dir, pid_str)
            
            for idx, row in group.iterrows():
                # 1. 解析日期
                raw_date = row["time"]
                if pd.isna(raw_date):
                    continue
                if isinstance(raw_date, pd.Timestamp):
                    date_str = raw_date.strftime("%Y-%m-%d")
                else:
                    date_str = str(raw_date).split()[0]
                    
                # 2. 获取标签
                sic = row["SIC"]
                sic = int(sic) if pd.notna(sic) else -1
                
                # 3. 获取表格特征 (依赖重置后的索引)
                tab_vec = self.tabular_data[idx]
                
                # 4. 查找图片路径
                date_path = os.path.join(patient_path, date_str)
                img_paths = []
                if os.path.isdir(date_path):
                    all_imgs = sorted(glob.glob(os.path.join(date_path, "*")))
                    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
                    img_paths = [p for p in all_imgs if p.lower().endswith(valid_exts)]
                    
                    # 🚨【核心修改3】图片超过上限时，使用“等间隔采样”代替“死板截取”
                    if len(img_paths) > self.max_images:
                        # 2026-10-31: 每次随访随机选取最多 max_images 张（原为等间隔采样）
                        indices = np.sort(self.rng.choice(len(img_paths), self.max_images, replace=False))
                        img_paths = [img_paths[i] for i in indices]
                        
                # 将记录加入样本列表（即使该次 visit 没有影像，也保留其时间、指标和诊断标签！）
                self.samples.append((img_paths, pid_str, date_str, sic, tab_vec))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_paths, pid, date, sic, tab_vec = self.samples[idx]

        imgs = []
        for p in img_paths:
            try:
                img = Image.open(p).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                imgs.append(img)
            except Exception as e:
                print(f"Warning: 读取图片失败并跳过 {p}: {e}")
                pass

        # 🚨【核心修改4】如果某次 visit 没有超声图片，返回全零 Tensor 进行补位
        if len(imgs) > 0:
            imgs = torch.stack(imgs)
        else:
            imgs = torch.zeros((1, 3, 224, 224), dtype=torch.float)

        return imgs, pid, date, torch.tensor(sic, dtype=torch.long), torch.tensor(tab_vec, dtype=torch.float)

# ======================
# Feature Extractor Model
# ======================
class VisitFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True, finetune=True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.cnn = resnet50(weights=weights)
        
        self.cnn.fc = nn.Identity()  # 移除最后的分类层，输出 2048 维向量
        
        if not finetune:
            for param in self.cnn.parameters():
                param.requires_grad = False

    def forward(self, x):
        # x: (num_images, 3, 224, 224)
        features = self.cnn(x)  # 输出: (num_images, 2048)
        return features


# ======================
# Main extraction logic
# ======================
def extract_and_save(root_dir, label_path, feature_save_path, finetune=True):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset = UltrasoundDataset(root_dir, label_path, transform=transform, max_images=DEFAULT_MAX_IMAGES)
    loader = DataLoader(dataset, batch_size=DEFAULT_BATCH_SIZE, shuffle=False)

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model = VisitFeatureExtractor(pretrained=True, finetune=finetune).to(device)
    model.eval()

    feats, tabular_feats, labels, patients, dates = [], [], [], [], []
    print(f"开始提取特征... 总有效 Visit 数量: {len(dataset)}")
    
    with torch.no_grad():
        for i, (imgs, pid, date, sic, tab_vec) in enumerate(loader):
            imgs = imgs.squeeze(0).to(device)
            feat = model(imgs).mean(dim=0)  # Visit Pooling (按单次就诊取平均)
            feats.append(feat.cpu().numpy())
            tabular_feats.append(tab_vec.squeeze(0).numpy())
            labels.append(int(sic.item()))
            patients.append(pid[0])
            dates.append(date[0])
            
            if (i + 1) % 100 == 0:
                print(f"已处理 {i + 1}/{len(dataset)} 个 Visit...")

    feats = np.stack(feats, axis=0)
    tabular_feats = np.stack(tabular_feats, axis=0)
    labels = np.array(labels)
    patients = np.array(patients)
    dates = np.array(dates)

    os.makedirs(os.path.dirname(feature_save_path), exist_ok=True)
    np.savez(
        feature_save_path,
        features=feats,
        tabular_feats=tabular_feats,
        labels=labels,
        patients=patients,
        dates=dates
    )
    print(f"特征提取完毕并保存至: {feature_save_path}")
    print(f"实际包含的 Visit 样本总数: {len(feats)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True, help="病人超声文件夹的根目录")
    parser.add_argument("--label_path", type=str, required=True, help="Excel 标签路径")
    parser.add_argument("--feature_save_path", type=str, required=True, help="特征保存路径 (.npz)")
    parser.add_argument("--no_finetune", action="store_true", help="是否冻结 ResNet 权重")
    args = parser.parse_args()

    extract_and_save(
        root_dir=args.root_dir,
        label_path=args.label_path,
        feature_save_path=args.feature_save_path,
        finetune=not args.no_finetune
    )
