import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os

class CheXpertDataset(Dataset):
    def __init__(self, csv_path, image_root, transform=None, target_col='Cardiomegaly', sensitive_col='Sex'):
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.transform = transform
        self.target_col = target_col
        self.sensitive_col = sensitive_col
        self.df = self.df[self.df[target_col].notna() & self.df[sensitive_col].notna()]
        self.df[target_col] = self.df[target_col].astype(int)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_root, row['Path'])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(row[self.target_col], dtype=torch.long)
        group = 0 if row[self.sensitive_col] == 'Male' else 1
        return img, label, group
