from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torch
import os

class CheXpertDatasetHF(Dataset):
    def __init__(self, csv_path, image_root, processor, target_col='Cardiomegaly', sensitive_col='Sex'):
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.processor = processor
        self.target_col = target_col
        self.sensitive_col = sensitive_col

        # Filter out missing or uncertain labels
        self.df = self.df[self.df[target_col].notna() & self.df[sensitive_col].notna()]
        self.df[target_col] = self.df[target_col].replace(-1, 0).astype(int)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_root, row['Path']).replace('\\', '/')
        image = Image.open(img_path).convert('RGB')

        # Hugging Face ViT preprocessing
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)  # [3, 224, 224]

        label = torch.tensor(row[self.target_col], dtype=torch.long)
        group = 0 if row[self.sensitive_col] == 'Male' else 1

        return pixel_values, label, group
