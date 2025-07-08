import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
from utils.dataloader import CheXpertDataset, vit_transforms
from models.vit_model import get_vit_b

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = CheXpertDataset(
        csv_path='data/archive(1)/train.csv',
        image_root='data/archive(1)/',
        transform=vit_transforms,
        target_col='Cardiomegaly',
        sensitive_col='Sex'
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    model = get_vit_b(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-5)

    for epoch in range(5):  # Train for 5 epochs
        model.train()
        epoch_loss = 0.0
        for x, y, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(train_loader)}")

    torch.save(model.state_dict(), "models/vit_b_chexpert.pth")


if __name__ == "__main__":
    train()
