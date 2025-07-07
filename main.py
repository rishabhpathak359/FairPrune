import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml
from utils.dataloader import CheXpertDataset
from models.densenet import get_densenet
from fairprune.prune_model import prune_model
from utils.metrics import evaluate
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(model, loader, criterion, device):
    model.eval()
    all_preds, all_labels, all_groups = [], [], []
    with torch.no_grad():
        for x, y, g in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds)
            all_labels.append(y)
            all_groups.append(g)

    return group_metrics(torch.cat(all_preds), torch.cat(all_labels), torch.cat(all_groups))


with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Dataset
val_dataset = CheXpertDataset(
    config['val_csv'],
    config['image_root'],
    transform=transform,
    target_col=config['target'],
    sensitive_col=config['sensitive_attr']
)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

# Model
model = get_densenet(num_classes=2).to(device)
model.load_state_dict(torch.load(config['pretrained_model']))

# Loss
criterion = nn.CrossEntropyLoss()

# Evaluate before pruning
print("Before Pruning:")
metrics = evaluate(model, val_loader, criterion, device)
print(metrics)

# Run FairPrune
pruned_model = prune_model(
    model=model,
    val_loader=val_loader,
    worse_performing_group=1,
    criterion=criterion,
    device=device,
    p_ratio=config['prune_ratio'],
    b_param=config['beta']
)

# Evaluate after pruning
print("After Pruning:")
metrics = evaluate(pruned_model, val_loader, criterion, device)
print(metrics)
