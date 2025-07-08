import torch
from torch.utils.data import DataLoader
import yaml
from utils.dataloader import CheXpertDatasetHF
from transformers import AutoModelForImageClassification, AutoImageProcessor
from fairprune.prune_model import prune_model
from utils.metrics import evaluate
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Load HF pretrained model and processor
model = AutoModelForImageClassification.from_pretrained("codewithdark/vit-chest-xray").to(device)
processor = AutoImageProcessor.from_pretrained("codewithdark/vit-chest-xray")

# Dataset
val_dataset = CheXpertDatasetHF(
    csv_path=config['val_csv'],
    image_root=config['image_root'],
    processor=processor,
    target_col=config['target'],
    sensitive_col=config['sensitive_attr']
)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

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
