import torch
from torch.utils.data import DataLoader
import yaml
from utils.dataloader import CheXpertDatasetHF
from transformers import AutoModelForImageClassification, AutoImageProcessor
from fairprune.prune_model import prune_model_iterative
from utils.metrics import evaluate
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Load HF pretrained model and processor
model = AutoModelForImageClassification.from_pretrained("codewithdark/vit-chest-xray").to(device)
processor = AutoImageProcessor.from_pretrained("codewithdark/vit-chest-xray")

# Dataset
train_dataset = CheXpertDatasetHF(
    csv_path=config['train_csv'],
    image_root=config['image_root'],
    processor=processor,
    target_col=config['target'],
    sensitive_col=config['sensitive_attr']
)
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

val_dataset = CheXpertDatasetHF(
    csv_path=config['val_csv'],
    image_root=config['image_root'],
    processor=processor,
    target_col=config['target'],
    sensitive_col=config['sensitive_attr']
)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
# Load test set
test_dataset = CheXpertDatasetHF(
    csv_path=config['test_csv'],
    image_root=config['image_root'],
    processor=processor,
    target_col=config['target'],
    sensitive_col=config['sensitive_attr']
)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

criterion = nn.CrossEntropyLoss()

✅ Freeze all except classifier head
for param in model.parameters():
    param.requires_grad = False

for param in model.classifier.parameters():
    param.requires_grad = True

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=float(config.get('lr', 1e-4))
)
num_epochs = config.get('epochs', 3)

print(f"\n🔧 Training classifier for {num_epochs} epochs...\n")
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for x, y, _ in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(pixel_values=x)
        loss = criterion(outputs.logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

print("\n📊 Before Pruning:")
model.eval()
metrics = evaluate(model, val_loader, criterion, device)
print(metrics)

print("\n🔧 Running FairPrune...")
for param in model.parameters():
    param.requires_grad = True

pruned_model = prune_model_iterative(
    model=model,
    val_loader=val_loader,
    worse_performing_group=1,
    criterion=criterion,
    device=device,
    p_ratio=config['prune_ratio'],
    lambda_align=0.1,
    num_iterations=5
)

print("\n📊 After Pruning:")
metrics = evaluate(pruned_model, val_loader, criterion, device)
print(metrics)


print("\n📊 Final Test Evaluation:")
metrics = evaluate(pruned_model, test_loader, criterion, device)
print(metrics)

