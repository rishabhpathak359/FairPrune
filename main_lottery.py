
import torch
import yaml
from transformers import AutoImageProcessor, AutoModelForImageClassification
from utils.dataloader import CheXpertDatasetHF
from torch.utils.data import DataLoader
from fairprune.lottery import replace_linear_with_masked
from fairprune.fairlottery import train_masked_model, binarize_and_prune

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("config.yaml") as f:
    config = yaml.safe_load(f)

processor = AutoImageProcessor.from_pretrained("codewithdark/vit-chest-xray")

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

# Step 1 & 2: Load model and replace Linear layers
from copy import deepcopy
model = AutoModelForImageClassification.from_pretrained("codewithdark/vit-chest-xray")
masked_model = deepcopy(model)
replace_linear_with_masked(masked_model)

# Step 3 & 4: Train with fairness-aware mask loss
train_masked_model(masked_model, val_loader, device, target_group=1)

# Step 5: Final pruning
binarize_and_prune(masked_model, threshold=0.5)

print("\n✅ Final pruned model is ready with fairness-aware masks.")
print("\n📊 Final Test Evaluation:")
test_metrics = evaluate(masked_model, test_loader, nn.CrossEntropyLoss(), device)
print(test_metrics)
