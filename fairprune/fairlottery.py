
import torch
import torch.nn as nn
from fairprune.lottery import replace_linear_with_masked
from utils.metrics import evaluate

def fairness_loss(logits, labels, groups, target_group):
    group_ids = (groups == target_group)
    ce = nn.CrossEntropyLoss()
    loss_odd = ce(logits[group_ids], labels[group_ids]) if group_ids.any() else 0
    loss_even = ce(logits[~group_ids], labels[~group_ids]) if (~group_ids).any() else 0
    return torch.abs(loss_odd - loss_even)

def train_masked_model(model, dataloader, device, target_group, lr=1e-4, lambda_fair=1.0, lambda_sparse=1e-4, epochs=5):
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for images, labels, groups in dataloader:
            images, labels, groups = images.to(device), labels.to(device), groups.to(device)
            outputs = model(pixel_values=images)
            logits = outputs.logits
            cls_loss = ce(logits, labels)
            fair_penalty = fairness_loss(logits, labels, groups, target_group)
            l1_penalty = sum(torch.norm(m.mask, 1) for m in model.modules() if hasattr(m, 'mask'))

            loss = cls_loss + lambda_fair * fair_penalty + lambda_sparse * l1_penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}: Loss={loss.item():.4f} | Fairness={fair_penalty.item():.4f}")
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                val_metrics = evaluate(model, val_loader, ce, device)
                print(f"[Val Metrics] Epoch {epoch+1}:", val_metrics)
            model.train()

def binarize_and_prune(model, threshold=0.5):
    for module in model.modules():
        if hasattr(module, 'mask'):
            binary_mask = (module.mask.data > threshold).float()
            module.weight.data *= binary_mask
            del module.mask
