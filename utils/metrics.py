from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import torch


def evaluate(model, loader, criterion, device):
    model.eval()
    all_preds, all_labels, all_groups = [], [], []
    with torch.no_grad():
        for x, y, g in loader:
            x, y = x.to(device), y.to(device)

            # Hugging Face model expects keyword argument 'pixel_values'
            outputs = model(pixel_values=x)
            logits = outputs.logits

            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
            all_groups.append(g.cpu())

    return group_metrics(torch.cat(all_preds), torch.cat(all_labels), torch.cat(all_groups))

def group_metrics(preds, labels, groups):
    preds = preds.numpy()
    labels = labels.numpy()
    groups = groups.numpy()
    
    metrics = {}
    for g in np.unique(groups):
        idx = groups == g
        g_labels = labels[idx]
        g_preds = preds[idx]

        try:
            metrics[f'precision_group_{g}'] = precision_score(g_labels, g_preds, average='macro', zero_division=0)
            metrics[f'recall_group_{g}'] = recall_score(g_labels, g_preds, average='macro', zero_division=0)
            metrics[f'f1_group_{g}'] = f1_score(g_labels, g_preds, average='macro', zero_division=0)
        except ValueError:
            # This group may have only one class present — fallback to 0
            metrics[f'precision_group_{g}'] = 0
            metrics[f'recall_group_{g}'] = 0
            metrics[f'f1_group_{g}'] = 0

    if 'recall_group_0' in metrics and 'recall_group_1' in metrics:
        metrics['eopp'] = abs(metrics['recall_group_0'] - metrics['recall_group_1'])
    else:
        metrics['eopp'] = None

    return metrics