from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import torch

def evaluate(model, loader, criterion, device):
    model.eval()
    all_preds, all_labels, all_groups, all_probs = [], [], [], []

    with torch.no_grad():
        for x, y, g in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(pixel_values=x)
            logits = outputs.logits

            probs = torch.softmax(logits, dim=1)[:, 1]  # Class 1 probability
            preds = torch.argmax(logits, dim=1)

            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
            all_groups.append(g.cpu())

    return group_metrics(
        preds=torch.cat(all_preds),
        labels=torch.cat(all_labels),
        groups=torch.cat(all_groups),
        probs=torch.cat(all_probs)
    )

def group_metrics(preds, labels, groups, probs):
    preds = preds.numpy()
    labels = labels.numpy()
    groups = groups.numpy()
    probs = probs.numpy()

    metrics = {}

    # Overall AUC
    try:
        metrics['auc_overall'] = roc_auc_score(labels, probs)
    except:
        metrics['auc_overall'] = None

    # Group-wise metrics
    for g in np.unique(groups):
        idx = groups == g
        g_labels = labels[idx]
        g_preds = preds[idx]
        g_probs = probs[idx]

        try:
            metrics[f'precision_group_{g}'] = precision_score(g_labels, g_preds, average='macro', zero_division=0)
            metrics[f'recall_group_{g}'] = recall_score(g_labels, g_preds, average='macro', zero_division=0)
            metrics[f'f1_group_{g}'] = f1_score(g_labels, g_preds, average='macro', zero_division=0)
            metrics[f'auc_group_{g}'] = roc_auc_score(g_labels, g_probs)
        except:
            metrics[f'precision_group_{g}'] = 0
            metrics[f'recall_group_{g}'] = 0
            metrics[f'f1_group_{g}'] = 0
            metrics[f'auc_group_{g}'] = None

    # Fairness: Equal Opportunity Gap (TPR difference)
    if 'recall_group_0' in metrics and 'recall_group_1' in metrics:
        metrics['eopp'] = abs(metrics['recall_group_0'] - metrics['recall_group_1'])
    else:
        metrics['eopp'] = None

    return metrics
