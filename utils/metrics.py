from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

def group_metrics(preds, labels, groups):
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    groups = groups.cpu().numpy()
    
    metrics = {}
    for g in np.unique(groups):
        idx = groups == g
        metrics[f'precision_group_{g}'] = precision_score(labels[idx], preds[idx])
        metrics[f'recall_group_{g}'] = recall_score(labels[idx], preds[idx])
        metrics[f'f1_group_{g}'] = f1_score(labels[idx], preds[idx])
    
    # Fairness Metrics: Equal Opportunity
    metrics['eopp'] = abs(metrics['recall_group_0'] - metrics['recall_group_1'])
    return metrics
