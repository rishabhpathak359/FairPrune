from utils.metrics import compute_group_accuracy
from utils.saliency import compute_saliency_with_alignment, compute_layer_fairness_impact, compute_layer_threshold_scalers
import numpy as np
import torch
from torch.nn.utils import prune
import copy

def prune_model_iterative(model, val_loader, worse_performing_group, criterion, device, p_ratio, lambda_align=0.1, num_iterations=5):
    print("\n🔁 Starting Iterative FairPrune.")
    total_ratio_per_iter = p_ratio / num_iterations
    pruned_model = copy.deepcopy(model)

    for iteration in range(num_iterations):
        print(f"\n🔂 Iteration {iteration + 1}/{num_iterations}")

        acc_odd, acc_even = compute_group_accuracy(pruned_model, val_loader, device, worse_performing_group)
        b_param = 1.0 + (acc_even - acc_odd)
        print(f"🔁 Adaptive β: {b_param:.4f} (acc_odd={acc_odd:.4f}, acc_even={acc_even:.4f})")

        saliency_diff = compute_saliency_with_alignment(pruned_model, criterion, device, val_loader, worse_performing_group, lambda_align)

        all_diff_flat = torch.cat([v.flatten() for v in saliency_diff.values()])
        kth = int(total_ratio_per_iter * all_diff_flat.numel())
        threshold = torch.kthvalue(all_diff_flat, kth)[0]
        print(f"🔻 Threshold: {threshold.item():.6f}")

        # Optional: Fairness calibration
        sensitivity = compute_layer_fairness_impact(pruned_model, val_loader, device, worse_performing_group)
        scale = compute_layer_threshold_scalers(sensitivity)

        for name, param in pruned_model.named_parameters():
            if name not in saliency_diff or not param.requires_grad:
                continue
            thresh = threshold * scale.get(name, 1.0)
            mask = saliency_diff[name] > thresh

            module_path = name.split('.')[:-1]
            param_name = name.split('.')[-1]
            module = pruned_model
            for attr in module_path:
                module = getattr(module, attr) if not attr.isdigit() else module[int(attr)]
            prune.custom_from_mask(module, name=param_name, mask=mask)

    print("✅ Iterative FairPrune Complete.")
    return pruned_model
