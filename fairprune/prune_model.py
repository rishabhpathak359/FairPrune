import numpy as np
import torch
from torch.nn.utils import prune
import copy

def prune_model(model, val_loader, worse_performing_group, criterion, device, p_ratio, b_param):
    print("\n🔧 Starting FairPrune...\n")
    print(f"Pruning ratio: {p_ratio}, β (b_param): {b_param}")
    print(f"Worse performing group: {worse_performing_group}")
    
    layer_wise_saliency_odd = {}
    layer_wise_saliency_even = {}
    seen_odd = False
    seen_even = False

    assert worse_performing_group is not None

    for i, (images, labels, groups) in enumerate(val_loader):
        images = images.to(device)
        labels = labels.to(device)

        if isinstance(worse_performing_group, int):
            odd_group_indices = groups == worse_performing_group
            even_group_indices = groups != worse_performing_group
        elif isinstance(worse_performing_group, str):
            odd_group_indices = np.array(groups) == worse_performing_group
            even_group_indices = np.array(groups) != worse_performing_group

        odd_images = images[odd_group_indices]
        odd_labels = labels[odd_group_indices]
        even_images = images[even_group_indices]
        even_labels = labels[even_group_indices]

        if len(odd_images) > 0 and not seen_odd:
            print(f"📦 Computing saliencies for worse group (odd), batch {i}")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    try:
                        outputs = model(pixel_values=odd_images)
                        logits = outputs.logits
                        loss = criterion(logits, odd_labels).mean()
                        grads = torch.autograd.grad(loss, param, retain_graph=True)
                    except Exception as e:
                        print(f"[ERROR][ODD] Grad failed on {name}: {e}")
                        continue
                    saliency = 0.5 * grads[0] ** 2 * param ** 2
                    layer_wise_saliency_odd.setdefault(name, []).append(saliency)
            seen_odd = True

        if len(even_images) > 0 and not seen_even:
            print(f"📦 Computing saliencies for better group (even), batch {i}")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    try:
                        outputs = model(pixel_values=even_images)
                        logits = outputs.logits
                        loss = criterion(logits, even_labels).mean()
                        grads = torch.autograd.grad(loss, param, retain_graph=True)
                    except Exception as e:
                        print(f"[ERROR][EVEN] Grad failed on {name}: {e}")
                        continue
                    saliency = 0.5 * grads[0] ** 2 * param ** 2
                    layer_wise_saliency_even.setdefault(name, []).append(saliency)
            seen_even = True

        if seen_odd and seen_even:
            print("✅ Collected saliencies from both groups.\n")
            break

    print("📊 Averaging saliencies across batches...")
    layer_wise_saliency_odd = {k: torch.mean(torch.stack(v), dim=0) for k, v in layer_wise_saliency_odd.items()}
    layer_wise_saliency_even = {k: torch.mean(torch.stack(v), dim=0) for k, v in layer_wise_saliency_even.items()}

    saliency_differences = {}
    for name in layer_wise_saliency_odd:
        saliency_differences[name] = layer_wise_saliency_odd[name] - b_param * layer_wise_saliency_even[name]

    print("📈 Calculating pruning threshold...")
    all_saliency_differences = torch.cat([v.flatten() for v in saliency_differences.values()])
    threshold = torch.kthvalue(all_saliency_differences, int(p_ratio * len(all_saliency_differences)))[0]
    print(f"🔻 Saliency threshold set at: {threshold.item():.6f}")

    pruning_mask = {
        name: diff > threshold for name, diff in saliency_differences.items()
    }

    print("✂️  Applying pruning masks to model...\n")
    pruned_model = copy.deepcopy(model)
    pruned_params = 0
    total_params = 0

    for name, param in model.named_parameters():
        if param.requires_grad and name in pruning_mask:
            mask = pruning_mask[name]
            pruned_params += (~mask).sum().item()
            total_params += mask.numel()
            # Resolve the module
            name_parts = ['pruned_model'] + name.split('.')
            name_first = '.'.join(name_parts[:-1])
            name_last = name_parts[-1]
            for e in range(12):  # handle layers like layer.0, layer.1...
                name_first = name_first.replace(f'.{e}.', f'[{e}].')
            for e in range(0, 10):
                name_first = name_first.replace(f'.{e}', f'[{e}]')
            module_part = eval(name_first)
            prune.custom_from_mask(module_part, name=name_last, mask=mask)

    print(f"\n✅ Pruning complete!")
    print(f"🧮 Total pruned params: {pruned_params:,}")
    print(f"📉 Pruned ratio: {pruned_params / (total_params or 1):.4f}\n")

    return pruned_model
