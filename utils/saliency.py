import torch
from utils.metrics import compute_group_accuracy
def compute_saliency_with_alignment(model, criterion, device, val_loader, worse_performing_group, lambda_align=0.1):
    print("📊 Computing saliency with gradient alignment.")
    model.eval()
    saliency_combined = {}
    for x, y, g in val_loader:
        x, y = x.to(device), y.to(device)
        odd_mask = g == worse_performing_group
        even_mask = ~odd_mask
        x_odd, y_odd = x[odd_mask], y[odd_mask]
        x_even, y_even = x[even_mask], y[even_mask]
        if len(x_odd) == 0 or len(x_even) == 0: continue

        grads_odd, grads_even = {}, {}
        loss_odd = criterion(model(pixel_values=x_odd).logits, y_odd).mean()
        grad_list = torch.autograd.grad(loss_odd, [p for p in model.parameters() if p.requires_grad], retain_graph=True)
        for n, g in zip([n for n, p in model.named_parameters() if p.requires_grad], grad_list): grads_odd[n] = g

        loss_even = criterion(model(pixel_values=x_even).logits, y_even).mean()
        grad_list = torch.autograd.grad(loss_even, [p for p in model.parameters() if p.requires_grad], retain_graph=True)
        for n, g in zip([n for n, p in model.named_parameters() if p.requires_grad], grad_list): grads_even[n] = g

        for name, param in model.named_parameters():
            if not param.requires_grad: continue
            g_odd, g_even = grads_odd[name], grads_even[name]
            alignment = torch.sum(g_odd * g_even) / (g_odd.norm() * g_even.norm() + 1e-8)
            s_odd = 0.5 * g_odd**2 * param**2
            s_even = 0.5 * g_even**2 * param**2
            saliency_combined[name] = s_odd + s_even + lambda_align * alignment
        break
    return saliency_combined

def compute_layer_fairness_impact(model, val_loader, device, target_group):
    print("🔍 Measuring layer-wise fairness sensitivity.")
    model.eval()
    base_odd, base_even = compute_group_accuracy(model, val_loader, device, target_group)
    base_gap = abs(base_even - base_odd)
    sensitivity = {}
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        with torch.no_grad():
            original = param.data.clone()
            param.data.zero_()
            acc_odd, acc_even = compute_group_accuracy(model, val_loader, device, target_group)
            param.data.copy_(original)
            gap = abs(acc_even - acc_odd)
            sensitivity[name] = gap - base_gap
    return sensitivity

def compute_layer_threshold_scalers(sensitivity_dict):
    values = list(sensitivity_dict.values())
    min_i, max_i = min(values), max(values)
    return {
        name: 1.5 - ((impact - min_i) / (max_i - min_i + 1e-8))
        for name, impact in sensitivity_dict.items()
    }
