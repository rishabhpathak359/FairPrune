
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.mask = nn.Parameter(torch.ones_like(self.weight))

    def forward(self, input):
        masked_weight = self.weight * torch.clamp(self.mask, 0, 1)
        return F.linear(input, masked_weight, self.bias)

def replace_linear_with_masked(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            masked_layer = MaskedLinear(child.in_features, child.out_features, child.bias is not None)
            masked_layer.weight.data = child.weight.data.clone()
            if child.bias is not None:
                masked_layer.bias.data = child.bias.data.clone()
            setattr(module, name, masked_layer)
        else:
            replace_linear_with_masked(child)
