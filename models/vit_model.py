import timm
import torch.nn as nn

def get_vit_b(num_classes=2):
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model
