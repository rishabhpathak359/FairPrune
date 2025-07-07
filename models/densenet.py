from torchvision.models import densenet121
import torch.nn as nn

def get_densenet(num_classes=2):
    model = densenet121(pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model
