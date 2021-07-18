import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
# Loading the data

def resnet18(pretrained=True):
    model = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'],model_dir='~/scratch/'))
    return model

model = resnet18(pretrained=True)

# If you just need to fine-tune the last layer, comment out the code below.
# for param in model.parameters():
#     param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 100)
