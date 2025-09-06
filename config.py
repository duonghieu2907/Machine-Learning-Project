from ResNet import *

from losses import *
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# model 정의
MODEL_DICT = {
    "resnet18": resnet18,
    "resnet18Beta": resnet18Beta,
    "resnet18Gamma": resnet18Gamma
}

CRITERION_DICT = {
    "CrossEntropyLoss": nn.CrossEntropyLoss,
    "LabelSmoothingLoss": LabelSmoothingLoss
}

# optimizer 정의
OPTIMIZER_DICT = {
    "SGD": optim.SGD,
    "Adam": optim.Adam,
    "AdamW": optim.AdamW
}

SCHEDULER_DICT = {
    "OneCycleLR": lambda optimizer: lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.1,
        total_steps=100
    )
}
