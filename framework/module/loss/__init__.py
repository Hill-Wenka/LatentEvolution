import torch.nn as nn

from .BinaryClassificationLoss import FocalLoss, MutualInformationLoss, WeightedLoss
from .ContrastiveLoss import ContrastiveRankLoss, NTXentLoss, SimCSELoss, SupConLoss
from .MMDLoss import MMDLoss


def get_loss(loss_config, **kwargs):
    if loss_config.name == 'CrossEntropy':
        loss = nn.CrossEntropyLoss(**loss_config.args, **kwargs)
    elif loss_config.name == 'BCELoss':
        loss = nn.BCELoss(**loss_config.args, **kwargs)
    elif loss_config.name == 'BCEWithLogitsLoss':
        loss = nn.BCEWithLogitsLoss(**loss_config.args, **kwargs)
    elif loss_config.name == 'FocalLoss':
        loss = FocalLoss(**loss_config.args, **kwargs)
    elif loss_config.name == 'WeightedLoss':
        loss = WeightedLoss(**loss_config.args, **kwargs)
    elif loss_config.name == 'NTXentLoss':
        loss = NTXentLoss(**loss_config.args, **kwargs)
    elif loss_config.name == 'SupConLoss':
        loss = SupConLoss(**loss_config.args, **kwargs)
    elif loss_config.name == 'SimCSELoss':
        loss = SimCSELoss(**loss_config.args, **kwargs)
    elif loss_config.name == 'MutualInformationLoss':
        loss = MutualInformationLoss(**loss_config.args, **kwargs)
    elif loss_config.name == 'ContrastiveRankLoss':
        loss = ContrastiveRankLoss(**loss_config.args, **kwargs)
    elif loss_config.name == 'MSELoss':
        loss = nn.MSELoss(**loss_config.args, **kwargs)
    elif loss_config.name == 'NLLLoss':
        loss = nn.NLLLoss(**loss_config.args, **kwargs)
    elif loss_config.name == 'MMDLoss':
        loss = MMDLoss(**loss_config.args, **kwargs)
    else:
        raise RuntimeError(f'No such pre-defined loss: {loss_config.name}')
    return loss
