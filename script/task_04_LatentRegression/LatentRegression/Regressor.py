import torch.nn as nn

from framework.module.model.Layer import MLP


class Net(nn.Module):
    def __init__(self, mlp_params):
        super(Net, self).__init__()
        self.mlp = MLP(**mlp_params)

    def forward(self, x):
        preds = self.mlp(x)
        pred_ddG, pred_dS = preds[:, 0], preds[:, 1]
        return pred_ddG, pred_dS
