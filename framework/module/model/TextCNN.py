import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.module.model import activations
from framework.module.model.Layer import Embedding, MLP
from framework.utils.data.json_utils import params2list


class TextCNN(nn.Module):
    def __init__(
            self,
            vocab_size,
            dim_embedding,
            filter_size_list,
            filter_num_list,
            conv_activation,
            conv_pooling,
            hiddens,
            activation,
            bias,
            batch_norm,
            dropout,
            custom_embedding=None,
            freeze_embedding=False,
            num_class=None,
    ):
        super(TextCNN, self).__init__()

        filter_size_list, filter_num_list, hiddens = params2list(filter_size_list, filter_num_list, hiddens)
        if len(filter_num_list) == 1:
            filter_num_list = filter_num_list * len(filter_size_list)

        self.embedding = Embedding(vocab_size, dim_embedding, custom_embedding, freeze_embedding)
        self.convs = nn.ModuleList([nn.Conv2d(1, filter_num_list[i], (filter_size, dim_embedding)) for i, filter_size in enumerate(filter_size_list)])
        self.conv_activation = activations[conv_activation]()
        self.conv_pooling = F.max_pool2d if conv_pooling == 'max' else F.avg_pool2d
        if hiddens[0] == -1:
            hiddens[0] = sum(filter_num_list)
        if num_class is not None:
            hiddens.append(num_class)
        self.head = MLP(hiddens, activation, bias, batch_norm, dropout)

    def add_feature(self, x, feature):
        # embedding: (N, L, D), feature: (N, L, D)
        # concat the token embedding and additional feature if necessary
        if feature is not None:
            x = torch.cat([x, feature], dim=-1)
        return x

    def forward(self, tokens, feature=None):
        # tokens: (N, L), feature: (N, L, d)
        x = self.embedding(tokens)  # (N, L, D)
        x = self.add_feature(x, feature)
        x = x.unsqueeze(1)  # (N, 1, L, D+d)
        channel_conv_features = [self.conv_activation(conv(x)) for conv in self.convs]  # [len(filter_sizes)]: (N, num_filter, L_after_conv, 1)
        channel_pooling_features = [self.conv_pooling(input=x_item, kernel_size=(x_item.size(-2), x_item.size(-1))).view(x_item.size(0), -1) for
                                    x_item in channel_conv_features]  # [len(filter_sizes)]: (N, num_filter, 1)
        conv_features = torch.cat(channel_pooling_features, 1)  # (N, sum(filter_num_list))
        logits = self.head(conv_features)  # (N, num_class)
        return logits
