import torch.nn as nn

from .Layer import Embedding, MLP


class SequenceMLP(nn.Module):
    def __init__(
            self,
            vocab_size,
            dim_embedding,
            max_seq_len,
            hiddens,
            activation=nn.ReLU,
            batch_norm=True,
            bias=True,
            dropout=None,
            custom_embedding=None,
            freeze_embedding=False,
            num_class=None,
    ):
        super(SequenceMLP, self).__init__()

        self.embedding = Embedding(vocab_size, dim_embedding, custom_embedding, freeze_embedding)
        hiddens = [max_seq_len * dim_embedding] + hiddens
        hiddens = hiddens + [num_class] if num_class is not None else hiddens
        self.mlp = MLP(hiddens, activation, bias, batch_norm, dropout)

    def forward(self, x):
        x = self.embedding(x)  # token_embeds: (B, L, D)
        x = x.reshape(x.size(0), -1)  # add a dimension: (B, L*D)
        logits = self.mlp(x)  # (B, num_class)
        return logits
