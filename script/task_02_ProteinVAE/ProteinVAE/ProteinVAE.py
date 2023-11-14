import torch
import torch.nn as nn
from esm.modules import RobertaLMHead

from framework.module.model.Layer import MLP, get_hiddens
from framework.module.model.Transformer import ESMTransformer


class ProteinVAE(nn.Module):
    def __init__(self, encoder_transformer, encoder_mlp, decoder_mlp, decoder_transformer, regressor_head=None, reparameterization=True):
        super(ProteinVAE, self).__init__()

        self.encoder_transformer = ESMTransformer(**encoder_transformer)  # (N, L, V) -> (N, L, H)

        encoder_mlp.hiddens = get_hiddens(encoder_mlp.hiddens, input_dim=encoder_transformer['embed_dim'])
        self.encoder_mlp = MLP(**encoder_mlp)  # (N, L, H) -> (N, L, D)

        self.regressor_head = MLP(**regressor_head) if regressor_head is not None else None

        decoder_mlp.hiddens = get_hiddens(decoder_mlp.hiddens, input_dim=encoder_mlp.hiddens[-1] // 2, output_dim=decoder_transformer['embed_dim'])
        self.decoder_mlp = MLP(**decoder_mlp)  # (N, L, D/2) -> (N, L, H)

        self.decoder_transformer = ESMTransformer(**decoder_transformer)  # (N, L, H) -> (N, L, H)

        # self.lm_head_weight = self.encoder_transformer.embed_tokens.weight # initialize with encoder_transformer embed_tokens weight
        self.lm_head_weight = nn.Parameter(torch.randn(size=self.encoder_transformer.embed_tokens.weight.shape))  # initialize randomly
        self.lm_head = RobertaLMHead(embed_dim=decoder_transformer['embed_dim'],
                                     output_dim=len(self.encoder_transformer.alphabet),
                                     weight=self.lm_head_weight)  # (N, L, H) -> (N, L, V)

        self.reparameterization = reparameterization
        self.reparameterize = self.reparameterize if self.reparameterization else lambda x, y: x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # (N, L*D/2)
        eps = torch.randn_like(std)  # (N, L*D/2)
        z = mu + eps * std  # (N, L*D/2)
        return z

    def encode(self, x):
        position_h = self.encoder_transformer(x)['representations']  # (N, L, H)
        position_h = self.encoder_mlp(position_h)  # (N, L, H) -> (N, L, D)
        concat_h = position_h.flatten(1)  # (N, L, D) -> (N, L*D)
        mean, logvar = torch.chunk(concat_h, 2, dim=-1)  # (N, L*D) -> (N, L*D/2), (N, L*D/2)
        z = self.reparameterize(mean, logvar)  # (N, Z) = (N, L*D/2)
        return z, mean, logvar, position_h

    def decode(self, z):
        position_h = z.reshape(z.shape[0], -1, self.decoder_mlp.hiddens[0])  # (N, L*D/2) -> (N, L, D/2)
        position_h = self.decoder_mlp(position_h)  # (N, L, D/2) -> (N, L, H)
        padding_mask = torch.zeros((position_h.shape[0], position_h.shape[1]), dtype=torch.bool, device=position_h.device)  # (N, L)
        position_h = self.decoder_transformer(position_h, padding_mask=padding_mask)['representations']  # (N, L, H)
        logits = self.lm_head(position_h)  # (N, L, V)
        return logits

    def forward(self, x):
        z, mean, logvar, position_h = self.encode(x)
        if self.regressor_head is not None:
            # preds = self.regressor_head(position_h[:, 0, :])
            preds = self.regressor_head(mean)
            pred_ddG, pred_dS = torch.tanh(preds[:, 0]), torch.tanh(preds[:, 1])  # range: [-1, 1]
        else:
            pred_ddG, pred_dS = None, None
        logits = self.decode(z)
        return logits, mean, logvar, pred_ddG, pred_dS
