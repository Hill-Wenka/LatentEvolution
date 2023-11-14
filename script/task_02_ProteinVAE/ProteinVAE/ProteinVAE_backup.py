import esm
import torch
import torch.nn as nn
from esm.modules import RobertaLMHead

from framework.module.model.Layer import MLP, get_hiddens
from framework.module.model.Transformer import ESMTransformer


class ProteinVAE_v1(nn.Module):
    def __init__(self, encoder_params, decoder_params, alphabet='ESM-1b'):
        super(ProteinVAE_v1, self).__init__()

        if not isinstance(alphabet, esm.data.Alphabet):
            alphabet = esm.data.Alphabet.from_architecture(alphabet)
        self.embed_tokens = nn.Embedding(
            num_embeddings=len(alphabet),
            embedding_dim=encoder_params['d_model'],
            padding_idx=alphabet.padding_idx,
        )
        self.lm_head = RobertaLMHead(embed_dim=decoder_params['d_model'],
                                     output_dim=len(alphabet),
                                     weight=self.embed_tokens.weight)

        self.encoder_transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=encoder_params['d_model'],
                nhead=encoder_params['nhead'],
                dim_feedforward=encoder_params['dim_feedforward'],
                norm_first=encoder_params['norm_first'],
                batch_first=encoder_params['batch_first'],
                activation=encoder_params['activation'],
                dropout=encoder_params['dropout'],
            ),
            num_layers=encoder_params['num_layers'],
        )

        # 与T5不同，T5的decoder是TransformerDecoder，而不是TransformerEncoder
        self.decoder_transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=decoder_params['d_model'],
                nhead=decoder_params['nhead'],
                dim_feedforward=decoder_params['dim_feedforward'],
                norm_first=decoder_params['norm_first'],
                batch_first=decoder_params['batch_first'],
                activation=decoder_params['activation'],
                dropout=decoder_params['dropout'],
            ),
            num_layers=decoder_params['num_layers'],
        )

        # 与T5相同，需要encoder的hiddens作为decoder的输入
        # self.decoder_transformer = nn.TransformerDecoder(
        #     decoder_layer=nn.TransformerDecoderLayer(
        #         d_model=decoder_params['d_model'],
        #         nhead=decoder_params['nhead'],
        #         dim_feedforward=decoder_params['dim_feedforward'],
        #         norm_first=decoder_params['norm_first'],
        #         batch_first=decoder_params['batch_first'],
        #         activation=decoder_params['activation'],
        #         dropout=decoder_params['dropout'],
        #     ),
        #     num_layers=decoder_params['num_layers'],
        # )

        encoder_mlp_params = get_hiddens(encoder_params['mlp'], input_dim=encoder_params['d_model'])
        self.encoder_mlp = MLP(encoder_mlp_params)

        decoder_mlp_params = get_hiddens(decoder_params['mlp'], output_dim=decoder_params['d_model'])
        self.decoder_mlp = MLP(decoder_mlp_params)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # (N, L*D/2)
        eps = torch.randn_like(std)  # (N, L*D/2)
        z = mu + eps * std  # (N, L*D/2)
        return z

    def encode(self, x):
        x = self.embed_tokens(x)  # (N, L, V) -> (N, L, H)
        encoder_hiddens = self.encoder_transformer(x)  # (N, L, H)
        h = self.encoder_mlp(encoder_hiddens)  # (N, L, H) -> (N, L, D)
        h = h.flatten(1)  # (N, L, D) -> (N, L*D)
        mean, logvar = torch.chunk(h, 2, dim=-1)  # (N, L*D) -> (N, L*D/2), (N, L*D/2)
        z = self.reparameterize(mean, logvar)  # (N, Z) = (N, L*D/2)
        return z, mean, logvar

    def decoder(self, z):
        h = self.decoder_mlp(z)  # (N, L*D/2) -> (N, L*H)
        h = h.reshape(h.shape[0], -1, self.encoder_mlp.hiddens[0])  # (N, L*H) -> (N, L, H)
        h = self.decoder_transformer(h)  # (N, L, H)
        logits = self.lm_head(h)  # (N, L, H) -> (N, L, V)
        return logits

    def forward(self, x):
        z, mean, logvar = self.encode(x)
        logits = self.decoder(z)
        return logits, mean, logvar


class ProteinVAE_v2(nn.Module):
    def __init__(self, encoder_params, encoder_mlp, encoder_mapping, decoder_mlp, decoder_mapping, decoder_params):
        super(ProteinVAE_v2, self).__init__()

        self.encoder_transformer = ESMTransformer(**encoder_params)  # (N, L, V) -> (N, L, H)

        encoder_mlp.hiddens = get_hiddens(encoder_mlp.hiddens, input_dim=encoder_params['embed_dim'])
        self.encoder_mlp = MLP(**encoder_mlp)  # (N, L, H) -> (N, L, D)

        self.encoder_mapping = MLP(**encoder_mapping)  # (N, L*D) -> (N, 2*Z)

        decoder_mapping.hiddens = get_hiddens(decoder_mapping.hiddens, input_dim=encoder_mapping.hiddens[-1] // 2)
        self.decoder_mapping = MLP(**decoder_mapping)  # (N, Z) -> (N, L*D)

        decoder_mlp.hiddens = get_hiddens(decoder_mlp.hiddens, input_dim=encoder_mlp.hiddens[-1], output_dim=decoder_params['embed_dim'])
        self.decoder_mlp = MLP(**decoder_mlp)  # (N, L, D) -> (N, L, H)

        self.decoder_transformer = ESMTransformer(**decoder_params)  # (N, L, H) -> (N, L, H)

        self.lm_head = RobertaLMHead(embed_dim=decoder_params['embed_dim'],
                                     output_dim=len(self.encoder_transformer.alphabet),
                                     weight=self.encoder_transformer.embed_tokens.weight)  # (N, L, H) -> (N, L, V)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # (N, L*D/2)
        eps = torch.randn_like(std)  # (N, L*D/2)
        z = mu + eps * std  # (N, L*D/2)
        return z

    def encode(self, x):
        position_h = self.encoder_transformer(x)['representations']  # (N, L, H)
        position_h = self.encoder_mlp(position_h)  # (N, L, H) -> (N, L, D)
        concat_h = position_h.flatten(1)  # (N, L, D) -> (N, L*D)
        latent = self.encoder_mapping(concat_h)  # (N, L*D) -> (N, 2*Z)
        mean, logvar = torch.chunk(latent, 2, dim=-1)  # (N, 2*Z) -> (N, Z), (N, Z)
        z = self.reparameterize(mean, logvar)  # (N, Z)
        return z, mean, logvar

    def decoder(self, z):
        concat_h = self.decoder_mapping(z)  # (N, Z) -> (N, L*D)
        position_h = concat_h.reshape(concat_h.shape[0], -1, self.decoder_mlp.hiddens[0])  # (N, L*D) -> (N, L, D)
        position_h = self.decoder_mlp(position_h)  # (N, L, D) -> (N, L, H)
        padding_mask = torch.zeros((position_h.shape[0], position_h.shape[1]), dtype=torch.bool, device=position_h.device)  # (N, L)
        position_h = self.decoder_transformer(position_h, padding_mask=padding_mask)['representations']  # (N, L, H)
        logits = self.lm_head(position_h)  # (N, L, V)
        return logits

    def forward(self, x):
        z, mean, logvar = self.encode(x)
        logits = self.decoder(z)
        return logits, mean, logvar


class ProteinVAE_v3(nn.Module):
    def __init__(self, encoder_transformer, encoder_mlp, decoder_mlp, decoder_transformer, regressor_head=None):
        super(ProteinVAE_v3, self).__init__()

        self.encoder_transformer = ESMTransformer(**encoder_transformer)  # (N, L, V) -> (N, L, H)

        encoder_mlp.hiddens = get_hiddens(encoder_mlp.hiddens, input_dim=encoder_transformer['embed_dim'])
        self.encoder_mlp = MLP(**encoder_mlp)  # (N, L, H) -> (N, L, D)

        self.regressor_head = MLP(**regressor_head) if regressor_head is not None else None

        decoder_mlp.hiddens = get_hiddens(decoder_mlp.hiddens, input_dim=encoder_mlp.hiddens[-1] // 2, output_dim=decoder_transformer['embed_dim'])
        self.decoder_mlp = MLP(**decoder_mlp)  # (N, L, D/2) -> (N, L, H)

        self.decoder_transformer = ESMTransformer(**decoder_transformer)  # (N, L, H) -> (N, L, H)

        self.lm_head = RobertaLMHead(embed_dim=decoder_transformer['embed_dim'],
                                     output_dim=len(self.encoder_transformer.alphabet),
                                     weight=self.encoder_transformer.embed_tokens.weight)  # (N, L, H) -> (N, L, V)

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
        return z, mean, logvar

    def decode(self, z):
        position_h = z.reshape(z.shape[0], -1, self.decoder_mlp.hiddens[0])  # (N, L*D/2) -> (N, L, D/2)
        position_h = self.decoder_mlp(position_h)  # (N, L, D/2) -> (N, L, H)
        padding_mask = torch.zeros((position_h.shape[0], position_h.shape[1]), dtype=torch.bool, device=position_h.device)  # (N, L)
        position_h = self.decoder_transformer(position_h, padding_mask=padding_mask)['representations']  # (N, L, H)
        logits = self.lm_head(position_h)  # (N, L, V)
        return logits

    def forward(self, x):
        z, mean, logvar = self.encode(x)
        if self.regressor_head is not None:
            preds = self.regressor_head(mean)
            pred_ddG, pred_dS = preds[:, 0], preds[:, 1]
        else:
            pred_ddG, pred_dS = None, None
        logits = self.decode(z)
        return logits, mean, logvar, pred_ddG, pred_dS
