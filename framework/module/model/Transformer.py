from typing import Union

import esm
import torch
import torch.nn as nn
import torch.nn.functional as F
from esm.modules import TransformerLayer, RobertaLMHead, ESM1bLayerNorm, gelu

from framework.module.model.Layer import MLP, GlobalAveragePooling, GlobalMaxPooling
from framework.utils.data.json_utils import json2list


class ESMModel(nn.Module):
    def __init__(
            self,
            model_name='esm2_t33_650M_UR50D',
            repr_layers=None,
            return_contacts=False,
            layer_fusion='mean',
            projection=None,
    ):
        super(ESMModel, self).__init__()

        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.num_layers = int(model_name.split('_')[1][1:])
        self.embed_dim = self.model.embed_dim
        self.repr_layers = [33] if repr_layers is None else json2list(repr_layers)
        self.return_contacts = return_contacts
        self.layer_fusion = layer_fusion  # ['none', 'mean', 'max', 'concat', 'attention']
        self.num_repr_layers = len(self.repr_layers)

        if self.layer_fusion == 'attention' and self.num_repr_layers > 1:
            self.layer_weights = nn.Parameter(torch.ones(self.num_repr_layers))

        self.projection = MLP(**projection) if projection else None

    def forward(self, tokens):
        esm_results = self.model(tokens, repr_layers=self.repr_layers, return_contacts=self.return_contacts)
        logits = esm_results['logits']  # (N, L, V)
        contacts = esm_results['contacts'] if self.return_contacts else None  # (N, L-2, L-2)
        representations = torch.stack([esm_results['representations'][l] for l in self.repr_layers], dim=0)  # (T, N, L, D)
        representations = self.layer_embedding_fusion(representations)
        results = {'logits': logits, 'contacts': contacts, 'representations': representations}
        return results

    def get_embedding_dim(self):
        if self.layer_fusion == 'concat':
            embedding_dim = self.num_repr_layers * self.embed_dim
        elif self.layer_fusion in ['mean', 'max', 'attention', 'none']:
            embedding_dim = self.embed_dim
        else:
            raise NotImplementedError
        return embedding_dim

    def layer_embedding_fusion(self, representations):
        # representations: (T, N, L, D)
        if self.projection:
            representations = self.projection(representations)

        if self.layer_fusion == 'none':
            esm_repr = representations
        elif self.layer_fusion == 'mean':
            esm_repr = representations.mean(dim=0)
        elif self.layer_fusion == 'max':
            esm_repr = representations.max(dim=0)[0]
        elif self.layer_fusion == 'concat':  # (T, N, L, D) -> (N, L, T*D)
            num_layer, batch_size, seq_len, feature_dim = representations.shape
            esm_repr = (representations.transpose(0, 1).transpose(1, 2)).reshape(batch_size, seq_len, num_layer * feature_dim)
        elif self.layer_fusion == 'attention':
            num_layer, batch_size, seq_len, feature_dim = representations.shape
            scores = F.softmax(self.layer_weights, dim=0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, batch_size, seq_len, feature_dim)
            esm_repr = (representations * scores).sum(dim=0)
        else:
            raise NotImplementedError
        return esm_repr


class ESMTransformer(nn.Module):
    def __init__(
            self,
            num_layers: int = 6,
            embed_dim: int = 1280,
            attention_heads: int = 20,
            alphabet: Union[esm.data.Alphabet, str] = "ESM-1b",
            token_dropout: bool = False,
            embedding_layer: bool = True,
            lm_head: bool = False,
            return_layer: int = -1,
    ):
        super(ESMTransformer, self).__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        if not isinstance(alphabet, esm.data.Alphabet):
            alphabet = esm.data.Alphabet.from_architecture(alphabet)
        self.alphabet = alphabet
        self.alphabet_size = len(alphabet)
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        self.prepend_bos = alphabet.prepend_bos
        self.append_eos = alphabet.append_eos
        self.token_dropout = token_dropout
        self.embedding_layer = embedding_layer
        self.lm_head = lm_head
        self.return_layer = num_layers if return_layer == -1 else return_layer

        self._init_submodules()

    def _init_submodules(self):
        # if self.token_dropout is None, it needn't embedding layer
        if self.embedding_layer:
            self.embed_scale = 1
            self.embed_tokens = nn.Embedding(
                self.alphabet_size,
                self.embed_dim,
                padding_idx=self.padding_idx,
            )

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    self.embed_dim,
                    4 * self.embed_dim,
                    self.attention_heads,
                    add_bias_kv=False,
                    use_esm1b_layer_norm=True,
                    use_rotary_embeddings=True,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.emb_layer_norm_after = ESM1bLayerNorm(self.embed_dim)

        if self.lm_head:
            self.lm_head = RobertaLMHead(
                embed_dim=self.embed_dim,
                output_dim=self.alphabet_size,
                weight=self.embed_tokens.weight,
            )

    def get_embedding_dim(self):
        return self.embed_dim

    def forward(self, tokens, padding_mask=None, repr_layers=True, need_head_weights=True):
        if self.embedding_layer:
            assert tokens.ndim == 2
            padding_mask = tokens.eq(self.padding_idx)  # B, T

            x = self.embed_scale * self.embed_tokens(tokens)

            if self.token_dropout:
                x.masked_fill_((tokens == self.mask_idx).unsqueeze(-1), 0.0)
                # x: B x T x C
                mask_ratio_train = 0.15 * 0.8
                src_lengths = (~padding_mask).sum(-1)
                mask_ratio_observed = (tokens == self.mask_idx).sum(-1).to(x.dtype) / src_lengths
                x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

            if padding_mask is not None:
                x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
        else:
            assert padding_mask is not None, 'Error: padding_mask should be not None!'
            x = tokens

        if self.embedding_layer and self.token_dropout:
            x.masked_fill_((tokens == self.mask_idx).unsqueeze(-1), 0.0)
            # x: B x T x C
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (tokens == self.mask_idx).sum(-1).to(x.dtype) / src_lengths
            x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

        if repr_layers:
            hidden_representations = {}
        if need_head_weights:
            attn_weights = []

        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)

        if not padding_mask.any():
            padding_mask = None

        for layer_idx, layer in enumerate(self.layers):
            x, attn = layer(
                x,
                self_attn_padding_mask=padding_mask,
                need_head_weights=need_head_weights,
            )
            if repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(0, 1)
            if need_head_weights:
                attn_weights.append(attn.transpose(1, 0))  # (H, B, T, T) => (B, H, T, T)

        x = self.emb_layer_norm_after(x)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

        # last hidden representation should have layer norm applied
        if repr_layers:
            hidden_representations[layer_idx + 1] = x
        result = {"representations": hidden_representations}

        if self.lm_head:
            x = self.lm_head(x)
            result["logits"] = x,

        if need_head_weights:
            attentions = torch.stack(attn_weights, 1)  # attentions: B x L x H x T x T
            if padding_mask is not None:
                attention_mask = 1 - padding_mask.type_as(attentions)
                attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
                attentions = attentions * attention_mask[:, None, None, :, :]
            result["attentions"] = attentions

        if self.return_layer:
            result['representations'] = result['representations'][self.return_layer]
        return result


class MultiheadAttentionLayer(nn.Module):
    def __init__(self, num_layers, embed_dim, attention_heads, ffn_embed_dim=None, cross_attn=True):
        super(MultiheadAttentionLayer, self).__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.ffn_embed_dim = ffn_embed_dim if ffn_embed_dim is not None else 4 * embed_dim
        self.cross_attn = cross_attn

        self._init_submodules()

    def _init_submodules(self):
        if self.cross_attn:
            self.query_layer_norm = ESM1bLayerNorm(self.embed_dim)
            self.key_layer_norm = ESM1bLayerNorm(self.embed_dim)
            self.value_layer_norm = ESM1bLayerNorm(self.embed_dim)
        else:
            self.layer_norm = ESM1bLayerNorm(self.embed_dim)

        self.attn = nn.MultiheadAttention(self.embed_dim, self.attention_heads, batch_first=True)
        self.fc1 = nn.Linear(self.embed_dim, self.ffn_embed_dim)
        self.fc2 = nn.Linear(self.ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = ESM1bLayerNorm(self.embed_dim)

    def forward(self, query, key, value):
        # query, key, value: B x T x C
        residual = query
        if self.cross_attn:
            query, key, value = self.query_layer_norm(query), self.key_layer_norm(key), self.value_layer_norm(value)
        else:
            query, key, value = self.layer_norm(query), self.layer_norm(key), self.layer_norm(value)

        x, attn = self.attn(key=key, value=value, query=query)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x
        return x, attn


class MultiheadInteractionLayer(nn.Module):
    def __init__(self, embed_dim, project_dim, num_heads, pooling=None):
        super(MultiheadInteractionLayer, self).__init__()
        assert project_dim % num_heads == 0, 'Error: project_dim should be divisible by num_heads!'
        self.embed_dim = embed_dim
        self.project_dim = project_dim
        self.num_heads = num_heads
        self.pooling = pooling

        self._init_submodules()

    def _init_submodules(self):
        self.proj_1 = nn.Linear(self.embed_dim, self.project_dim)
        self.proj_2 = nn.Linear(self.embed_dim, self.project_dim)
        if self.pooling is None:
            self.pooling = None
        elif self.pooling in ['mean', 'avg', 'average']:
            self.pooling = GlobalAveragePooling(dim=2, size=1)
        elif self.pooling in ['max', 'maximum']:
            self.pooling = GlobalMaxPooling(dim=2, size=1)
        else:
            raise NotImplementedError

    def forward(self, x1, x2):
        # x1, x2: B x T x C
        x1, x2 = self.proj_1(x1), self.proj_2(x2)  # B x T x C => B x T x P
        x1 = x1.reshape(x1.size(0), x1.size(1), self.num_heads, self.project_dim // self.num_heads)  # B x T x P => B x T x H x D
        x2 = x2.reshape(x2.size(0), x2.size(1), self.num_heads, self.project_dim // self.num_heads)  # B x T x P => B x T x H x D
        # matrix multiplication, (B x T x H x D) * (B x T x H x D) => (B x H x T x D) * (B x H x D x T) => (B x H x T x T) => (B x H, T x T)
        maps = torch.matmul(x1.transpose(1, 2), x2.transpose(1, 2).transpose(2, 3))  # B x H x T x T
        feats = self.pooling(maps) if self.pooling is not None else maps  # B x H x T x T => B x H
        return feats
