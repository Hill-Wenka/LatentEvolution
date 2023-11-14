import numpy as np
import torch
from torch_geometric.data import Data

from framework.data.Tokenizer import Tokenizer
from framework.utils.bio.struct_utils import construct_graph


class ProteinGraph:
    def __init__(self, node_feature=None, edge_index=None, edge_attr=None, node_label=None, **kwargs):
        self.node_feature = node_feature
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.node_label = node_label
        self.tokenizer = Tokenizer(**kwargs)

    def construct_graph(self, sequence, structure, node_label=None, feature_dict=None, node_feature=None, one_hot=True, edge_threshold=8, **kwargs):
        node, edge, distance = construct_graph(structure, edge_threshold)
        node_feature = self._get_node_features(sequence, feature_dict, node_feature, one_hot)
        edge_index = self.get_edge_index(edge)
        edge_attr = self._get_edge_attr(structure, distance, edge_index)
        node_label = self._get_label(node_label)
        if node_label is not None:
            assert len(node_label) == len(node_feature), 'node_label must have the same length as node_feature'
        self.node_feature = node_feature
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.node_label = node_label
        return {'node_feature': node_feature, 'edge_index': edge_index, 'edge_attr': edge_attr, 'node_label': node_label}

    def get_graph(self):
        # construct graph data for pytorch geometric
        return Data(x=self.node_feature, edge_index=self.edge_index, edge_attr=self.edge_attr, y=self.node_label)

    def _get_node_features(self, sequence, feature_dict, node_feature=None, one_hot=True):
        if node_feature is None:
            node_feature = []
        else:
            if node_feature == 'all':
                node_feature = feature_dict.keys()
            elif isinstance(node_feature, str):
                node_feature = [node_feature]
            else:
                assert isinstance(node_feature, list), 'node_feature_keys must be a list'

        if len(node_feature) > 0:
            node_feature = np.concatenate([feature_dict[key].reshape(len(sequence), -1) for key in node_feature], axis=-1)  # [seq_len, feature_dim]
            if one_hot:
                onehot_feature = self.tokenizer.one_hot(sequence, padding=False)[0]  # [seq_len, 20]
                node_feature = torch.cat([onehot_feature, torch.from_numpy(node_feature)], dim=-1).float()  # [seq_len, feature_dim + 20]
            else:
                node_feature = torch.from_numpy(node_feature).float()  # [seq_len, feature_dim]
        else:
            assert one_hot, 'feature must be provided if one_hot is False'
            node_feature = self.tokenizer.one_hot(sequence, padding=False)[0]  # [seq_len, 20]
        return node_feature.numpy()  # [seq_len, num_node_features]

    def get_edge_index(self, edge):
        # convert edge to edge_index
        edge_index = np.argwhere(edge == 1).transpose()  # [2, num_edges]
        return edge_index  # [2, num_edges]

    def _get_edge_attr(self, structure, distance, edge_index):
        # get edge_attr from edge
        edge_attr = distance[edge_index[0], edge_index[1]].reshape(-1, 1)  # [num_edges, num_edge_features]
        return edge_attr  # [num_edges, num_edge_features]

    def _get_label(self, node_label):
        # get node label for each node
        node_label = np.array(node_label) if node_label is not None else None  # [num_nodes]
        return node_label  # [num_nodes]
