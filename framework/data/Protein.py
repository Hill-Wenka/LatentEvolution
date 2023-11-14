from framework.utils.bio.seq_utils import mutate
from framework.utils.bio.struct_utils import compute_sasa
from framework.utils.bio.struct_utils import get_atom_array
from .ProteinGraph import ProteinGraph


class Protein:
    def __init__(self, index=None, name=None, sequence=None, structure=None, partition=None, features=None, attributes=None, **kwargs):
        self.index = str(index)  # str
        self.name = str(name)  # strs
        self.sequence = sequence  # str
        self.structure = structure  # str or AtomArray
        self.length = len(sequence) if sequence is not None else None  # int
        self.graph_data = None  # dict
        self.protein_graph = ProteinGraph(**kwargs)  # ProteinGraph object
        self.partition = partition  # str, 'train', 'val', 'test'

        self.features = features if features is not None else {}
        self.attributes = attributes if attributes is not None else {}
        for key, value in self.features.items():
            setattr(self, key, value)
        for key, value in self.attributes.items():
            setattr(self, key, value)

    @property
    def data(self):
        if self.structure is not None:
            structure = self.structure if isinstance(self.structure, str) else True
        else:
            structure = None
        features = {k: True if v is not None else None for k, v in self.features.items()}

        return {
            'index': self.index,
            'name': self.name,
            'partition': self.partition,
            'length': self.length,
            'sequence': self.sequence,
            'structure': structure,
            'graph': True if self.graph_data is not None else None,
            **features,
            **self.attributes
        }

    @property
    def graph(self):
        assert self.structure is not None
        if self.graph_data is None:
            print('Warning: construct graph with default parameters. Please use construct_graph() to construct graph with customized parameters.')
            self.graph_data = self.protein_graph.construct_graph(self.sequence, self.structure)
        else:
            if self.protein_graph.node_feature is None:
                self.protein_graph.node_feature = self.graph_data['node_feature']
                self.protein_graph.edge_index = self.graph_data['edge_index']
                self.protein_graph.edge_attr = self.graph_data['edge_attr']
                self.protein_graph.node_label = self.graph_data['node_label']
        return self.protein_graph.get_graph()

    def construct_graph(self, **kwargs):
        assert self.sequence is not None and self.structure is not None
        self.structure = get_atom_array(self.structure) if isinstance(self.structure, str) else self.structure
        self.graph_data = self.protein_graph.construct_graph(self.sequence, self.structure, **kwargs)

    def compute_sasa(self, **kwargs):
        assert self.sequence is not None and self.structure is not None
        self.structure = get_atom_array(self.structure) if isinstance(self.structure, str) else self.structure
        sasa = compute_sasa(self.structure, **kwargs)  # (seq_len,)
        self.set_feature('sasa', sasa)

    def mutate(self, positions, mutations):
        return mutate(self.sequence, positions, mutations)

    def get_feature(self, key):
        return self.features[key] if key in self.features else None

    def set_feature(self, key, value):
        setattr(self, key, value)
        self.features[key] = value

    def get_attribute(self, key):
        return self.attributes[key] if key in self.attributes else None

    def set_attribute(self, key, value):
        setattr(self, key, value)
        self.attributes[key] = value

    def __str__(self):
        # 定义的输出格式，index最长为16个字符，不足则用空格补齐，超过则截断
        index_len, max_len, = min(len(self.index) + 2, 40), 40
        attr = ', '.join(f'{k}={v:.2f}' if isinstance(v, float) else f'{k}={v}' for i, (k, v) in enumerate(self.attributes.items()) if i < 1)
        attr += ', ...' if len(self.attributes) > 1 else ''
        if len(self.index) < index_len:
            pad_left = (index_len - len(self.index)) // 2
            pad_right = index_len - len(self.index) - pad_left
            index = ' ' * pad_left + self.index + ' ' * pad_right
        else:
            index = self.index[:index_len - 3] + '...'
        sequence = self.sequence[:max_len // 2] + '......' + self.sequence[-max_len // 2:] if len(self.sequence) > max_len else self.sequence
        return f'Protein([{index}]\t\t{sequence}\t\t({attr}))'

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.sequence)
