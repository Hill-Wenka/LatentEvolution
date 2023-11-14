from framework.utils.bio.bioinfo import scFv_linker
from framework.utils.data.json_utils import obj2str
from .Protein import Protein


class Antibody(Protein):
    def __init__(self, index=None, name=None, scFv=None, VH=None, VL=None, linker=None, structure=None, partition=None, features=None,
                 attributes=None):
        linker = linker if linker is not None else scFv_linker
        if scFv is not None:
            self.scFv = scFv
            split_VH, split_VL = scFv.split(linker)
            if VL is None or VH is None:
                self.VH, self.VL = split_VH, split_VL
            else:
                assert VL == split_VL and VH == split_VH, 'VL and VH do not match scFv'
                self.VH, self.VL = VH, VL
        elif VH is not None and VL is not None:
            self.scFv = VH + linker + VL
            self.VH, self.VL = VH, VL
        else:
            self.scFv, self.VH, self.VL = None, None, None

        self.linker = linker
        self.annotation = None
        super().__init__(index=index, name=name, sequence=self.scFv, structure=structure, partition=partition,
                         features=features, attributes=attributes)

    @property
    def data(self):
        if self.structure is not None:
            structure = self.structure if isinstance(self.structure, str) else True
        else:
            structure = None
        features = {k: True if v is not None else None for k, v in self.features.items()}
        # attributes = {k: obj2str(v) for k, v in self.attributes.items()}

        return {
            'index': self.index,
            'name': self.name,
            'partition': self.partition,
            'length': self.length,
            'sequence': self.sequence,
            'scFv': self.scFv,
            'VH': self.VH,
            'VL': self.VL,
            'structure': structure,
            'graph': True if self.graph_data is not None else None,
            'annotation': obj2str(self.annotation) if self.annotation is not None else None,
            **features,
            **self.attributes
        }

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
        return f'Antibody([{index}]\t\t{sequence}\t\t({attr}))'

    def __repr__(self):
        return self.__str__()
