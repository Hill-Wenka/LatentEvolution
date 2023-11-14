import numpy as np
from aaindex import aaindex1
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class AAFeature():
    def __init__(self, no_gap=True):
        super(AAFeature, self).__init__()
        amino_acids = aaindex1.amino_acids()
        if no_gap:
            amino_acids.remove('-')
        self.one_hot_array = np.eye(len(amino_acids))
        self.aa_row_index, self.aaindex_array = self.get_aaindex_table(amino_acids)
    
    def get_aaindex_table(self, amino_acids):
        aa_row_index = {a: i for i, a in enumerate(amino_acids)}
        aaindex_array = np.zeros([len(aa_row_index), aaindex1.num_records()])
        col_index = 0
        for i, x in aaindex1.parse_aaindex().items():
            # print(f'{i}: {x}')
            for a, v in x['values'].items():
                if a in aa_row_index:
                    aaindex_array[aa_row_index[a], col_index] = v
                else:
                    if a != '-':
                        raise Warning()
            col_index += 1
        return aa_row_index, aaindex_array
    
    def batch_encode(self, seqs, onehot=True, aaindex=True, flatten=True, normalize=False, method='standard'):
        if normalize:
            if not flatten:
                raise RuntimeError('flatten is needed as True if normalize=True')
            if method == 'standard':
                scalar = StandardScaler()
            elif method == 'standard':
                scalar = MinMaxScaler()
            else:
                raise RuntimeError(f'No such normalization method: {method}')
        
        one_hot = np.array([[self.aa_row_index[aa] for aa in seq] for seq in seqs])
        encoding = {}
        if onehot:
            onehot_encoding = self.one_hot_array[one_hot]
            if flatten:
                onehot_encoding = onehot_encoding.reshape(onehot_encoding.shape[0], -1)
            encoding['onehot'] = onehot_encoding
        if aaindex:
            aaindex_encoding = self.aaindex_array[one_hot]
            if flatten:
                aaindex_encoding = aaindex_encoding.reshape(aaindex_encoding.shape[0], -1)
                if normalize:
                    aaindex_encoding = scalar.fit_transform(aaindex_encoding)
            encoding['aaindex'] = aaindex_encoding
        return encoding


if __name__ == '__main__':
    aaindex = AAFeature()
    print(len(aaindex.aa_row_index), aaindex.aa_row_index)
    print(aaindex.one_hot_array)
    print(aaindex.one_hot_array.shape)
    print(aaindex.aaindex_array)
    print(aaindex.aaindex_array.shape)
