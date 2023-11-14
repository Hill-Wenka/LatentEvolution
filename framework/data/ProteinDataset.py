import os

import numpy as np
import pandas as pd
import torch.utils.data as Data
from tqdm.notebook import tqdm

from framework import paths
from framework.data.Antibody import Antibody
from framework.data.HDF5 import HDF5
from framework.data.Protein import Protein
from framework.tool import ANARCI, ESM, ESMFold, A3D
from framework.utils.bio.struct_utils import get_atom_array, atom_array_to_sequence
from framework.utils.data.dict_utils import merge_dicts
from framework.utils.file.path_utils import remove_files
from framework.utils.file.write_utils import data2file
from framework.utils.parallel.parallel_utils import asyn_parallel


def parallel_construct_graph(index, data, **kwargs):
    if data.graph_data is None:
        data.construct_graph(**kwargs)
    return {index: data.graph_data}


class ProteinDataset(Data.Dataset):
    def __init__(self, name, data_path=None, **kwargs):
        self.name = name
        self.data_path = data_path if data_path is not None else os.path.join(paths.data, name)
        self.pt = os.path.join(self.data_path, 'cooked', f'{self.name}.pt')
        self.csv = os.path.join(self.data_path, 'cooked', f'{self.name}.csv')
        self.fasta = os.path.join(self.data_path, 'cooked', f'{self.name}.fasta')
        self.h5 = os.path.join(self.data_path, 'cooked', f'{self.name}.h5')
        self.save_keys = ['index', 'name', 'partition', 'length', 'sequence', 'structure', 'graph_data', 'features', 'attributes']
        self._data = None

        self.anarci = os.path.join(self.data_path, 'anarci', '')
        self.esm = os.path.join(self.data_path, 'esm', '')
        self.esmfold = os.path.join(self.data_path, 'esmfold', '')
        self.a3d = os.path.join(self.data_path, 'a3d', '')

        self.anarci_fasta = os.path.join(self.data_path, 'temp', 'anarci_load_list.fasta')
        self.esm_fasta = os.path.join(self.data_path, 'temp', 'esm_load_list.fasta')
        self.esmfold_fasta = os.path.join(self.data_path, 'temp', 'esmfold_load_list.fasta')
        self.a3d_fasta = os.path.join(self.data_path, 'temp', 'a3d_load_list.fasta')

        self.hdf5 = HDF5(self.h5, **kwargs)

    def init_submodules(self, **kwargs):
        self.anarci_wrapper = ANARCI.ANARCIWrapper(self.anarci, temp_path=self.data_path, **kwargs)  # ANARCI
        self.esm_wrapper = ESM.ESMWrapper(self.esm, temp_path=self.data_path, **kwargs)  # ESM
        self.esmfold_wrapper = ESMFold.ESMFoldWrapper(self.esmfold, temp_path=self.data_path, **kwargs)  # ESMFold
        self.a3d_wrapper = A3D.Aggrescan3DWrapper(self.a3d, temp_path=self.data_path, **kwargs)  # A3D

    def construct(self, data, partitions=None, append=True):
        # data: list of Antibody or Protein objects, partition: list of str
        partitions = [None] * len(data) if partitions is None else partitions
        if append:
            self._data = {} if self._data is None else self._data
        else:
            self._data = {}
        for d, p in zip(data, partitions):
            self._data[d.index] = d
            if self._data[d.index].partition is None:
                self._data[d.index].partition = p

    def construct_subset(self, indices, name=None):
        name = self.name + ' subset' if name is None else name
        subset = ProteinDataset(name)
        subset.construct(self.__getitem__(indices))
        return subset

    def save(self, overwrite=True, **kwargs):
        if overwrite:
            remove_files([self.csv, self.fasta, self.h5])

        # save data to csv
        data2file(self.dataframe, self.csv, index=False)

        # save data to h5 (this approach saves the data with the same structure as the original data)
        save_keys = kwargs.get('save_keys', self.save_keys.copy())
        if isinstance(self.data[self.index[0]], Antibody):
            save_keys += ['scFv', 'VH', 'VL', 'annotation']
        print(f'[HDF5] save keys: {save_keys}')

        pbar = tqdm(self.index)
        pbar.set_description('Saving dataset to HDF5')
        [self.hdf5.save(idx, key, getattr(self.data[idx], key) if hasattr(self.data[idx], key) else None) for key in save_keys for idx in pbar]
        # 不能并行写入，会报错

        # save data to fasta
        data2file(self.sequences, self.fasta, custom_index=self.index)

    def load_meta_data(self, h5_file=None):
        self.h5 = self.h5 if h5_file is None else h5_file
        self.hdf5 = HDF5(self.h5)

    def load(self, groups=None, data_class='Protein', lazy_load=False, h5_file=None):
        if data_class == 'Protein':
            data_class = Protein
        elif data_class == 'Antibody':
            data_class = Antibody
        else:
            raise ValueError(f'Invalid data class: {data_class}')

        if self.index is None or h5_file is not None:
            self.load_meta_data(h5_file)  # load data from h5

        groups = self.index if groups is None else groups  # load all data if groups is None else load data with specified index
        load_data = self.hdf5.load(groups=groups)
        self._data = {} if self._data is None else self._data
        temp_data = []
        for idx in load_data:
            if lazy_load:  # load data only when needed
                data = data_class()
                [setattr(data, key, value) for key, value in load_data[idx].items()]
                temp_data.append(data)
            else:
                self.data[idx] = data_class()
                [setattr(self.data[idx], key, value) for key, value in load_data[idx].items()]

        if load_data:
            return temp_data

    def is_partitioned(self):
        data_partitions = np.array([self.data[idx].partition for idx in self.index])
        return not (data_partitions == None).all()

    def get_partition_set(self, partition):
        # partition: 'train', 'valid', 'test', 'predict'
        if self.data is None:
            raise RuntimeError('Please load data first')

        if self.is_partitioned():
            indices = np.array([idx for idx in self.index if self.data[idx].partition == partition])
            data = self.__getitem__(indices)
        else:
            data = None
        return data

    @property
    def train_set(self):
        return self.get_partition_set('train')

    @property
    def valid_set(self):
        return self.get_partition_set('valid')

    @property
    def test_set(self):
        return self.get_partition_set('test')

    @property
    def predict_set(self):
        return self.get_partition_set('predict')

    @property
    def data(self):
        return self._data

    @property
    def index(self):
        if self.data is None:
            index = np.array(self.hdf5.groups())  # load index from h5 file, groups in h5 file are the index of data
        else:
            index = np.array(list(self.data.keys()))  # after python 3.7, dict is ordered
        return index

    @property
    def lengths(self):
        if self.data is None:
            raise RuntimeError('Please load data first')
        else:
            return np.array([len(self.data[idx].sequence) for idx in self.index])

    @property
    def sequences(self):
        if self.data is None:
            raise RuntimeError('Please load data first')
        else:
            return np.array([self.data[idx].sequence for idx in self.index])

    @property
    def structures(self):
        if self.data is None:
            raise RuntimeError('Please load data first')
        else:
            return np.array([self.data[idx].structure for idx in self.index])

    @property
    def labels(self):
        if self.data is None:
            raise RuntimeError('Please load data first')
        else:
            if 'label' in self.data[self.index[0]].attributes:
                labels = np.array([self.data[idx].attributes['label'] for idx in self.index])
            else:
                print('Data have not assigned label')
                labels = None
            return labels

    def get_property(self, attribute, property, indices=None):
        # property: 'attribute', 'feature', 'property'
        indices = self.index if indices is None else indices
        if self.data is None:
            raise RuntimeError('Please load data first')
        else:
            data = self.__getitem__(indices)
            if property == 'attribute':
                data = [d.attributes[attribute] for d in data]
            elif property == 'feature':
                data = [d.features[attribute] for d in data]
            elif property == 'property':
                data = [getattr(d, attribute) for d in data]
            else:
                raise ValueError(f'Invalid property: {property}')
        return data

    @property
    def dataframe(self):
        if self.data is None:
            dataframe = pd.read_csv(self.csv)
        else:
            dataframe = pd.DataFrame.from_dict({i: item.data for i, (index, item) in enumerate(self.data.items())}, orient='index')
            dataframe = dataframe.sort_values(by='index').reset_index(drop=True)
            # because when load data from h5, the order of data based on the order of groups in h5 file
        return dataframe

    def show(self, n=None):
        n = len(self.data) if n is None else n
        return self.dataframe.head(n)

    def check_sequence(self):
        pbar = tqdm(self.index)
        pbar.set_description('Checking sequences')
        for index in pbar:
            if self.data[index].sequence is None:
                assert self.data[index].structure is not None
                structure = self.data[index].structure
                if isinstance(structure, str):
                    structure = get_atom_array(structure)
                convert_seq = atom_array_to_sequence(structure)
                self.data[index].sequence = convert_seq

                if convert_seq is None:
                    print(f'Warning: {index} failed to get sequence')

    def run_anarci(self):
        print('=' * 30, 'ANARCI', '=' * 30)
        if 'annotation' not in self.dataframe.columns:
            return None
        else:
            run_df = self.dataframe[self.dataframe.annotation.isna()]
        data2file(run_df.sequence.values, self.anarci_fasta, custom_index=run_df['index'].values)
        run_states = self.anarci_wrapper.annotate_sequence(self.anarci_fasta)
        failed_records = self.anarci_wrapper.check_data(self.anarci_fasta)
        results = self.anarci_wrapper.load_data(self.anarci_fasta)
        if len(failed_records) > 0:
            print(f'[ANARCI] failed records: {failed_records}\nrun states: {run_states}')
        pbar = tqdm(results.items())
        pbar.set_description('Setting ANARCI data')
        for head, annotation in pbar:
            self.__getitem__(head).annotation = annotation

    def run_esm(self):
        print('=' * 30, 'ESM', '=' * 30)
        if 'esm' not in self.dataframe.columns:
            run_df = self.dataframe
        else:
            run_df = self.dataframe[self.dataframe.esm.isna()]
        data2file(run_df.sequence.values, self.esm_fasta, custom_index=run_df['index'].values)
        run_states = self.esm_wrapper.extract_embeddings(self.esm_fasta)
        failed_records = self.esm_wrapper.check_data(self.esm_fasta)
        results = self.esm_wrapper.load_data(self.esm_fasta)
        if len(failed_records) > 0:
            print(f'[ESM] failed records: {failed_records}\nrun states: {run_states}')
        pbar = tqdm(results.items())
        pbar.set_description('Setting ESM data')
        for head, esm in pbar:
            self.__getitem__(head).set_feature('esm', esm)

    def run_esmfold(self, require_sasa):
        print('=' * 30, 'ESMFold', '=' * 30)
        run_df = self.dataframe[self.dataframe.structure.isna()]
        data2file(run_df.sequence.values, self.esmfold_fasta, custom_index=run_df['index'].values)
        run_states = self.esmfold_wrapper.fold_sequence(self.esmfold_fasta)
        failed_records = self.esmfold_wrapper.check_data(self.esmfold_fasta)
        results = self.esmfold_wrapper.load_data(fasta=self.esmfold_fasta, require_sasa=require_sasa)
        if len(failed_records) > 0:
            print(f'[ESMFold] failed records: {failed_records}\nrun states: {run_states}')
        for head, (structure, sasa) in results.items():
            protein = self.__getitem__(head)
            protein.structure = structure
            struct_seq = atom_array_to_sequence(structure)  # check if the sequence is consistent with the structure
            assert protein.sequence == struct_seq, f'Error:\nsequence: {protein.sequence}\nstructure: {struct_seq}'
            if require_sasa:
                protein.set_feature('sasa', sasa)

        for index in self.index:
            if isinstance(self.__getitem__(index).structure, str):
                self.__getitem__(index).structure = get_atom_array(self.__getitem__(index).structure)

    def run_a3d(self):
        print('=' * 30, 'Aggrescan3D', '=' * 30)
        if 'a3d' not in self.dataframe.columns:
            run_df = self.dataframe
        else:
            run_df = self.dataframe[self.dataframe.a3d.isna()]
        data2file(run_df.sequence.values, self.a3d_fasta, custom_index=run_df['index'].values)
        run_states = self.a3d_wrapper.predict_protein(self.a3d_fasta, self.esmfold)
        failed_records = self.a3d_wrapper.check_data(self.a3d_fasta)
        results = self.a3d_wrapper.load_data(self.a3d_fasta)
        if len(failed_records) > 0:
            print(f'[Aggrescan3D] failed records: {failed_records}\nrun states: {run_states}')
        pbar = tqdm(results.items())
        pbar.set_description('Setting A3D data')
        for head, (a3d, a3d_avg) in pbar:
            self.__getitem__(head).set_feature('a3d', a3d)
            self.__getitem__(head).set_feature('a3d_avg', a3d_avg)

    def construct_graph(self, **kwargs):
        print('=' * 30, 'Construct Graph', '=' * 30)
        results = asyn_parallel(parallel_construct_graph, [(index, self.__getitem__(index)) for index in self.index], kwds=kwargs,
                                desc='Constructing Graphs')
        pbar = tqdm(merge_dicts(results).items())
        pbar.set_description('Setting graph data')
        for index, graph_data in pbar:
            self.__getitem__(index).graph_data = graph_data

    def run_submodules(self, require_sequence=True, require_annotation=True, require_esm=False, require_structure=True, require_A3D=True,
                       require_graph=True, require_sasa=True, **kwargs):
        self.init_submodules(**kwargs)  # initialize submodules
        if require_sequence:
            self.check_sequence()

        exist_antibody = np.array([isinstance(v, Antibody) for k, v in self.data.items()]).any()
        if require_annotation and exist_antibody:
            self.run_anarci()

        if require_esm:
            self.run_esm()

        if require_structure:
            self.run_esmfold(require_sasa)

        if require_A3D and require_structure:
            self.run_a3d()

        if require_graph and require_structure:
            self.construct_graph(**kwargs)

    def drop_data(self, columns):
        if isinstance(columns, str):
            columns = [columns]

        for column in columns:
            if column in self.dataframe.columns:
                drop_df = self.dataframe[self.dataframe[column].isna()]
                print(f'Dropping [{len(drop_df)}] data point with missing values in the column [{column}]')
                invalid_ids = drop_df['index'].tolist()
                self.__delitem__(invalid_ids)
            else:
                raise KeyError(f'Column [{column}] not found')

    def __str__(self):
        len_dataset = self.__len__() if self.data is not None else 'unconstructed'
        return f'{self.name}({len_dataset}), location: {self.data_path}'

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.index)

    def __getitem__(self, key):
        if isinstance(key, tuple) or isinstance(key, list) or isinstance(key, np.ndarray):
            if len(key) == 0:
                return []
            else:
                if key[0] in self.data:
                    method = lambda x: x
                elif isinstance(key[0], int) or isinstance(key[0], np.int64):
                    method = lambda x: self.index[x]
                else:
                    raise KeyError(f'Key {key} not found, key[0]: {type(key[0])}')

                items = [self.data[method(k)] for k in key]
        else:
            if key in self.data:
                method = lambda x: x
            elif isinstance(key, int) or isinstance(key[0], np.int64):
                method = lambda x: self.index[x]
            else:
                raise KeyError(f'Key {key} not found, key[0]: {type(key)}')

            items = self.data[method(key)]
        return items

    def __delitem__(self, key):
        if isinstance(key, tuple) or isinstance(key, list) or isinstance(key, np.ndarray):
            if len(key) != 0:
                if key[0] in self.data:
                    method = lambda x: x
                elif isinstance(key[0], int) or isinstance(key[0], np.int64):
                    method = lambda x: self.index[x]
                else:
                    print(f'Key {key[0]} not found')
                    return

                for k in key:
                    if k in self.index:
                        del self.data[method(k)]
                    else:
                        print(f'Key {k} not found')
        else:
            if key in self.data:
                method = lambda x: x
            elif isinstance(key, int) or isinstance(key[0], np.int64):
                method = lambda x: self.index[x]
            else:
                print(f'Key {key[0]} not found')
                return

            if key in self.index:
                del self.data[method(key)]
            else:
                print(f'Key {key} not found')
