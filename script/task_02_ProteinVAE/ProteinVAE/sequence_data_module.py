import lightning as pl
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from framework.data.Antibody import Antibody
from framework.data.Protein import Protein
from framework.data.ProteinDataset import ProteinDataset
from framework.data.Tokenizer import Tokenizer


class SequenceDataModule(pl.LightningDataModule):
    def __init__(self, args, log=True):
        super().__init__()
        if log:
            print('=' * 30, 'PeptideDataModule __init__ Start', '=' * 30)

        self.log = log
        self.args = args
        self.data_args = self.args.data
        self.tokenization_args = self.args.tokenization
        self.train_dataloader_args = self.args.train_dataloader
        self.valid_dataloader_args = self.args.valid_dataloader
        self.test_dataloader_args = self.args.test_dataloader

        self.dataset = ProteinDataset(self.data_args.dataset)
        self.tokenizer = Tokenizer(**self.tokenization_args)
        self.dataframe = None
        self.mode = None  # 数据模式: [train, test, predict]

        if self.log:
            print('=' * 30, 'PeptideDataModule __init__ End', '=' * 30)

    def prepare_data(self, mode=None, **kwargs):
        # auto invoked by trainer
        if self.log:
            print('=' * 30, f'prepare_data: mode[{mode}] Start', '=' * 30)

        if self.dataframe is None and mode != 'predict':
            if self.log:
                print('=' * 30, 'prepare_dataset Start', '=' * 30)
            self.prepare_dataset()
            if self.log:
                print('=' * 30, 'prepare_dataset End', '=' * 30)

        if mode is not None:
            self.mode = mode
            if mode == 'train':
                self.prepare_train_data(**kwargs)
            elif mode == 'test':
                self.prepare_test_data(**kwargs)
            elif mode == 'predict':
                print('prepare_data predict_data kwargs', kwargs)
                self.prepare_predict_data(**kwargs)
            else:
                raise RuntimeError(f'No such pre-defined mode: {mode}')
        else:
            if self.log:
                print('=' * 30, 'prepare_data has been invoked, mode=None', '=' * 30)

        if self.log:
            print('=' * 30, f'prepare_data mode[{mode}] End', '=' * 30)

    def setup(self, stage=None):
        if self.log:
            print('=' * 30, f'Setup [{stage}] Start', '=' * 30)
        if stage == 'fit':
            self.train_dataset = self.dataset.construct_subset(self.train_index, 'train_dataset')
            self.valid_dataset = self.dataset.construct_subset(self.valid_index, 'valid_dataset')
            if self.log:
                print('[len self.train_dataset]', len(self.train_dataset))
                print('[len self.val_dataset]', len(self.valid_dataset))
        elif stage == 'validate':
            self.valid_dataset = self.dataset.construct_subset(self.valid_index, 'valid_dataset')
            if self.log:
                print('[len self.val_dataset]', len(self.valid_dataset))
        elif stage == 'test':
            # self.mode == 'train' 对应于trainer在train结束后立即进行test的情况
            if self.mode == 'train' or self.mode == 'test':
                self.test_dataset = self.dataset.construct_subset(self.test_index, 'test_dataset')
                if self.log:
                    print('[len self.test_dataset]', len(self.test_dataset))
            elif self.mode == 'predict':
                self.predict_dataset = self.dataset.construct_subset(self.predict_index, 'predict_dataset')
                if self.log:
                    print('[len self.predict_dataset]', len(self.predict_dataset))
            else:
                raise RuntimeError(f'No such parameter mode: {self.mode}')
        elif stage == 'predict':
            self.predict_dataset = self.dataset.construct_subset(self.predict_index, 'predict_dataset')
            if self.log:
                print('[len self.predict_dataset]', len(self.predict_dataset))
        else:
            print('stage', stage)
            raise RuntimeError('Parameter {stage} is None or illegal, please set it properly')
        if self.log:
            print('=' * 30, f'Setup [{stage}] End', '=' * 30)

    def prepare_dataset(self):
        self.dataset.load_meta_data()

        # select data with sequence length <= max_len
        max_len = self.data_args.sequence_length if self.data_args.sequence_length is not None else self.dataset.dataframe.length.max()
        seq_len_indices = self.dataset.dataframe['length'] <= max_len
        self.selected_index = self.dataset.index[seq_len_indices]

        # select subset of the dataset for debug
        subset_ratio = self.data_args.mini_set_ratio if self.data_args.mini_set_ratio is not None else 1
        self.selected_index = pd.DataFrame(self.selected_index).sample(frac=subset_ratio).reset_index(drop=True)[0].to_numpy()

        # load data according to selected index
        if not self.data_args.lazy_load:
            if self.log:
                print('<' * 30, 'load data according to selected index', '>' * 30)
            self.dataset.load(self.selected_index, data_class=self.data_args.data_class)

        # get dataframe
        self.dataframe = self.dataset.dataframe

        # partition train/valid/test dataset
        self.partition_dataset()

        if self.log:
            print(f'select the subset for debug, max_len: {max_len}, ratio: {subset_ratio}, number: {len(self.selected_index)}')

    def partition_dataset(self, q=None):
        label_key = self.data_args.label
        mode = 'continuous' if isinstance(self.dataframe[label_key].iloc[0], float) else 'discrete'
        # 根据dataframe.label的分布按比例划分, 如果label是连续值, 则先分箱再按比例划分
        if mode == 'continuous':
            q = len(self.dataframe) // 10 if q is None else q
            self.dataframe['bins'] = pd.qcut(self.dataframe[label_key], q=q)
            groups = self.dataframe['bins'].unique()
            group_column = 'bins'
        elif mode == 'discrete':
            groups = self.dataframe[label_key].unique()
            group_column = label_key
        else:
            raise NotImplementedError

        if self.dataset.is_partitioned():
            if self.log:
                print('the dataset has been partitioned, split valid set from train set')
            if len(self.dataset.valid_set) == 0:  # split valid set from train set
                for g in groups:
                    train_indices = self.dataframe[(self.dataframe['partition'] == 'train') & (self.dataframe[group_column] == g)].index
                    num_valid = int(len(train_indices) * self.data_args.split_valid)
                    valid_indices = train_indices[:num_valid]
                    self.dataframe.loc[valid_indices, 'partition'] = 'valid'
        else:
            if self.log:
                print('the dataset has not been partitioned, split dataset with specific ratio')
            train_ratio = self.args.data.train_ratio if self.args.data.train_ratio is not None else 0.7
            valid_ratio = self.args.data.valid_ratio if self.args.data.valid_ratio is not None else 0.1
            test_ratio = self.args.data.test_ratio if self.args.data.test_ratio is not None else 0.2
            assert train_ratio + valid_ratio + test_ratio == 1

            for g in groups:
                l_idx = self.dataframe[self.dataframe[group_column] == g].index
                num_train = int(len(l_idx) * train_ratio)
                num_valid = int(len(l_idx) * valid_ratio)
                num_test = int(len(l_idx) * test_ratio)
                self.dataframe.loc[l_idx[:num_train], 'partition'] = 'train'
                self.dataframe.loc[l_idx[num_train:(num_train + num_valid)], 'partition'] = 'valid'
                self.dataframe.loc[l_idx[(num_train + num_valid):], 'partition'] = 'test'

        if self.log:
            log_dict = self.dataframe['partition'].value_counts()
            print(f'dataframe partition values:\n{log_dict}')

    def prepare_train_data(self, **kwargs):
        self.train_index = self.dataframe[self.dataframe.partition == 'train']['index'].tolist()
        self.valid_index = self.dataframe[self.dataframe.partition == 'valid']['index'].tolist()
        self.test_index = self.dataframe[self.dataframe.partition == 'test']['index'].tolist()

        if self.log:
            print('len(self.train_dataset)', len(self.train_index))
            print('len(self.valid_dataset)', len(self.valid_index))
            print('len(self.test_dataset)', len(self.test_index))

    def prepare_test_data(self, **kwargs):
        self.test_index = self.dataframe[self.dataframe.partition == 'test']['index'].tolist()

        if self.log:
            print('len(self.test_dataset)', len(self.test_index))

    def prepare_predict_data(self, predict_data='valid', data_class='Protein', **kwargs):
        if data_class == 'Protein':
            data_class = Protein
        elif data_class == 'Antibody':
            data_class = Antibody
        else:
            raise ValueError(f'Invalid data class: {data_class}')

        if isinstance(predict_data, str):
            if predict_data == 'train':
                self.prepare_data('train')  # mode: None -> train
                self.predict_index = self.train_index
            elif predict_data == 'valid':
                self.prepare_data('train')  # mode: None -> train
                self.predict_index = self.valid_index
            elif predict_data == 'test':
                self.prepare_data('test')  # mode: None -> train
                self.predict_index = self.test_index
            elif predict_data == 'train_valid':
                self.prepare_data('train')  # mode: None -> train
                self.predict_index = self.train_index + self.valid_index
            elif predict_data == 'train_valid_test':
                self.prepare_data('train')  # mode: None -> train
                self.predict_index = self.train_index + self.valid_index + self.test_index
            else:
                raise ValueError(f'predict_data: {predict_data} is not valid')
            self.mode = 'predict'  # mode: train -> predict
        elif isinstance(predict_data, (list, np.ndarray, pd.DataFrame, pd.Series)):
            self.predict_index = [f'predict_seq_{i}' if len(seq) > 10 else f'{seq}' for i, seq in enumerate(predict_data)]
            predict_data = [
                data_class(
                    self.predict_index[i],
                    self.predict_index[i],
                    seq,
                    attributes={'ddG': np.nan, 'dS': np.nan}
                )
                for i, seq in enumerate(predict_data)
            ]
            partitions = ['predict'] * len(predict_data)
            self.dataset.construct(predict_data, partitions, True)
        elif isinstance(predict_data, ProteinDataset):
            self.predict_index = predict_data.index
            self.dataset = predict_data
        else:
            raise ValueError(f'predict_data: {predict_data} is not valid:, {type(predict_data)}')

        if self.log:
            print('len(self.predict_index)', len(self.predict_index))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, collate_fn=self.collate_fn, **self.args.train_dataloader)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, collate_fn=self.collate_fn, **self.args.valid_dataloader)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, collate_fn=self.collate_fn, **self.args.test_dataloader)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, collate_fn=self.collate_fn, **self.args.predict_dataloader)

    def collate_fn(self, batch):
        batch_seqs, batch_ddG, batch_dS = [], [], []
        for i, protein in enumerate(batch):
            batch_seqs.append(protein.sequence)
            batch_ddG.append(protein.attributes['ddG'])
            batch_dS.append(protein.attributes['dS'])
        batch_ids, batch_seqs, batch_tokens = self.tokenizer.tokenize(batch_seqs)
        batch_ddG = torch.tensor(batch_ddG)
        batch_dS = torch.tensor(batch_dS)
        return batch_tokens, batch_ddG, batch_dS
