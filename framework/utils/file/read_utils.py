import json

import numpy as np
import pandas as pd
import torch
import yaml
from Bio import SeqIO


def file2data(path, **kwargs):
    '''
    从硬盘读取文件，根据文件路径后缀判断读取文件的类型，返回文件内容
    :param path: 文件路径
    :return: True
    '''
    suffix = path.split('.')[-1]
    if suffix == 'fasta':
        data = read_fasta(path)
    elif suffix == 'pt':
        data = torch.load(path, **kwargs)
    elif suffix == 'npy':
        data = np.load(path, **kwargs)
    elif suffix == 'xlsx':
        data = pd.read_excel(path, **kwargs)
    elif suffix == 'tsv' or suffix == 'csv':
        data = pd.read_csv(path, **kwargs)
    elif suffix == 'fasta':
        data = read_fasta(path)
    else:
        data = read_file(path)
    return data


def read_file(path):
    with open(path, 'r') as f:
        text = f.read()
    return text


def read_fasta(path):
    '''
    调取biopython包读取fasta文件，返回fasta中每一条记录的序列以及对应的描述
    :param path: fasta文件路径
    :return: seqs: list, descriptions: list
    '''
    seqs = [str(fa.seq) for fa in SeqIO.parse(path, 'fasta')]
    descriptions = [fa.description for fa in SeqIO.parse(path, 'fasta')]
    return seqs, descriptions


def read_yaml(path, encoding='utf-8'):
    '''
    读取yaml并转为dict，主要用于从yaml文件读取默认的项目配置以及模型超参数
    :param path: yaml文件路径
    :param encoding: 编码方案，一般是None或者'utf-8'
    :return: yaml_dict: dict
    '''
    try:
        with open(path, encoding=encoding) as file:
            yaml_dict = yaml.load(file.read(), Loader=yaml.FullLoader)
    except Exception:
        raise RuntimeError('Failed to read the yaml file, the specific encoding is wrong')
    return yaml_dict


def read_json(path):
    '''
    读取json文件，返回对应的数据字典
    :param path: json文件路径
    :return: data: dict
    '''
    with open(path, 'r') as load_f:
        data = json.load(load_f)
    return data


def read_data_label_file(path, index_col=None, data_col=None, label_col=None):
    '''
    专门处理类似seq-label pair风格的数据，从xlsx, csv, tsv等类表格格式文件中读取数据并转换为 data, labels 列表
    :param path: 读取的文件路径
    :param index_col: 文件中作为 index 列的字段
    :param data_col: 文件中作为 data 列的字段
    :param label_col: 文件中作为 label 列的字段
    :return: data: list, label: list
    '''
    with open(path, 'r') as file:
        for line in file:
            if line[-1] == '\n':
                line = line[:-1]
            columns = line.split('\t')
            break

    if index_col is None:
        if 'index' in columns:
            index_col = 'index'
        elif 'Index' in columns:
            index_col = 'Index'
        elif 'idx' in columns:
            index_col = 'idx'
        elif 'Idx' in columns:
            index_col = 'Idx'
        else:
            raise RuntimeError('Please specify param \"index_col\" since no default keywords match')
    if data_col is None:
        if 'sequence' in columns:
            data_col = 'sequence'
        elif 'Sequence' in columns:
            data_col = 'Sequence'
        elif 'seq' in columns:
            data_col = 'seq'
        elif 'Seq' in columns:
            data_col = 'Seq'
        elif 'data' in columns:
            data_col = 'data'
        elif 'Data' in columns:
            data_col = 'Data'
        else:
            raise RuntimeError('Please specify param \"data_col\" since no default keywords match')
    if label_col is None:
        if 'label' in columns:
            label_col = 'label'
        elif 'Label' in columns:
            label_col = 'Label'
        elif 'class' in columns:
            label_col = 'class'
        elif 'Class' in columns:
            label_col = 'Class'
        else:
            raise RuntimeError('Please specify param \"label_col\" since no default keywords match')

    if 'xlsx' not in path:
        df = pd.read_excel(path, index_col=index_col)
    else:
        sep = '\t' if '.tsv' in path else ','
        df = pd.read_csv(path, sep=sep, index_col=index_col)
    data = df[data_col].tolist()
    labels = df[label_col].tolist()
    return data, labels
