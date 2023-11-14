import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from framework.utils.file.path_utils import check_path


def data2file(data, path, **kwargs):
    '''
    将数据保存至硬盘，根据输出路径后缀判断输出文件类型，输出成功返回True，否则报错
    :param data: 数据
    :param path: 输出路径
    :return: True
    '''
    check_path(path)
    suffix = path.split('.')[-1]
    if suffix == 'pt':
        torch.save(data, path, **kwargs)
    elif suffix == 'npy':
        np.save(path, data, **kwargs)
    elif suffix == 'xlsx':
        data.to_excel(path, **kwargs)
    elif suffix == 'tsv':
        if type(data) != pd.DataFrame:
            data = pd.DataFrame(data)
        data.to_csv(path, sep='\t', **kwargs)
    elif suffix == 'csv':
        if type(data) != pd.DataFrame:
            data = pd.DataFrame(data)
        data.to_csv(path, sep=',', **kwargs)
    elif suffix == 'fasta':
        write_fasta(path, data, **kwargs)
    else:
        write_file(data, path, **kwargs)
    return True


def write_file(text, file, **kwargs):
    with open(file, 'w') as f:
        f.write(text)
    return True


def write_fasta(path, seqs, custom_index=None, descriptions=None):
    '''
    调取biopython包输出fasta文件，写入成功则返回True，否则会报错
    :param path: 输出的目标路径
    :param seqs: 序列列表
    :param descriptions: 描述/标签列表
    :param custom_index: 自定义索引，如果为None则使用默认index，从0至len(seqs)-1
    :return: result: bool
    '''
    check_path(path)
    records = []
    if custom_index is None:
        custom_index = [str(i) for i in range(len(seqs))]
    for i in range(len(seqs)):
        if descriptions is None:
            seq_record = SeqRecord(Seq(seqs[i]), id=custom_index[i], description='')
        else:
            seq_record = SeqRecord(Seq(seqs[i]), id=custom_index[i], description=f'| {descriptions[i]}')
        records.append(seq_record)
    try:
        SeqIO.write(records, path, 'fasta')
    except Exception:
        raise RuntimeError('Failed to write fasta')
    return True


def write_data_label_file(path, seqs, labels, custom_index=None):
    '''
    将seq_list, label_list输出至xlsx, csv, tsv等类表格格式文件，写入成功则返回True，否则会报错
    :param path: 输出的目标路径
    :param seqs: 序列列表
    :param labels: 标签列表
    :param custom_index: 自定义索引，如果为None则使用默认index，从0至len(seqs)-1
    :return: True
    '''
    check_path(path)
    if custom_index is None:
        custom_index = [i for i in range(len(seqs))]
    df = pd.DataFrame({'Index': custom_index, 'Data': seqs, 'Label': labels})
    if 'xlsx' in path:
        df.to_excel(path, index=False)
    else:
        sep = '\t' if '.tsv' in path else ','
        df.to_csv(path, sep=sep, index=False)
    return True
