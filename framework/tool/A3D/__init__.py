import os

import pandas as pd

from framework import paths
from framework.utils.data.dict_utils import merge_dicts
from framework.utils.file.path_utils import get_basename, check_path, list_dirs, is_path_exist
from framework.utils.file.read_utils import read_fasta
from framework.utils.file.write_utils import write_fasta
from framework.utils.parallel.parallel_utils import asyn_parallel

command = 'source /home/hew/anaconda3/bin/activate a3d\naggrescan -i {} -w {} -v 0'


def run_script(pdb, output_dir):
    output_dir = os.path.join(output_dir, get_basename(pdb), '')
    script_file = output_dir + 'command.sh'
    result = os.system(f'sh {script_file}')
    return result


def read_A3D_result(head, result_dir):
    path = os.path.join(result_dir, 'A3D.csv')
    A3D_scores = pd.read_csv(path).score.to_numpy() if is_path_exist(path) else None
    A3D_scores_avg = A3D_scores.mean() if A3D_scores is not None else None
    return {head: (A3D_scores, A3D_scores_avg)}


def write_temp_fasta(fasta, pdb_dir, pdbs, output_dir, temp_path):
    if pdbs is None:
        seqs, desc = read_fasta(fasta)
        pdbs = [os.path.join(pdb_dir, head + '.pdb') for head in desc]
    else:
        seqs = [get_basename(pdb) + '_seq' for pdb in pdbs]
        desc = [get_basename(pdb) + '_head' for pdb in pdbs]

    temp_seqs, temp_heads, temp_pdbs = [], [], []
    for i, pdb in enumerate(pdbs):
        output_dir_i = os.path.join(output_dir, get_basename(pdb), '')
        result_file_i = os.path.join(output_dir_i, 'A3D.csv')
        if not is_path_exist(result_file_i):
            temp_seqs.append(seqs[i])
            temp_heads.append(desc[i])
            temp_pdbs.append(pdb)
            check_path(output_dir_i, log=False)
            with open(os.path.join(output_dir_i, 'command.sh'), 'w') as f:
                f.write(command.format(pdb, output_dir_i))
    check_path(temp_path)
    write_fasta(temp_path, temp_seqs, temp_heads)
    max_len = None if len(temp_seqs) == 0 else max([len(s) for s in temp_seqs])
    return len(pdbs) - len(temp_pdbs), len(pdbs), max_len, temp_pdbs


class Aggrescan3DWrapper:
    def __init__(self, output_dir, temp_path=paths.root, **kwargs):
        self.output_dir = output_dir
        self.temp_path = os.path.join(temp_path, 'temp', 'a3d_unfinished_list.fasta')
        check_path(self.output_dir)

    def predict_protein(self, fasta=None, pdb_dir=None, pdbs=None):
        num_finished, num_total, max_len, unfinished_pdbs = write_temp_fasta(fasta, pdb_dir, pdbs, self.output_dir, self.temp_path)
        print(f'Number of finished before running: {num_finished}/{num_total}, max_len: {max_len}')
        results = asyn_parallel(run_script, [(pdb, self.output_dir) for pdb in unfinished_pdbs], desc='Predicting A3D data')
        results = 0 if len(results) == 0 else results
        return results

    def check_data(self, fasta):
        seqs, heads = read_fasta(fasta)
        failed_records = [head for head in heads if not is_path_exist(os.path.join(self.output_dir, head, f'A3D.csv'))]
        return failed_records

    def list_data(self, absolute=True):
        return list_dirs(self.output_dir, absolute)

    def load_data(self, fasta=None, heads=None, load_all=False):
        if load_all:
            A3D_scores = asyn_parallel(read_A3D_result, [(get_basename(dir), dir) for dir in self.list_data()], desc='Loading A3D data')
        else:
            if fasta is not None:
                heads = read_fasta(fasta)[1]
            else:
                assert heads is not None
            A3D_scores = asyn_parallel(read_A3D_result, [(head, os.path.join(self.output_dir, head, '')) for head in heads], desc='Loading A3D data')
        return merge_dicts(A3D_scores)
