import os

from tqdm.notebook import tqdm

from framework import paths
from framework.utils.file.path_utils import check_path, list_files, get_basename, is_path_exist
from framework.utils.file.read_utils import read_fasta, file2data
from framework.utils.file.write_utils import write_fasta

esm_model_names = ['esm1_t34_670M_UR50S',
                   'esm1_t34_670M_UR50D',
                   'esm1_t34_670M_UR100',
                   'esm1_t12_85M_UR50S',
                   'esm1_t6_43M_UR50S',
                   'esm1b_t33_650M_UR50S',
                   'esm_msa1_t12_100M_UR50S',
                   'esm_msa1b_t12_100M_UR50S',
                   'esm1v_t33_650M_UR90S_1',
                   'esm1v_t33_650M_UR90S_2',
                   'esm1v_t33_650M_UR90S_3',
                   'esm1v_t33_650M_UR90S_4',
                   'esm1v_t33_650M_UR90S_5',
                   'esm_if1_gvp4_t16_142M_UR50',
                   'esm2_t6_8M_UR50D',
                   'esm2_t12_35M_UR50D',
                   'esm2_t30_150M_UR50D',
                   'esm2_t33_650M_UR50D',
                   'esm2_t36_3B_UR50D',
                   'esm2_t48_15B_UR50D']

extract_script = os.path.join(paths.framework, 'tool', 'ESM', 'extract.sh')


def get_command(input, output, model_name, repr_layers, include):
    return f'sh {extract_script} {model_name} {input} {output} {repr_layers} {include}'


def write_temp_fasta(input, output_dir, temp_path):
    if '.fasta' in input:
        seqs, heads = read_fasta(input)
    else:
        seqs = [input] if type(input) is str else input
        heads = [f'temp_{i}' for i in range(len(seqs))]

    temp_seqs, temp_heads = [], []
    for i, head in enumerate(heads):
        if not is_path_exist(os.path.join(output_dir, head + '.pt')):
            temp_seqs.append(seqs[i])
            temp_heads.append(head)
    check_path(temp_path)
    write_fasta(temp_path, temp_seqs, temp_heads)
    max_len = None if len(temp_seqs) == 0 else max([len(s) for s in temp_seqs])
    return len(seqs) - len(temp_seqs), len(seqs), max_len


class ESMWrapper:
    def __init__(
            self,
            output_dir,
            temp_path=paths.root,
            model_name='esm2_t33_650M_UR50D',
            repr_layers='33',
            include='logits,mean,per_tok,contacts',
            **kwargs
    ):
        self.output_dir = output_dir
        self.temp_path = os.path.join(temp_path, 'temp', 'esm_unfinished_list.fasta')
        self.model_name = model_name
        self.repr_layers = repr_layers
        self.include = include
        check_path(self.output_dir)

    def extract_embeddings(self, input):
        num_finished, num_total, max_len = write_temp_fasta(input, self.output_dir, self.temp_path)
        print(f'Number of finished before running: {num_finished}/{num_total}, max_len: {max_len}')
        if max_len is not None:
            result = os.system(get_command(self.temp_path, self.output_dir, self.model_name, self.repr_layers, self.include))
        else:
            result = 0
        return result

    def check_data(self, fasta):
        seqs, heads = read_fasta(fasta)
        failed_records = [head for head in heads if not is_path_exist(os.path.join(self.output_dir, head + '.pt'))]
        return failed_records

    def list_data(self, absolute=True):
        return list_files(self.output_dir, absolute)

    def load_data(self, fasta=None, heads=None, load_all=False):
        if load_all:
            pbar = tqdm(self.list_data())
            pbar.set_description('Loading ESM data')
            results = {get_basename(file): file2data(file) for file in pbar}
        else:
            if fasta is not None:
                heads = read_fasta(fasta)[1]
            else:
                assert heads is not None
            pbar = tqdm(heads)
            pbar.set_description('Loading ESM data')
            results = {head: file2data(os.path.join(self.output_dir, head + '.pt')) for head in pbar}
        return results
