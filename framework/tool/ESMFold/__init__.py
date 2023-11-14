import os

from framework import paths
from framework.utils.bio.struct_utils import get_atom_array, compute_sasa
from framework.utils.data.dict_utils import merge_dicts
from framework.utils.file.path_utils import check_path, list_files, get_basename, is_path_exist
from framework.utils.file.read_utils import read_fasta
from framework.utils.file.write_utils import write_fasta
from framework.utils.parallel.parallel_utils import asyn_parallel

extract_script = os.path.join(paths.framework, 'tool', 'ESMFold', 'extract.sh')
command = f'sh {extract_script}'


def get_command(input, output, max_tokens_per_batch, num_recycles, cpu_only, cpu_offload):
    return f'{command} {input} {output} {max_tokens_per_batch} {num_recycles} {cpu_only} {cpu_offload}'


def run_script(input, output_dir, temp_path, max_tokens_per_batch, num_recycles=5, cpu_only=False, cpu_offload=False):
    mode = 'cpu' if cpu_only else 'gpu'
    num_finished, num_total, max_len = write_temp_fasta(input, output_dir, temp_path)
    print(f'Number of finished before {mode} running: {num_finished}/{num_total}, max_len: {max_len}')
    if max_len is not None:
        # max_tokens_per_batch = min(max_len, max_tokens_per_batch)
        command = get_command(temp_path, output_dir, max_tokens_per_batch, num_recycles, cpu_only, cpu_offload)
        print('command', command)
        result = os.system(command)
    else:
        result = 0
    return result


def write_temp_fasta(input, output_dir, temp_path):
    if '.fasta' in input:
        seqs, heads = read_fasta(input)
    else:
        seqs = [input] if type(input) is str else input
        heads = [f'temp_{i}' for i in range(len(seqs))]

    temp_seqs, temp_heads = [], []
    for i, head in enumerate(heads):
        file = os.path.join(output_dir, head + '.pdb')
        if not is_path_exist(file):
            temp_seqs.append(seqs[i])
            temp_heads.append(head)
    check_path(temp_path)
    write_fasta(temp_path, temp_seqs, temp_heads)
    max_len = None if len(temp_seqs) == 0 else max([len(s) for s in temp_seqs])
    return len(seqs) - len(temp_seqs), len(seqs), max_len


def parallel_load_atom_array(structure, require_sasa):
    head = get_basename(structure)
    try:  # must use try except, otherwise the program will be terminated if one of the structure is invalid
        structure = get_atom_array(structure)
        sasa = compute_sasa(structure) if require_sasa else None
    except:
        print(f'Warning: {head} failed to load')
        structure = None
        sasa = None
    return {head: (structure, sasa)}


class ESMFoldWrapper:
    def __init__(
            self,
            output_dir,
            temp_path=paths.root,
            num_recycles=5,
            cpu_only=False,
            cpu_offload=False,
            max_tokens_per_batch=800,
            **kwargs
    ):
        self.output_dir = output_dir
        self.temp_path = os.path.join(temp_path, 'temp', 'esmfold_unfinished_list.fasta')
        self.num_recycles = num_recycles
        self.cpu_only = cpu_only
        self.cpu_offload = cpu_offload
        self.max_tokens_per_batch = max_tokens_per_batch
        check_path(self.output_dir)

    def fold_sequence(self, input):
        if self.cpu_only:  # only run on cpu
            result = run_script(input, self.output_dir, self.temp_path, self.max_tokens_per_batch, self.num_recycles, True, self.cpu_offload)
        else:  # run on gpu first, then run on cpu
            result = run_script(input, self.output_dir, self.temp_path, self.max_tokens_per_batch, self.num_recycles, False, self.cpu_offload)
            if result != 0:
                return 'fail'
            else:
                if not self.cpu_only:
                    result = run_script(input, self.output_dir, self.temp_path, self.max_tokens_per_batch, self.num_recycles, True, self.cpu_offload)
                else:
                    result = 0
        return 'success' if result == 0 else 'fail'

    def check_data(self, fasta):
        seqs, heads = read_fasta(fasta)
        failed_records = [head for head in heads if not is_path_exist(os.path.join(self.output_dir, head + '.pdb'))]
        return failed_records

    def list_data(self, absolute=True):
        return list_files(self.output_dir, absolute)

    def load_data(self, fasta=None, heads=None, load_all=False, require_sasa=True):
        if load_all:
            data = asyn_parallel(parallel_load_atom_array, [(x, require_sasa) for x in self.list_data()], desc='Loading ESMFold data')
        else:
            if fasta is not None:
                heads = read_fasta(fasta)[1]
            else:
                assert heads is not None
            params = [(os.path.join(self.output_dir, head + '.pdb'), require_sasa) for head in heads]
            data = asyn_parallel(parallel_load_atom_array, params, desc='Loading ESMFold data')
        return merge_dicts(data)
