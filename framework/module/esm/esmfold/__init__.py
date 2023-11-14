import os

from framework import paths
from framework.utils.file.path_utils import list_files, check_path
from framework.utils.file.read_utils import read_fasta
from framework.utils.file.write_utils import write_fasta


class ESMFoldWrapper:
    def __init__(self,
                 num_recycles=5,
                 cpu_only=False,
                 cpu_offload=False,
                 extract_script='sh /home/hew/python/AggNet/framework/module/esm/esmfold/extract.sh',
                 temp_path=os.path.join(paths.temp, 'temp.fasta')):
        self.num_recycles = num_recycles
        self.cpu_only = cpu_only
        self.cpu_offload = cpu_offload
        self.extract_script = extract_script
        self.temp_path = temp_path

    def weite_temp_fasta(self, fasta, output):
        seqs, des = read_fasta(fasta)
        done_list = list_files(output)
        temp_seqs, temp_des = [], []
        for i, d in enumerate(des):
            if d + '.pdb' not in done_list:
                temp_seqs.append(seqs[i])
                temp_des.append(d)
        check_path(self.temp_path)
        write_fasta(self.temp_path, temp_seqs, temp_des)
        max_len = None if len(temp_seqs) == 0 else max([len(s) for s in temp_seqs])
        return len(done_list), max_len

    def get_command(self, input, output, max_tokens_per_batch, cpu_only):
        return f'{self.extract_script} {input} {output} {max_tokens_per_batch} {self.num_recycles} {cpu_only} {self.cpu_offload}'

    def run_script(self, input, output, mode='GPU'):
        num_finished, max_len = self.weite_temp_fasta(input, output)
        print(f'Number of finished before {mode} running: {num_finished}, max_len: {max_len}')
        if max_len is not None:
            cpu_only = True if mode == 'CPU' else False
            command = self.get_command(self.temp_path, output, max_len, cpu_only)
            result = os.system(command)
        else:
            result = 0
        return result

    def fold_sequence(self, input, output):
        check_path(output)

        result = self.run_script(input, output, 'GPU')
        if result != 0:
            return 'fail'
        else:
            result = self.run_script(input, output, 'CPU') if not self.cpu_only else 0
        return 'success' if result == 0 else 'fail'
