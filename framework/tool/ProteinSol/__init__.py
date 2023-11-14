import os
import subprocess
import numpy as np
from framework import paths
from framework.utils.file.path_utils import check_path


class ProteinSolWrapper:
    def __init__(self, output_dir, protein_sol_dir=None):
        self.protein_sol_dir = os.path.join(paths.framework, 'tool/ProteinSol/protein-sol/') if protein_sol_dir is None else protein_sol_dir
        self.protein_sol_script = os.path.join(self.protein_sol_dir, 'multiple_prediction_wrapper_export.sh')
        self.output_dir = output_dir
        check_path(self.output_dir)

    def generate_command(self, fasta_path):
        command = f'cd {self.protein_sol_dir} && {self.protein_sol_script} {fasta_path}'
        return command

    def predict_solubility(self, fasta_path, job_name='predict_solubility'):
        command = self.generate_command(fasta_path)
        print('command:', command)
        result = subprocess.run(command, capture_output=True, text=True, shell=True)

        prediction_file = f'{self.protein_sol_dir}/seq_prediction.txt'
        with open(prediction_file) as f:
            outputs = f.read()

        results = [line.replace('SEQUENCE PREDICTIONS,', '').split(',') for line in outputs.split('\n') if 'SEQUENCE PREDICTIONS,' in line]
        solubility_scores = np.array(results)[:, 2].astype(float)
        np.save(self.output_dir + f'{job_name}.npy', solubility_scores)
        return 0

    def read_results(self, job_name='predict_solubility'):
        solubility_scores = np.load(self.output_dir + f'{job_name}.npy')
        return solubility_scores
