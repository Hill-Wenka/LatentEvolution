import os

import pandas as pd
from tqdm.notebook import tqdm

from framework.utils.file.path_utils import is_path_exist, get_basename, check_path, list_dirs
from framework.utils.parallel.parallel_utils import asyn_parallel

command = 'source /home/hew/anaconda3/bin/activate a3d\naggrescan -i {} -w {} -v 0'


def A3D_parallel_func(pdb, output_dir):
    output_dir = os.path.join(output_dir, get_basename(pdb), '')
    A3D_result_file = os.path.join(output_dir, 'A3D.csv')
    if not is_path_exist(A3D_result_file):
        check_path(output_dir, log=False)
        script_file = output_dir + 'command.sh'
        with open(script_file, 'w') as f:
            f.write(command.format(pdb, output_dir))
        os.system(f'sh {script_file}')


def read_A3D_result(result_dir):
    A3D_scores = pd.read_csv(os.path.join(result_dir, 'A3D.csv')).score.to_numpy()
    return A3D_scores


class Aggrescan3D:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        check_path(self.output_dir)

    def run_A3D(self, pdbs):
        params = [(pdb, self.output_dir) for pdb in pdbs]
        asyn_parallel(A3D_parallel_func, params)

    def load_A3D_data(self, A3D_result_dirs=None):
        A3D_result_dirs = list_dirs(self.output_dir, absolute=True) if A3D_result_dirs is None else A3D_result_dirs
        A3D_scores = {get_basename(dir): read_A3D_result(dir) for dir in tqdm(A3D_result_dirs)}
        return A3D_scores

    def data_list(self, absolute=True):
        A3D_result_dirs = list_dirs(self.output_dir, absolute=absolute)
        return A3D_result_dirs
