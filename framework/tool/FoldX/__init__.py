import concurrent.futures
import io
import os
import subprocess
import time

import pandas as pd

from framework import paths
from framework.utils.bio.seq_utils import format_mutation, mutate
from framework.utils.file.path_utils import get_basename, check_path, is_path_exist, list_files
from framework.utils.file.read_utils import file2data
from framework.utils.file.write_utils import data2file

'''
use the following command to add execution permissions:
"chmod +x /home/hew/python/LatentEvolution/framework/tool/FoldX/foldx_20231231"

FoldX BuildModel Result Directory Structure:
WT_{pdb_name}_1_{run_id}.pdb: for each input structure
{pdb_name}_1_{run_id}.pdb: for each mutated structure
Average_{pdb_name}.fxout: average mutation energy of each run (only 1 row)
Dif_{pdb_name}.fxout: energy difference between mutant structure and wild-type structure for each run (e.g. numberOfRuns=5, there are 5 rows)
Raw_{pdb_name}.fxout: energy of each structure for each run (e.g. numberOfRuns=5, there are 10 rows, 5 for wild-type, 5 for mutant)
'''

mutation_command = '{} -c BuildModel --numberOfRuns {} --pdb={} --mutant-file={} --output-dir={} --pdb-dir={}'
repair_command = '{} -c RepairPDB --pdb={} --output-dir={} --pdb-dir={}'

# use for ibex
ibex_submit_command = 'sbatch --job-name={} --output={}%j.out --error={}%j.err --time={}:00 {}'
time_per_mut_site = 3  # minutes


def get_squeue_number(username="hew"):
    command = f"squeue -u {username}"
    output = subprocess.check_output(command, shell=True, text=True)
    lines = output.strip().split('\n')
    return len(lines) - 1


def submit_ibex_jobs(scripts, max_job_num=1000, is_return=False):
    current_i = 0
    while current_i < len(scripts):
        if get_squeue_number() < max_job_num:
            os.system(f'{scripts[current_i]}')
            current_i += 1
            if current_i % 10 == 0:
                print('[submitting job] current script i:', current_i)
        else:
            if is_return:
                break
            else:
                print('[waiting to submit] current script i:', current_i)
                time.sleep(5)
    return True


def run_command(command):
    result = subprocess.run(command, capture_output=True, text=True, shell=True)
    return result.stdout.strip()


def run_script(commands, max_workers=25):
    # Run the commands in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(run_command, commands))
    return True


def read_mutation_results(file=None, job_name=None, output_dir=None):
    if job_name is not None:
        assert file is None and output_dir is not None
        job_dir = os.path.join(output_dir, job_name)
        file = [f for f in list_files(job_dir) if f.startswith('Average_') and f.endswith('.fxout')][0]
        file = os.path.join(job_dir, file)
    else:
        assert file is not None

    df_string = '\n'.join(file2data(file).split('\n')[8:])
    result_df = pd.read_csv(io.StringIO(df_string), sep='\t')
    return result_df


class FoldXWrapper:
    def __init__(self, output_dir, temp_path=paths.root, numberOfRuns=5, num_cpu=25, ibex=False, foldx_exe=None, **kwargs):
        self.foldx_exe = os.path.join(paths.framework, 'tool/FoldX/foldx_20231231') if foldx_exe is None else foldx_exe
        self.output_dir = output_dir
        self.temp_path = os.path.join(temp_path, 'temp', 'foldx_unfinished_list.fasta')
        self.numberOfRuns = numberOfRuns
        self.num_cpu = num_cpu
        self.ibex = ibex
        check_path(self.output_dir)

    def build_mutation_model(self, wt_seqs, wt_pdbs, mt_seqs=None, job_names=None, mutation_sites=None, pdb_dir=None, offset=0):
        # process input sequences
        if isinstance(wt_seqs, str):
            wt_seqs = [wt_seqs]
        if isinstance(mt_seqs, str):
            mt_seqs = [mt_seqs]
        if isinstance(wt_pdbs, str):
            wt_pdbs = [wt_pdbs]

        num_mutations = len(mutation_sites) if mt_seqs is None else len(mt_seqs)
        wt_seqs = wt_seqs * num_mutations if len(wt_seqs) == 1 else wt_seqs
        wt_pdbs = wt_pdbs * num_mutations if len(wt_pdbs) == 1 else wt_pdbs

        # process input pdbs
        if pdb_dir is None:
            pdb_dirs = [pdb.split(get_basename(pdb, suffix=True))[0] for pdb in wt_pdbs]
            wt_pdbs = [get_basename(pdb, suffix=True) for pdb in wt_pdbs]
        else:
            pdb_dirs = [pdb_dir] * len(wt_pdbs)
            wt_pdbs = [get_basename(pdb, suffix=True) if pdb_dir in pdb else pdb for pdb_dir, pdb in zip(pdb_dirs, wt_pdbs)]

        # process mutations
        if mutation_sites is not None:
            assert mt_seqs is None
            mt_seqs = [mutate(wt_seq, mutation_site) for wt_seq, mutation_site in zip(wt_seqs, mutation_sites)]
        else:
            assert mt_seqs is not None

        job_names = [f'job_{i}' for i in range(len(wt_seqs))] if job_names is None else job_names
        assert len(wt_seqs) == len(wt_pdbs) == len(mt_seqs) == len(job_names)

        foldx_commands, ibex_commands = self.generate_mutation_command(wt_seqs, mt_seqs, wt_pdbs, pdb_dirs, job_names, offset)
        if 0 < len(foldx_commands) < 10:
            print('\n'.join(foldx_commands))

        if self.ibex:
            results = submit_ibex_jobs(ibex_commands)  # submit jobs to ibexs
        else:
            results = run_script(foldx_commands, max_workers=self.num_cpu)  # run FoldX in parallel
        return results

    def generate_mutation_command(self, wt_seqs, mt_seqs, wt_pdbs, pdb_dirs, job_names, offset):
        foldx_commands = []
        ibex_commands = []
        num_finished = 0

        for i, wt_seq, mt_seq, wt_pdb, pdb_dir, job_name in zip(range(len(wt_seqs)), wt_seqs, mt_seqs, wt_pdbs, pdb_dirs, job_names):
            # prepare job directory
            job_dir = self.output_dir + f'{job_name}/'  # FoldX每一次运行mutation的输出目录
            job_script = job_dir + 'script.sh'  # FoldX每一次运行mutation的脚本
            check_path(job_dir)

            if is_path_exist(os.path.join(job_dir, f'Average_{get_basename(wt_pdb)}.fxout')):
                num_finished += 1
            else:
                data2file(mt_seq, job_dir + 'mutant_sequence.txt')  # write sequence.txt

                # prepare mutant file (individual_list.txt)
                mut_string = format_mutation(wt_seq, mt_seq, seperator=',', end_seperator=';', offset=offset)
                mutant_file = job_dir + 'individual_list.txt'
                data2file(mut_string, mutant_file)

                # prepare shell script for FoldX
                foldx_command = mutation_command.format(self.foldx_exe, self.numberOfRuns, wt_pdb, mutant_file, job_dir, pdb_dir)
                data2file('#!/bin/bash\n\n' + foldx_command, job_script)
                foldx_commands.append(foldx_command)

                # prepare shell script for Ibex
                if self.ibex:
                    num_mut_sites = len(mut_string.split(';'))
                    estimated_run_time = num_mut_sites * time_per_mut_site if num_mut_sites <= 5 else num_mut_sites * (time_per_mut_site + 1)
                    ibex_command = ibex_submit_command.format(f'job_{i}', job_dir, job_dir, estimated_run_time, job_script)
                    ibex_commands.append(ibex_command)

        if self.ibex and len(ibex_commands) > 0:
            ibex_run_file = self.output_dir + 'run.sh'
            data2file('#!/bin/bash\n\n' + '\n'.join(ibex_commands), ibex_run_file)

        print(f'Number of finished before running: {num_finished}/{len(wt_seqs)}')
        return foldx_commands, ibex_commands

    def repair_pdb(self, pdbs, pdb_dir=None, job_names=None):
        # process input pdbs
        if isinstance(pdbs, str):
            pdbs = [pdbs]

        if pdb_dir is None:
            pdb_dirs = [pdb.split(get_basename(pdb, suffix=True))[0] for pdb in pdbs]
            pdbs = [get_basename(pdb, suffix=True) for pdb in pdbs]
        else:
            pdb_dirs = [pdb_dir] * len(pdbs)
            pdbs = [get_basename(pdb, suffix=True) if pdb_dir in pdb else pdb for pdb_dir, pdb in zip(pdb_dirs, pdbs)]

        job_names = [f'job_{i}' for i in range(len(pdbs))] if job_names is None else job_names

        foldx_commands, ibex_commands = self.generate_repair_command(pdbs, pdb_dirs, job_names)
        if 0 < len(foldx_commands) < 10:
            print('\n'.join(foldx_commands))

        if self.ibex:
            results = submit_ibex_jobs(ibex_commands)
        else:
            results = run_script(foldx_commands, max_workers=self.num_cpu)
        return results

    def generate_repair_command(self, pdbs, pdb_dirs, job_names):
        repair_commands = []
        ibex_commands = []
        num_finished = 0

        for i, pdb, pdb_dir, job_name in zip(range(len(pdbs)), pdbs, pdb_dirs, job_names):
            # prepare job directory
            job_dir = self.output_dir + f'{job_name}/'
            job_script = job_dir + 'script.sh'
            check_path(job_dir)

            name, suffix = pdb.split('.')
            if is_path_exist(os.path.join(job_dir, f'{name}_Repair.{suffix}')):
                num_finished += 1
            else:
                # prepare shell script for FoldX
                foldx_command = repair_command.format(self.foldx_exe, pdb, job_dir, pdb_dir)
                data2file('#!/bin/bash\n\n' + foldx_command, job_script)
                repair_commands.append(foldx_command)

                # prepare shell script for Ibex
                if self.ibex:
                    ibex_command = ibex_submit_command.format(f'job_{i}', job_dir, job_dir, 10, job_script)
                    ibex_commands.append(ibex_command)

        if self.ibex and len(ibex_commands) > 0:
            ibex_run_file = self.output_dir + 'run.sh'
            data2file('#!/bin/bash\n\n' + '\n'.join(ibex_commands), ibex_run_file)

        print(f'Number of finished before running: {num_finished}/{len(pdbs)}')
        return repair_commands, ibex_commands
