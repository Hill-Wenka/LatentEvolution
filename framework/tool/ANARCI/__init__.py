import os

import numpy as np

from framework import paths
from framework.utils.data.dict_utils import merge_dicts
from framework.utils.file.path_utils import check_path, list_files, is_path_exist
from framework.utils.file.read_utils import read_fasta
from framework.utils.file.write_utils import write_fasta
from framework.utils.parallel.parallel_utils import asyn_parallel

'''
Antibody Numbering and CDR Annotation
Use Chothia numbering by default for all CDR annotation schemes

IMGT + IMGT
CDR1: 27-38
CDR2: 56-65
CDR3: 105-117

Chothia/Kabat/Martin + IMGT
CDRH1: 26-33
CDRH2: 51-57 (原文写错了：51-56)
CDRH3: 93-102
CDRL1: 27-32
CDRL2: 50-52 (原文写错了：50-51)
CDRL3: 89-97

Chothia/Kabat/Martin + Kabat
CDRH1: 31-35
CDRH2: 56-65
CDRH3: 95-102
CDRL1: 24-34
CDRL2: 50-56
CDRL3: 89-97

Chothia/Kabat/Martin + Chothia
CDRH1: 26-32
CDRH2: 52-56
CDRH3: 95-102
CDRL1: 24-34
CDRL2: 50-56
CDRL3: 89-97

Chothia/Martin + Contact
CDRH1: 30-35
CDRH2: 47-58
CDRH3: 93-101
CDRL1: 30-36
CDRL2: 46-55
CDRL3: 89-96
'''

antobody_regions = {'H': ['FRH1', 'CDRH1', 'FRH2', 'CDRH2', 'FRH3', 'CDRH3', 'FRH4'],
                    'L': ['FRL1', 'CDRL1', 'FRL2', 'CDRL2', 'FRL3', 'CDRL3', 'FRL4']}
antobody_regions_index = {'FRH1': 1, 'CDRH1': 2, 'FRH2': 3, 'CDRH2': 4, 'FRH3': 5, 'CDRH3': 6, 'FRH4': 7,
                          'FRL1': 8, 'CDRL1': 9, 'FRL2': 10, 'CDRL2': 11, 'FRL3': 12, 'CDRL3': 13, 'FRL4': 14}
annotaion_scheme = ['imgt', 'chothia', 'kabat', 'contact']
antobody_annotation = {scheme: {'H': {}, 'L': {}} for scheme in annotaion_scheme}
for x in ['index', 'string']:
    # IMGT
    scheme = 'imgt'
    antobody_annotation[scheme]['H'][x] = np.zeros(128, dtype=int) if x == 'index' else np.array(['_____' for i in range(128)])
    antobody_annotation[scheme]['H'][x][:26 - 1] = 'FRH1' if x == 'string' else 1
    antobody_annotation[scheme]['H'][x][26 - 1:33] = 'CDRH1' if x == 'string' else 2
    antobody_annotation[scheme]['H'][x][33:51 - 1] = 'FRH2' if x == 'string' else 3
    antobody_annotation[scheme]['H'][x][51 - 1:57] = 'CDRH2' if x == 'string' else 4
    antobody_annotation[scheme]['H'][x][57:93 - 1] = 'FRH3' if x == 'string' else 5
    antobody_annotation[scheme]['H'][x][93 - 1:102] = 'CDRH3' if x == 'string' else 6
    antobody_annotation[scheme]['H'][x][102:128] = 'FRH4' if x == 'string' else 7
    antobody_annotation[scheme]['L'][x] = np.zeros(128, dtype=int) if x == 'index' else np.array(['_____' for i in range(128)])
    antobody_annotation[scheme]['L'][x][:27 - 1] = 'FRL1' if x == 'string' else 8
    antobody_annotation[scheme]['L'][x][27 - 1:32] = 'CDRL1' if x == 'string' else 9
    antobody_annotation[scheme]['L'][x][32:50 - 1] = 'FRL2' if x == 'string' else 10
    antobody_annotation[scheme]['L'][x][50 - 1:52] = 'CDRL2' if x == 'string' else 11
    antobody_annotation[scheme]['L'][x][52:89 - 1] = 'FRL3' if x == 'string' else 12
    antobody_annotation[scheme]['L'][x][89 - 1:97] = 'CDRL3' if x == 'string' else 13
    antobody_annotation[scheme]['L'][x][97:128] = 'FRL4' if x == 'string' else 14

    # Kabat
    scheme = 'kabat'
    antobody_annotation[scheme]['H'][x] = np.zeros(128, dtype=int) if x == 'index' else np.array(['_____' for i in range(128)])
    antobody_annotation[scheme]['H'][x][:31 - 1] = 'FRH1' if x == 'string' else 1
    antobody_annotation[scheme]['H'][x][31 - 1:35] = 'CDRH1' if x == 'string' else 2
    antobody_annotation[scheme]['H'][x][35:56 - 1] = 'FRH2' if x == 'string' else 3
    antobody_annotation[scheme]['H'][x][56 - 1:65] = 'CDRH2' if x == 'string' else 4
    antobody_annotation[scheme]['H'][x][65:95 - 1] = 'FRH3' if x == 'string' else 5
    antobody_annotation[scheme]['H'][x][95 - 1:102] = 'CDRH3' if x == 'string' else 6
    antobody_annotation[scheme]['H'][x][102:128] = 'FRH4' if x == 'string' else 7
    antobody_annotation[scheme]['L'][x] = np.zeros(128, dtype=int) if x == 'index' else np.array(['_____' for i in range(128)])
    antobody_annotation[scheme]['L'][x][:24 - 1] = 'FRL1' if x == 'string' else 8
    antobody_annotation[scheme]['L'][x][24 - 1:34] = 'CDRL1' if x == 'string' else 9
    antobody_annotation[scheme]['L'][x][34:50 - 1] = 'FRL2' if x == 'string' else 10
    antobody_annotation[scheme]['L'][x][50 - 1:56] = 'CDRL2' if x == 'string' else 11
    antobody_annotation[scheme]['L'][x][56:89 - 1] = 'FRL3' if x == 'string' else 12
    antobody_annotation[scheme]['L'][x][89 - 1:97] = 'CDRL3' if x == 'string' else 13
    antobody_annotation[scheme]['L'][x][97:128] = 'FRL4' if x == 'string' else 14

    # Chothia
    scheme = 'chothia'
    antobody_annotation[scheme]['H'][x] = np.zeros(128, dtype=int) if x == 'index' else np.array(['_____' for i in range(128)])
    antobody_annotation[scheme]['H'][x][:26 - 1] = 'FRH1' if x == 'string' else 1
    antobody_annotation[scheme]['H'][x][26 - 1:32] = 'CDRH1' if x == 'string' else 2
    antobody_annotation[scheme]['H'][x][32:52 - 1] = 'FRH2' if x == 'string' else 3
    antobody_annotation[scheme]['H'][x][52 - 1:56] = 'CDRH2' if x == 'string' else 4
    antobody_annotation[scheme]['H'][x][56:95 - 1] = 'FRH3' if x == 'string' else 5
    antobody_annotation[scheme]['H'][x][95 - 1:102] = 'CDRH3' if x == 'string' else 6
    antobody_annotation[scheme]['H'][x][102:128] = 'FRH4' if x == 'string' else 7
    antobody_annotation[scheme]['L'][x] = np.zeros(128, dtype=int) if x == 'index' else np.array(['_____' for i in range(128)])
    antobody_annotation[scheme]['L'][x][:24 - 1] = 'FRL1' if x == 'string' else 8
    antobody_annotation[scheme]['L'][x][24 - 1:34] = 'CDRL1' if x == 'string' else 9
    antobody_annotation[scheme]['L'][x][34:50 - 1] = 'FRL2' if x == 'string' else 10
    antobody_annotation[scheme]['L'][x][50 - 1:56] = 'CDRL2' if x == 'string' else 11
    antobody_annotation[scheme]['L'][x][56:89 - 1] = 'FRL3' if x == 'string' else 12
    antobody_annotation[scheme]['L'][x][89 - 1:97] = 'CDRL3' if x == 'string' else 13
    antobody_annotation[scheme]['L'][x][97:128] = 'FRL4' if x == 'string' else 14

    # Contact
    scheme = 'contact'
    antobody_annotation[scheme]['H'][x] = np.zeros(128, dtype=int) if x == 'index' else np.array(['_____' for i in range(128)])
    antobody_annotation[scheme]['H'][x][:30 - 1] = 'FRH1' if x == 'string' else 1
    antobody_annotation[scheme]['H'][x][30 - 1:35] = 'CDRH1' if x == 'string' else 2
    antobody_annotation[scheme]['H'][x][35:47 - 1] = 'FRH2' if x == 'string' else 3
    antobody_annotation[scheme]['H'][x][47 - 1:58] = 'CDRH2' if x == 'string' else 4
    antobody_annotation[scheme]['H'][x][58:93 - 1] = 'FRH3' if x == 'string' else 5
    antobody_annotation[scheme]['H'][x][93 - 1:101] = 'CDRH3' if x == 'string' else 6
    antobody_annotation[scheme]['H'][x][101:128] = 'FRH4' if x == 'string' else 7
    antobody_annotation[scheme]['L'][x] = np.zeros(128, dtype=int) if x == 'index' else np.array(['_____' for i in range(128)])
    antobody_annotation[scheme]['L'][x][:30 - 1] = 'FRL1' if x == 'string' else 8
    antobody_annotation[scheme]['L'][x][30 - 1:36] = 'CDRL1' if x == 'string' else 9
    antobody_annotation[scheme]['L'][x][36:46 - 1] = 'FRL2' if x == 'string' else 10
    antobody_annotation[scheme]['L'][x][46 - 1:55] = 'CDRL2' if x == 'string' else 11
    antobody_annotation[scheme]['L'][x][55:89 - 1] = 'FRL3' if x == 'string' else 12
    antobody_annotation[scheme]['L'][x][89 - 1:96] = 'CDRL3' if x == 'string' else 13
    antobody_annotation[scheme]['L'][x][96:128] = 'FRL4' if x == 'string' else 14

command = 'ANARCI -s chothia -i {} -o {}'


def run_script(seq, output_file, overwrite=True):
    if not is_path_exist(output_file):
        check_path(output_file)
        result = os.system(command.format(seq, output_file))
    else:
        result = os.system(command.format(seq, output_file)) if overwrite else 0
    return result


def read_result(output_file):
    with open(output_file, 'r') as f:
        results = f.read()
    return results.split('\n')


def write_temp_fasta(input, output_dir, temp_path, linker):
    if '.fasta' in input:
        seqs, heads = read_fasta(input)
    else:
        seqs = [input] if type(input) is str else input
        heads = [f'temp_{i}' for i in range(len(seqs))]

    temp_seqs, temp_heads = [], []
    seq_type = 'scFv' if linker is None else '_VH'
    for i, head in enumerate(heads):
        if not is_path_exist(os.path.join(output_dir, f'{head}_{seq_type}.txt')):
            temp_seqs.append(seqs[i])
            temp_heads.append(head)
    check_path(temp_path)
    write_fasta(temp_path, temp_seqs, temp_heads)
    max_len = None if len(temp_seqs) == 0 else max([len(s) for s in temp_seqs])
    return len(seqs) - len(temp_seqs), len(seqs), max_len


def annote_cdrs(numbering, seq, scheme='imgt', mode='string'):
    if len(numbering) > 0:
        species = numbering[0][0]
        numbering = np.array([int(row[1:6]) - 1 for row in numbering if row[-1] != '-'])
        annotation = antobody_annotation[scheme][species][mode][numbering]
        seq_array = np.array(list(seq))
        regions = {region: ''.join(seq_array[annotation == (region if mode == 'string' else antobody_regions_index[region])])
                   for region in antobody_regions[species]}
    else:
        annotation = np.array([])
        regions = {}
    return annotation, regions


def post_process_VH(VH, raw_VH, VH_annot, VH_regions, mode):
    if VH in raw_VH:
        prefix, suffix = raw_VH.split(VH)
        FRH1 = 'FRH1' if mode == 'string' else antobody_regions_index['FRH1']
        CDRH3 = 'CDRH3' if mode == 'string' else antobody_regions_index['CDRH3']
        FRH4 = 'FRH4' if mode == 'string' else antobody_regions_index['FRH4']

        VH_regions['FRH1'] = prefix + VH_regions['FRH1']
        if len(VH_regions['FRH4']) > 0:
            VH_regions['FRH4'] = VH_regions['FRH4'] + suffix
            VH_annot = np.concatenate([np.array([FRH1] * len(prefix)), VH_annot, np.array([FRH4] * len(suffix))])
        else:
            VH_regions['CDRH3'] = VH_regions['CDRH3'] + suffix
            VH_annot = np.concatenate([np.array([FRH1] * len(prefix)), VH_annot, np.array([CDRH3] * len(suffix))])
    else:
        VH_annot, VH_regions = None, None
    return VH_annot, VH_regions


def post_process_VL(VL, raw_VL, VL_annot, VL_regions, mode):
    if VL in raw_VL:
        prefix, suffix = raw_VL.split(VL)
        FRL1 = 'FRL1' if mode == 'string' else antobody_regions_index['FRL1']
        CDRL3 = 'CDRL3' if mode == 'string' else antobody_regions_index['CDRL3']
        FRL4 = 'FRL4' if mode == 'string' else antobody_regions_index['FRL4']

        VL_regions['FRL1'] = prefix + VL_regions['FRL1']
        if len(VL_regions['FRL4']) > 0:
            VL_regions['FRL4'] = VL_regions['FRL4'] + suffix
            VL_annot = np.concatenate([np.array([FRL1] * len(prefix)), VL_annot, np.array([FRL4] * len(suffix))])
        else:
            VL_regions['CDRL3'] = VL_regions['CDRL3'] + suffix
            VL_annot = np.concatenate([np.array([FRL1] * len(prefix)), VL_annot, np.array([CDRL3] * len(suffix))])
    else:
        VL_annot, VL_regions = None, None
    return VL_annot, VL_regions


def post_process(VH, VL, raw_VH, raw_VL, VH_annot, linker_annot, VL_annot, VH_regions, VL_regions, mode):
    if len(VH) > 0 and len(VL) > 0:
        # scFv, both VH and VL are found in the raw sequence
        VH_annot, VH_regions = post_process_VH(VH, raw_VH, VH_annot, VH_regions, mode)
        VL_annot, VL_regions = post_process_VL(VL, raw_VL, VL_annot, VL_regions, mode)
        annotations = np.concatenate([VH_annot, linker_annot, VL_annot])
        regions = {**VH_regions, **VL_regions}
    elif len(VH) > 0:
        # only VH is found in the raw sequence
        VH_annot, VH_regions = post_process_VH(VH, raw_VH, VH_annot, VH_regions, mode)
        annotations = VH_annot
        regions = VH_regions
    elif len(VL) > 0:
        # only VL is found in the raw sequence
        VL_annot, VL_regions = post_process_VL(VL, raw_VL, VL_annot, VL_regions, mode)
        annotations = VL_annot
        regions = VL_regions
    else:
        # neither VH nor VL is found in the raw sequence
        annotations = None
        regions = None
    return annotations, regions


def parse_annotation(seq, head, output_dir, linker=None, scheme='imgt', mode='string'):
    if linker is not None:
        VH_data_path = os.path.join(output_dir, f'{head}_VH.txt')
        VL_data_path = os.path.join(output_dir, f'{head}_VL.txt')
        if is_path_exist(VH_data_path) and is_path_exist(VL_data_path):
            VH_results = read_result(VH_data_path)
            VH_numbers = [row for row in VH_results if len(row) > 0 and 'H' == row[0]]
            VH = [row[-1] for row in VH_results if len(row) > 0 and 'H' == row[0] and row[-1] != '-']
            VH_annot, VH_regions = annote_cdrs(VH_numbers, VH, scheme, mode)

            VL_results = read_result(VL_data_path)
            VL_numbers = [row for row in VL_results if len(row) > 0 and 'L' == row[0]]
            VL = [row[-1] for row in VL_results if len(row) > 0 and 'L' == row[0] and row[-1] != '-']
            VL_annot, VL_regions = annote_cdrs(VL_numbers, VL, scheme, mode)

            linker_annot = ['Linker'] * len(linker) if mode == 'string' else [0] * len(linker)
            annotations = np.concatenate([VH_annot, linker_annot, VL_annot])
        else:
            linker_annot = []
            annotations = None
            VH_regions, VL_regions = {}, {}
    else:
        scFv_data_path = os.path.join(output_dir, f'{head}_scFv.txt')
        if is_path_exist(scFv_data_path):
            seq_results = read_result(scFv_data_path)

            VH_numbers = [row for row in seq_results if len(row) > 0 and 'H' == row[0]]
            VH = [row[-1] for row in seq_results if len(row) > 0 and 'H' == row[0] and row[-1] != '-']
            VH_annot, VH_regions = annote_cdrs(VH_numbers, VH, scheme, mode)

            VL_numbers = [row for row in seq_results if len(row) > 0 and 'L' == row[0]]
            VL = [row[-1] for row in seq_results if len(row) > 0 and 'L' == row[0] and row[-1] != '-']
            VL_annot, VL_regions = annote_cdrs(VL_numbers, VL, scheme, mode)

            if len(VH_annot) > 0 and len(VL_annot) > 0:
                len_linker = len(seq) - len(VH) - len(VL)
                linker_annot = ['Linker'] * len_linker if mode == 'string' else [0] * len_linker
                annotations = np.concatenate([VH_annot, linker_annot, VL_annot])
            elif len(VH_annot) > 0:
                linker_annot = []
                annotations = VH_annot
            elif len(VL_annot) > 0:
                linker_annot = []
                annotations = VL_annot
            else:
                annotations = None
        else:
            linker_annot = []
            annotations = None
            VH_regions, VL_regions = {}, {}

    regions = {**VH_regions, **VL_regions}

    if annotations is not None and len(annotations) != len(seq):
        VH = ''.join(VH) if VH != '' else ''
        VL = ''.join(VL) if VL != '' else ''
        if linker is not None:
            linker = linker
        else:
            if VH == '' or VL == '':
                linker = ''
            else:
                linker = seq.split(VH)[1].split(VL)[0]
        if linker == '':
            if VH == '':
                raw_VH, raw_VL = '', seq
            elif VL == '':
                raw_VH, raw_VL = seq, ''
            else:
                raise RuntimeError('VH and VL not found in raw sequence')
        else:
            raw_VH, raw_VL = seq.split(linker)

        annotations, regions = post_process(VH, VL, raw_VH, raw_VL, VH_annot, linker_annot, VL_annot, VH_regions, VL_regions, mode)
        if annotations is not None:
            assert len(annotations) == len(seq)
        else:
            print(f'Annotation of head[{head}] is invalid.')

    return annotations, regions


def annotate_func(sequence, head, output_dir, linker, overwrite):
    if linker is not None:
        VH, VL = sequence.split(linker)
        output_file = os.path.join(output_dir, f'{head}_VH.txt')
        VH_results = run_script(VH, output_file, overwrite)
        output_file = os.path.join(output_dir, f'{head}_VL.txt')
        VL_results = run_script(VL, output_file, overwrite)
        run_results = VH_results + VL_results
    else:
        output_file = os.path.join(output_dir, f'{head}_scFv.txt')
        run_results = run_script(sequence, output_file, overwrite)
    return {head: run_results}


def parallel_parse_func(seq, head, output_dir, linker, scheme, mode):
    annotation = parse_annotation(seq, head, output_dir, linker, scheme, mode)[0]
    return {head: annotation}


class ANARCIWrapper:
    def __init__(
            self,
            output_dir=None,
            temp_path=paths.root,
            linker=None,
            scheme='imgt',
            mode='string',
            overwrite=True,
            **kwargs
    ):
        self.output_dir = os.path.join(output_dir, scheme, '')
        self.temp_path = os.path.join(temp_path, 'temp', 'anarci_unfinished_list.fasta')
        self.linker = linker
        self.scheme = scheme
        self.mode = mode
        self.overwrite = overwrite
        check_path(self.output_dir)

    def annotate_sequence(self, input):
        num_finished, num_total, max_len = write_temp_fasta(input, self.output_dir, self.temp_path, self.linker)
        seqs, heads = read_fasta(self.temp_path)
        print(f'Number of finished before running: {num_finished}/{num_total}, max_len: {max_len}')
        results = asyn_parallel(annotate_func, [[seq, head, self.output_dir, self.linker, self.overwrite] for seq, head in zip(seqs, heads)],
                                desc='Annotating sequences')
        return results

    def check_data(self, fasta):
        seqs, heads = read_fasta(fasta)
        seq_type = 'scFv' if self.linker is None else '_VH'
        failed_records = [head for head in heads if not is_path_exist(os.path.join(self.output_dir, f'{head}_{seq_type}.txt'))]
        return failed_records

    def list_data(self, absolute=True):
        return list_files(self.output_dir, absolute)

    def load_data(self, fasta):
        seqs, heads = read_fasta(fasta)
        annotations = asyn_parallel(parallel_parse_func,
                                    [(seq, head, self.output_dir, self.linker, self.scheme, self.mode) for seq, head in zip(seqs, heads)],
                                    desc='Loading ANARCI data')
        return merge_dicts(annotations)

    def get_annotation(self, sequence, linker=None, overwrite=True):
        linker = self.linker if linker is None else linker
        overwrite = self.overwrite if overwrite is None else overwrite
        annotate_func(sequence, 'temp', paths.temp, linker, overwrite)
        return parse_annotation(sequence, 'temp', paths.temp, linker, self.scheme, self.mode)
