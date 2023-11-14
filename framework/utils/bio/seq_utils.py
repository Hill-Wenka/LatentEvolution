import Levenshtein
import numpy as np

from framework.utils.bio import bioinfo


def split_kmer(seq, k=6, unique=True, return_type=list):
    '''
    split a seq into overlapped kmers
    '''
    kmers = [seq[i:i + k] for i in range(len(seq) - k + 1)]
    if return_type == list:
        kmers = np.array(kmers)
    if unique:
        kmers = list(set(kmers))
    elif return_type == np.ndarray:
        pass
    else:
        raise RuntimeError(f'No such pre-defined return_type: {return_type}')
    return kmers


def split_seqs(seqs, k=6, unique=True, return_type=list):
    '''
    split a list of seqs into overlapped kmers
    '''
    kmers = []
    for seq in seqs:
        kmers.extend(split_kmer(seq, k, unique))
    if unique:
        kmers = list(set(kmers))
    if return_type == list:
        pass
    elif return_type == np.ndarray:
        kmers = np.array(kmers)
    return kmers


def distance(s1, s2, dist='Levenshtein'):
    '''
    calculate the distance between two sequences
    '''
    if dist == 'Levenshtein':
        d = Levenshtein.distance(s1, s2)
    elif dist == 'Hamming':
        d = sum([bool(a != b) for a, b in zip(s1, s2)])
    else:
        raise RuntimeError(f'No such pre-defined dist: {dist}')
    return d


def get_seq_d_neighbors(seq, d, token='residue', dist='Levenshtein', unique=True):
    # get all neighbors of a seq with distance d through recursion
    if token == 'residue':
        tokens = bioinfo.residues
    elif token == 'nucleotide':
        tokens = bioinfo.nucleotides
    else:
        raise RuntimeError(f'No such pre-defined token: {token}')

    if d == 0:
        return [seq]
    if len(seq) == 1:
        return tokens
    current_neighbor = []
    suffix_neighbors = get_seq_d_neighbors(seq[1:], d, token, dist)
    for suffix in suffix_neighbors:
        if distance(suffix, seq[1:], dist) < d:
            current_neighbor += [x + suffix for x in tokens]
        else:
            current_neighbor.append(seq[0] + suffix)

    if unique:
        current_neighbor = list(set(current_neighbor))
    return current_neighbor


def single_mutate(seq, i, mutate_res_list=None, keep_wt=False):
    '''
    遍历给定的突变氨基酸列表，将原始序列的第i位替换为给定的突变氨基酸
    '''
    if mutate_res_list is None:
        mutate_res_list = bioinfo.residues
    seq = list(seq)
    if keep_wt:
        mutants = [''.join(seq[:i] + [r] + seq[i + 1:]) for r in mutate_res_list]
    else:
        mutants = [''.join(seq[:i] + [r] + seq[i + 1:]) for r in mutate_res_list if r != seq[i]]
    return mutants


def single_mutate_scan(seq, mutate_res_list=None, keep_wt=False):
    '''
    遍历给定的突变氨基酸列表，将原始序列的每一个位点都替换为给定的突变氨基酸
    '''
    if mutate_res_list is None:
        mutate_res_list = bioinfo.residues
    mutants = [single_mutate(seq, i, mutate_res_list, keep_wt) for i in range(len(seq))]
    return mutants


def distance_matrix(seqs, only_upper=True, dist='Levenshtein'):
    '''
    输入序列列表，返回所有序列对之间的距离矩阵
    '''
    matrix = np.zeros([len(seqs), len(seqs)])
    for i, pep1 in enumerate(seqs):
        for j, pep2 in enumerate(seqs):
            if j > i:
                d_ij = distance(pep1, pep2, dist=dist)
                matrix[i][j] = d_ij
                if not only_upper:
                    matrix[j][i] = d_ij
    return matrix


def get_single_mutants(seqs, keep_wt=True):
    mutants = []
    for seq in seqs:
        seq_mutants = np.array(single_mutate_scan(seq, keep_wt=keep_wt)).reshape(-1).tolist()
        mutants.extend(seq_mutants)
    return mutants


# def convert_mutation_string_to_list(mutation_string):

def mutate(sequence, mutation_string=None, seperator=' ', chains=None, positions=None, mutations=None, zero_based=False):
    # sequence: wide-type sequence
    if isinstance(sequence, str):
        sequence = {'A': list(sequence)}
    elif isinstance(sequence, dict):
        sequence = {k: list(v) for k, v in sequence.items()}
    else:
        raise RuntimeError(f'No such pre-defined sequence type: {type(sequence)}')

    if mutation_string is not None:
        assert positions is None and mutations is None
        mutation_list = mutation_string.split(seperator)
        wt_residues, chain_positions, mt_residues = zip(*[(x[0], x[1:-1], x[-1]) for x in mutation_list])
        chains = [x[0] if x[0].isalpha() else 'A' for x in chain_positions]
        positions = [int(x[1:]) if x[0].isalpha() else int(x) for x in chain_positions]
        positions = [x - 1 for x in positions] if not zero_based else positions
    else:
        assert positions is not None and mutations is not None
        chains = ['A'] * len(positions) if chains is None else chains
        positions = [x - 1 for x in positions] if not zero_based else positions
        wt_residues = [sequence[chain][position] for chain, position in zip(chains, positions)]
        mt_residues = mutations

    for chain, wt_res, position, mt_res in zip(chains, wt_residues, positions, mt_residues):
        # print(wt_res, chain, position, mt_res)
        assert sequence[chain][position] == wt_res
        sequence[chain][position] = mt_res

    # print('sequence', sequence)
    sequence = {k: ''.join(v) for k, v in sequence.items()} if len(sequence) > 1 else ''.join(list(sequence.values())[0])
    return sequence


def format_mutation(wt_seq, mt_seq, chian_id='A', offset=0, return_type='string', seperator=' ', end_seperator='', omit_chain=False):
    assert len(wt_seq) == len(mt_seq)
    mut_list = ['{}{}{}{}'.format(wt_seq[i], '' if omit_chain else chian_id, i + 1 + offset, mt_seq[i]) for i in range(len(wt_seq)) if
                wt_seq[i] != mt_seq[i]]
    if return_type == 'string':
        mut_list = seperator.join(mut_list) + end_seperator
    elif return_type == 'list':
        pass
    return mut_list


if __name__ == '__main__':
    arr = get_seq_d_neighbors('ACT', 1, token='nucleotide')
    print(type(arr), len(arr), len(set(arr)))
    print(arr)
    arr = get_seq_d_neighbors('ACT', 2, token='nucleotide')
    print(type(arr), len(arr), len(set(arr)))
    print(arr)
