import os

import biotite.sequence as seq
import biotite.structure as struc
import biotite.structure.io as strucio
import numpy as np
import torch

from framework import paths
from framework.utils.file.path_utils import get_basename, check_path
from framework.utils.file.path_utils import is_path_exist

Gly_X_Gly_MaxASA = {
    'ALA': 129.0,
    'ARG': 274.0,
    'ASN': 195.0,
    'ASP': 193.0,
    'CYS': 167.0,
    'GLN': 223.0,
    'GLU': 225.0,
    'GLY': 104.0,
    'HIS': 224.0,
    'ILE': 197.0,
    'LEU': 201.0,
    'LYS': 236.0,
    'MET': 224.0,
    'PHE': 240.0,
    'PRO': 159.0,
    'SER': 155.0,
    'THR': 172.0,
    'TRP': 285.0,
    'TYR': 263.0,
    'VAL': 174.0,
}


def get_atom_array(file_path, clean=True, cache_dir=None):
    cache_dir = os.path.join(paths.cache, 'atom_array', '') if cache_dir is None else cache_dir
    cache_file = os.path.join(cache_dir, f'{get_basename(file_path)}.npy')
    check_path(cache_file)

    if is_path_exist(cache_file):
        atom_array = torch.load(cache_file)
    else:
        atom_array = strucio.load_structure(file_path)
        atom_array = clean_atom_array(atom_array) if clean else atom_array
        torch.save(atom_array, cache_file)
    return atom_array


def write_atom_array(atom_array, file_path):
    strucio.save_structure(file_path, atom_array)


def write_prediction(atom_array, prediction, file_path):
    atom_array.set_annotation('b_factor', prediction)
    write_atom_array(atom_array, file_path)


def clean_atom_array(atom_array):
    return atom_array[atom_array.hetero == False]


def atom_array_to_sequence(atom_array, check=True):
    try:
        ids, res_names = struc.get_residues(atom_array)
        if check:
            residue_list = [r for r in res_names if r not in seq.ProteinSequence._dict_3to1.keys()]
            if len(residue_list) > 0:
                raise Warning(f"Residues {residue_list} are not in the 20 standard amino acids")
        convert_seq = ''.join([seq.ProteinSequence.convert_letter_3to1(r) for r in res_names if r in seq.ProteinSequence._dict_3to1.keys()])
    except:
        convert_seq = None
    return convert_seq


def compute_adjacency_matrix(atom_array, threshold=8):
    ca = atom_array[atom_array.atom_name == "CA"]  # Filter only CA atoms
    cell_list = struc.CellList(ca, cell_size=threshold)  # Create cell list of the CA atom array for efficient measurement of adjacency
    adjacency_matrix = cell_list.create_adjacency_matrix(threshold)  # default 8 Angstrom adjacency threshold
    return adjacency_matrix


def compute_distance_matrix(atom_array):
    ca = atom_array[atom_array.atom_name == "CA"]  # Filter only CA atoms
    distance_matrix = np.array([struc.distance(c, ca) for c in ca])
    return distance_matrix


def construct_graph(atom_array, threshold=8):
    # threshold: adjacency threshold, if None, use distance matrix
    node = np.array(list(atom_array_to_sequence(atom_array)))
    edge = compute_adjacency_matrix(atom_array, threshold).astype(int)
    distance = compute_distance_matrix(atom_array)
    return node, edge, distance


def get_neighbors(atom_coord, atom_array, cell_list=None, cell_size=5, radius=8.0, near_residues=True):
    # atom_coord: (N, 3), for example, np.array([1, 2, 3])
    # clean: remove atoms and residues not in the protein sequence
    atom_coord = np.array(atom_coord) if isinstance(atom_coord, list) else atom_coord
    cell_list = struc.CellList(atom_array, cell_size=cell_size) if cell_list is None else cell_list
    atom_indices = cell_list.get_atoms(atom_coord, radius=radius)
    near_atoms = atom_array[atom_indices]

    if near_residues:
        residue_indices, near_residue_names = struc.get_residues(near_atoms)
        near_residue_atoms = None
        for x in [atom_array[atom_array.res_id == res_id] for res_id in set(residue_indices)]:
            if near_residue_atoms is None:
                near_residue_atoms = x
            else:
                near_residue_atoms = near_residue_atoms + x
        near_residue_c_atoms = near_residue_atoms[near_residue_atoms.atom_name == "CA"]

        return atom_indices, near_atoms, residue_indices, near_residue_atoms, near_residue_c_atoms
    else:
        return atom_indices, near_atoms


def compute_sasa(structure, level='residue', mode='SASA', **params):
    if type(structure) == str:
        atom_array = strucio.load_structure(structure)
    else:
        atom_array = structure

    atom_sasa = struc.sasa(atom_array, **params)  # compute atom-level SASA
    if level == 'residue':
        res_sasa = struc.apply_residue_wise(atom_array, atom_sasa, np.sum)  # compute residue-level SASA
        if mode == 'SASA':
            return res_sasa
        elif mode == 'RSA':
            res_ids, res_names = struc.get_residues(atom_array)  # achieve residue list
            maxASA = np.array([Gly_X_Gly_MaxASA[r] for r in res_names])  # compute MaxASA
            res_rsa = res_sasa / maxASA
            return res_rsa
        else:
            raise RuntimeError(f'No such pre-defined mode: {mode}')
    elif level == 'atom':
        if mode == 'SASA':
            return atom_sasa
        elif mode == 'RSA':
            raise RuntimeError('Atom level not support RSA!')
        else:
            raise RuntimeError(f'No such pre-defined mode: {mode}')
    else:
        raise RuntimeError(f'No such pre-defined level: {level}')
