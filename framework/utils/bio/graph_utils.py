from functools import partial

from graphein.protein.config import DSSPConfig, ProteinGraphConfig
from graphein.protein.edges.distance import (add_aromatic_interactions, add_aromatic_sulphur_interactions,
                                             add_backbone_carbonyl_carbonyl_interactions, add_cation_pi_interactions,
                                             add_distance_threshold, add_disulfide_interactions, add_hydrogen_bond_interactions,
                                             add_hydrophobic_interactions, add_ionic_interactions, add_k_nn_edges, add_peptide_bonds,
                                             add_pi_stacking_interactions, add_t_stacking, add_vdw_clashes, add_vdw_interactions)
from graphein.protein.features.nodes import asa, rsa, secondary_structure
from graphein.protein.features.nodes.amino_acid import expasy_protein_scale
from graphein.protein.features.nodes.geometry import add_beta_carbon_vector, add_sequence_neighbour_vector, add_sidechain_vector
from graphein.protein.features.sequence.embeddings import esm_residue_embedding, esm_sequence_embedding
from graphein.protein.graphs import construct_graph

edge_fns = [
    add_peptide_bonds,
    add_hydrophobic_interactions,
    add_disulfide_interactions,
    add_hydrogen_bond_interactions,
    add_ionic_interactions,
    add_aromatic_interactions,
    add_aromatic_sulphur_interactions,
    add_cation_pi_interactions,
    add_vdw_interactions,
    add_vdw_clashes,
    add_pi_stacking_interactions,
    add_t_stacking,
    add_backbone_carbonyl_carbonyl_interactions,
    partial(add_k_nn_edges, k=10, long_interaction_threshold=0),
    partial(add_distance_threshold, long_interaction_threshold=5, threshold=10.)
]

config = ProteinGraphConfig(
    graph_metadata_functions=[secondary_structure, asa, rsa, esm_sequence_embedding, esm_residue_embedding],
    node_metadata_functions=[expasy_protein_scale],
    edge_construction_functions=edge_fns,
    dssp_config=DSSPConfig())


def construct_protein_graph(pdb_code="3eiy", pdb_path=None, config=config):
    if pdb_code:
        graph = construct_graph(config=config, pdb_code=pdb_code)
    elif pdb_path:
        graph = construct_graph(config=config, path=pdb_path)
    else:
        raise ValueError("Must provide either a pdb code or a pdb path")
    add_sidechain_vector(graph)
    add_beta_carbon_vector(graph)
    add_sequence_neighbour_vector(graph)
    return graph


if __name__ == '__main__':
    from framework import paths

    pdb_path = paths.data + 'TPBLA/esmfold/WFL/5_WFL.pdb'
    graph = construct_protein_graph(pdb_path=pdb_path)
