import os
import py3Dmol
import glob


# from colabfold.colabfold import plot_plddt_legend


def load_multimer_pdb(dir, jobname, result_dir='output'):
    # path = os.path.join(dir, jobname, result_dir)
    msa_mode = 'single_sequence'
    use_amber = False
    rank_num = 1  # @param ["1", "2", "3", "4", "5"] {type:"raw"}
    jobname_prefix = '.custom' if msa_mode == 'custom' else ''
    if use_amber:
        pdb_filename = f'{jobname}_relaxed_rank_{rank_num}_model_*.pdb'
    else:
        pdb_filename = f'{jobname}_unrelaxed_rank_{rank_num}_model_*.pdb'
    pdb_filename = os.path.join(dir, jobname, result_dir, jobname_prefix, pdb_filename)
    return pdb_filename


def show_pdb(pdb_filename, show_sidechains=True, show_mainchains=True, color="lDDT"):
    pdb_file = glob.glob(pdb_filename)
    view = py3Dmol.view(js='https://3dmol.org/build/3Dmol.js', )
    view.addModel(open(pdb_file[0], 'r').read(), 'pdb')
    
    if color == 'lDDT':
        view.setStyle({'cartoon': {'colorscheme': {'prop': 'b', 'gradient': 'roygb', 'min': 50, 'max': 90}}})
    elif color == 'rainbow':
        view.setStyle({'cartoon': {'color': 'spectrum'}})
    if show_sidechains:
        BB = ['C', 'O', 'N']
        view.addStyle(
                {'and': [{'resn': ["GLY", "PRO"], 'invert': True}, {'atom': BB, 'invert': True}]},
                {'stick': {'colorscheme': f"WhiteCarbon", 'radius': 0.3}}
                )
        view.addStyle(
                {'and': [{'resn': "GLY"}, {'atom': 'CA'}]},
                {'sphere': {'colorscheme': f"WhiteCarbon", 'radius': 0.3}}
                )
        view.addStyle(
                {'and': [{'resn': "PRO"}, {'atom': ['C', 'O'], 'invert': True}]},
                {'stick': {'colorscheme': f"WhiteCarbon", 'radius': 0.3}}
                )
    if show_mainchains:
        BB = ['C', 'O', 'N', 'CA']
        view.addStyle({'atom': BB}, {'stick': {'colorscheme': f"WhiteCarbon", 'radius': 0.3}})
    
    view.zoomTo()
    # if color == "lDDT":
    #     plot_plddt_legend().show()
    return view
