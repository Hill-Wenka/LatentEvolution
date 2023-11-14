residues = ['G', 'A', 'V', 'L', 'I', 'F', 'W', 'Y', 'D', 'N', 'E', 'K', 'Q', 'M', 'S', 'T', 'C', 'P', 'H', 'R']
special_residues = ['X', 'B', 'U', 'Z', 'O']
residue_alphabet = residues + special_residues
nucleotides = ['A', 'T', 'C', 'G']

# 这一组是score最高的残基，都是疏水性残基 (with hydrophobic side chain)
# 占了所有疏水性残基的7/8，剩下的一个是'A'，因为疏水性较弱且score较低暂不考虑
high_agg_res = ['F', 'I', 'V', 'Y', 'W', 'L', 'M']  # num = 7

# 这一组是score处于中间的残基
# 其中'S, T, N, Q'是极性不带电残基
middle_agg_res = ['S', 'T', 'N', 'Q', 'C', 'G', 'P', 'A']  # num = 8

# 这一组是score最低的残基，都是带电残基，'D, E'带负电，'R, H, K'带正电
low_agg_res = ['D', 'E', 'R', 'H', 'K']  # num = 5

# This group is used in ANuPP paper
hydrophobic_residues = ['G', 'A', 'C', 'Y', 'V', 'L', 'I', 'M', 'F', 'W']

scFv_linker = 'GGGSSGGGGSGGGGGA'
