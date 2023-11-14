from scipy.stats import pearsonr, spearmanr


def pearson_corrcoef(x, y):
    return pearsonr(x, y)[0]


def spearman_corrcoef(x, y):
    return spearmanr(x, y)[0]
