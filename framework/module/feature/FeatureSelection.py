import numpy as np
from scipy.stats import pearsonr
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import chi2
from sklearn.linear_model import LinearRegression

from framework.module.metric.mine import MyMINE


class FeatureSelection():
    def __init__(self, pipline, params):
        super(FeatureSelection, self).__init__()

        self.selectors = []
        self.options = ['variance', 'chi2', 'pearsonr', 'mic', 'rfe']
        for i, s in enumerate(pipline):
            if s not in self.options:
                raise RuntimeError(f'No such selector option: {s}')
            if s == 'variance':
                self.selectors.append(self.variance_select(**params[i]))
            if s == 'chi2':
                self.selectors.append(self.chi2_select(**params[i]))
            if s == 'pearsonr':
                self.selectors.append(self.pearsonr_select(**params[i]))
            if s == 'mic':
                self.selectors.append(self.mic_select(**params[i]))
            if s == 'rfe':
                self.selectors.append(self.rfe_select(**params[i]))

    def variance_select(self, t=0.8):
        self.var_selector = VarianceThreshold(threshold=(t * (1 - t)))
        return self.var_selector

    def chi2_select(self, k):
        self.chi2_selector = SelectKBest(chi2, k=k)
        return self.chi2_selector

    def pearsonr_select(self, k):
        self.pearsonr_selector = SelectKBest(lambda X, Y: np.array(list(map(lambda x: pearsonr(x, Y)[0], X.T))).T, k=k)
        return self.pearsonr_selector

    def mic_select(self, k):
        mine = MyMINE()
        self.mic_selector = SelectKBest(
            lambda X, Y: np.array(list(map(lambda x: mine.mic(x, Y), X.T))).T, k=k
        )
        return self.mic_selector

    def rfe_select(self, k, estimator=None):
        if estimator is None:
            estimator = LinearRegression()
        self.rfe_selector = RFE(estimator=estimator, n_features_to_select=k)
        return self.rfe_selector

    def fit_transform(self, X, Y):
        self.selected_id_list = []
        for selector in self.selectors:
            X = selector.fit_transform(X, Y)
            selected_ids = selector.get_support(True)
            self.selected_id_list.append(selected_ids)
        self.final_features = X

        self.final_feature_ids = np.arange(X.shape[1])
        for id in self.selected_id_list:
            self.final_feature_ids = self.final_feature_ids[id]
        return self.final_features
