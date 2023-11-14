import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm.notebook import tqdm

from framework.module.metric import bin_cls
from framework.module.sklearn import Classifier, Regressor
from framework.utils.parallel.parallel_utils import asyn_parallel


class SKTrainer():
    def __init__(self, model, params=None):
        super(SKTrainer, self).__init__()
        if params is None:
            params = {}
        self.model = model
        self.ML_model = self.init_model(model, params)
        self.train_X = None
        self.train_Y = None
        self.test_X = None
        self.test_Y = None

    def init_model(self, model, params):
        if model in Classifier.classifiers:
            ML_model = Classifier.Classifier(model, params)
        elif model in Regressor.regressors:
            ML_model = Regressor.Regressor(model, params)
        else:
            raise RuntimeError(f'No such pre-defined ML model: {model}')
        return ML_model

    def set_dataset(self, train_X, train_Y, test_X=None, test_Y=None, normalization=None):
        train_X = np.array(train_X)
        train_Y = np.array(train_Y)
        test_X = np.array(test_X) if test_X is not None else None
        test_Y = np.array(test_Y) if test_Y is not None else None
        assert len(train_X) == len(train_Y)
        if test_X is not None or test_Y is not None:
            assert len(test_X) == len(test_Y)
            assert train_X.shape[-1] == test_X.shape[-1]

        if normalization is not None:
            if normalization == 'MinMax':
                scalar = MinMaxScaler()
            elif normalization == 'Standard':
                scalar = StandardScaler()
            else:
                raise RuntimeError(f'No such pre-defined normalization method: {normalization}')
            train_X = scalar.fit_transform(train_X)
            test_X = scalar.fit_transform(test_X)

        self.train_X = train_X
        self.train_Y = train_Y
        self.test_X = test_X
        self.test_Y = test_Y

    def check_dataset(self, require_test_data=True):
        assert self.train_X is not None, 'train_X is None, please set dataset'
        if require_test_data:
            assert self.test_X is not None, 'test_X is None, please set dataset'

    def train_test(self, pred_mode='score', task='classification'):
        self.check_dataset(require_test_data=True)
        self.ML_model.fit(self.train_X, self.train_Y)
        if task == 'classification':
            if pred_mode == 'score':
                train_pred = self.ML_model.predict_proba(self.train_X)
                test_pred = self.ML_model.predict_proba(self.test_X)
            elif pred_mode == 'label':
                train_pred = self.ML_model.predict(self.train_X)
                test_pred = self.ML_model.predict(self.test_X)
            else:
                raise RuntimeError(f'No such pre-defined pred_mode: {self.pred_mode}')

            train_metrics = bin_cls.get_functional_torch_metrics(train_pred, self.train_Y, softmax=False)
            test_metrics = bin_cls.get_functional_torch_metrics(test_pred, self.test_Y, softmax=False)
        elif task == 'regression':
            train_pred = self.ML_model.predict(self.train_X)
            test_pred = self.ML_model.predict(self.test_X)

            train_metrics = {
                'spearmanr': spearmanr(train_pred, self.train_Y)[0],
                'pearsonr': pearsonr(train_pred, self.train_Y)[0]
            }
            test_metrics = {
                'spearmanr': spearmanr(test_pred, self.test_Y)[0],
                'pearsonr': pearsonr(test_pred, self.test_Y)[0]
            }
        else:
            raise RuntimeError(f'No such pre-defined task: {task}')

        self.train_pred = train_pred
        self.test_pred = test_pred
        return train_metrics, test_metrics

    def parallel_loocv_func(self, train_ids, valid_ids):
        fold_train_X, fold_valid_X = self.train_X[train_ids], self.train_X[valid_ids]
        fold_train_Y, fold_valid_Y = self.train_Y[train_ids], self.train_Y[valid_ids]
        self.ML_model.fit(fold_train_X, fold_train_Y)
        if self.pred_mode == 'score':
            pred_fold_valid_Y = self.ML_model.predict_proba(fold_valid_X)
        elif self.pred_mode == 'label':
            pred_fold_valid_Y = self.ML_model.predict(fold_valid_X)
        else:
            raise RuntimeError(f'No such pre-defined pred_mode: {self.pred_mode}')
        return {'pred': pred_fold_valid_Y, 'label': fold_valid_Y}

    def loocv(self, pred_mode='score', task='classification'):
        self.check_dataset(require_test_data=False)
        loocv = LeaveOneOut().split(range(len(self.train_X)))
        groups = [group for group in loocv]
        self.pred_mode = pred_mode
        results = asyn_parallel(self.parallel_loocv_func, groups, cpu_num=20)
        loocv_valid_preds = [fold['pred'] for fold in results]
        loocv_valid_labels = [fold['label'] for fold in results]
        loocv_valid_preds = np.concatenate(loocv_valid_preds, axis=0)
        loocv_valid_labels = np.concatenate(loocv_valid_labels, axis=0)
        if task == 'classification':
            loocv_metrics = bin_cls.get_functional_torch_metrics(loocv_valid_preds, loocv_valid_labels, softmax=False)
        elif task == 'regression':
            loocv_metrics = {
                'spearmanr': spearmanr(loocv_valid_preds, loocv_valid_labels)[0],
                'pearsonr': pearsonr(loocv_valid_preds, loocv_valid_labels)[0]
            }
        else:
            raise RuntimeError(f'No such pre-defined task: {task}')

        self.loocv_valid_preds = loocv_valid_preds
        self.loocv_valid_labels = loocv_valid_labels
        return loocv_metrics

    def cross_val(self, model=None, mode='StratifiedKFold', k=10, scoring='roc_auc'):
        self.check_dataset(require_test_data=False)
        if mode == 'LOOCV':
            cv = LeaveOneOut()
        elif mode == 'KFold':
            cv = KFold(n_splits=k)
        elif mode == 'StratifiedKFold':
            cv = StratifiedKFold(n_splits=k)
        else:
            raise RuntimeError(f'No such pre-defined mode: {mode}')
        model = self.ML_model.model if model is None else model.model
        result = cross_val_score(model, self.train_X, self.train_Y, cv=cv, scoring=scoring)
        return result.mean()

    def parallel_search_pararms_func(self, params):
        print('params:', type(params), params)
        ML_model = self.init_model(self.model, params)
        metric = self.cross_val(ML_model, self.mode, self.k, self.scoring)
        return [metric, params]

    def search_params(self, params_list, mode='StratifiedKFold', k=10, scoring='roc_auc', parallel=True):
        best_metric, best_params = 0, None
        self.mode, self.k, self.scoring = mode, k, scoring
        if parallel:
            results = asyn_parallel(self.parallel_search_pararms_func, [[p] for p in params_list], cpu_num=20)
        else:
            results = [self.parallel_search_pararms_func(params) for params in tqdm(params_list)]
        search_metrics = [pair[0] for pair in results]
        search_params = [pair[1] for pair in results]
        for i, metric in enumerate(search_metrics):
            if metric > best_metric:
                best_metric, best_params = metric, search_params[i]
        return best_metric, best_params
