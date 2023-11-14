from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, KDTree
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC, OneClassSVM
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

classifiers = ['LogisticRegression', 'GaussianNB', 'BernoulliNB', 'MultinomialNB', 'SVC', 'OneClassSVM',
               'DecisionTreeClassifier', 'KNeighborsClassifier', 'KDTree', 'RandomForestClassifier',
               'AdaBoostClassifier', 'BaggingClassifier', 'MLPClassifier', 'XGBClassifier',
               'LGBMClassifier']


class Classifier():
    def __init__(self, model_name, params):
        super(Classifier, self).__init__()
        self.model_name = model_name
        self.model = self.init_model(model_name, params)
        self.init_model(model_name, params)

    def init_model(self, name, params):
        if name == 'RidgeClassifier':
            model = RidgeClassifier(**params)
        elif name == 'LogisticRegression':
            model = LogisticRegression(**params)
        elif name == 'GaussianNB':
            model = GaussianNB(**params)
        elif name == 'BernoulliNB':
            model = BernoulliNB(**params)
        elif name == 'MultinomialNB':
            model = MultinomialNB(**params)
        elif name == 'SVC':
            model = SVC(**params)
        elif name == 'LinearSVC':
            model = LinearSVC(**params)
        elif name == 'OneClassSVM':
            model = OneClassSVM(**params)
        elif name == 'DecisionTreeClassifier':
            model = DecisionTreeClassifier(**params)
        elif name == 'KNeighborsClassifier':
            model = KNeighborsClassifier(**params)
        elif name == 'KDTree':
            model = KDTree(**params)
        elif name == 'RandomForestClassifier':
            model = RandomForestClassifier(**params)
        elif name == 'AdaBoostClassifier':
            model = AdaBoostClassifier(**params)
        elif name == 'BaggingClassifier':
            model = BaggingClassifier(**params)
        elif name == 'MLPClassifier':
            model = MLPClassifier(**params)
        elif name == 'XGBClassifier':
            model = XGBClassifier(**params)
        elif name == 'LGBMClassifier':
            model = LGBMClassifier(**params)
        else:
            raise RuntimeError(f'No such pre-defined ML model: {name}')
        self.model = model

    def fit(self, train_X, train_Y, params=None):
        if params is not None:
            self.model.fit(train_X, train_Y, **params)
        else:
            self.model.fit(train_X, train_Y)
        self.num_train_data, self.train_x, self.train_y = train_X.shape[0], train_X, train_Y

    def predict(self, X):
        self.pred_x, self.pred_y = X, self.model.predict(X)
        return self.pred_y

    def predict_proba(self, X):
        self.pred_x, self.pred_y = X, self.model.predict_proba(X)
        return self.pred_y
