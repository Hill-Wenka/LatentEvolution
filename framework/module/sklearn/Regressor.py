from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

regressors = ['LinearRegression', 'Lasso', 'Ridge', 'SVR', 'LinearSVR', 'DecisionTreeRegressor',
              'KNeighborsRegressor', 'RandomForestRegressor', 'AdaBoostRegressor', 'BaggingRegressor',
              'MLPRegressor', 'XGBRegressor', 'LGBMRegressor']


class Regressor():
    def __init__(self, model_name, params):
        super(Regressor, self).__init__()
        self.model_name = model_name
        self.model = self.init_model(model_name, params)
        self.init_model(model_name, params)

    def init_model(self, name, params):
        if name == 'LinearRegression':
            model = LinearRegression(**params)
        elif name == 'Lasso':
            model = Lasso(**params)
        elif name == 'Ridge':
            model = Ridge(**params)
        elif name == 'SVR':
            model = SVR(**params)
        elif name == 'LinearSVR':
            model = LinearSVR(**params)
        elif name == 'DecisionTreeRegressor':
            model = DecisionTreeRegressor(**params)
        elif name == 'KNeighborsRegressor':
            model = KNeighborsRegressor(**params)
        elif name == 'RandomForestRegressor':
            model = RandomForestRegressor(**params)
        elif name == 'AdaBoostRegressor':
            model = AdaBoostRegressor(**params)
        elif name == 'BaggingRegressor':
            model = BaggingRegressor(**params)
        elif name == 'MLPRegressor':
            model = MLPRegressor(**params)
        elif name == 'XGBRegressor':
            model = XGBRegressor(**params)
        elif name == 'LGBMRegressor':
            model = LGBMRegressor(**params)
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
