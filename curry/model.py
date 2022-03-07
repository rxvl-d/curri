import logging

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from curry.features import Extractor
from curry.loader import Loader


class _XGBClassifier:
    def __init__(self, params):
        self.bst = None
        self.params = params

    def fit(self, X, y):
        self.bst = xgb.train(self.params, xgb.DMatrix(X, label=y))

    def score(self, X, y):
        y_pred = self.bst.predict(xgb.DMatrix(X, label=y))
        return accuracy_score(y, y_pred)


class Models:
    @classmethod
    def xgb_params(self, nthreads):
        return {'max_depth': 4,
                'use_label_encoder': False,
                'objective': 'multi:softmax',
                'eval_metric': 'merror',
                'seed': 42,
                'nthread': nthreads,
                'num_class': 9}

    @classmethod
    def xgbClassifier(self, nthreads):
        return _XGBClassifier(Models.xgb_params(nthreads))


class Trainer:
    def __init__(self, data_dir):
        self.loader = Loader(data_dir)
        self.extractor = Extractor()

    def get_X_y(self):
        df = self.loader.sublessons_w_content()
        X = self.extractor.join(df.content, df.land)
        y = df.klass.astype('category').cat.codes.values
        return X, y

    def train_score(self, model_desc):
        logging.info(f"train_score: {model_desc}")
        model_name = model_desc['name']
        args = model_desc['args']
        clf = getattr(Models, model_name)(*args)
        X, y = self.get_X_y()
        scores = []
        folder = StratifiedKFold(n_splits=3)
        for train_index, test_index in folder.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            scores.append(score)
        return model_desc, np.mean(scores)
