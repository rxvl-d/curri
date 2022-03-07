import logging
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
import xgboost as xgb

from curry.features import Extractor
from curry.loader import Loader


class Models:
    @classmethod
    def xgb_params(self, nthread):
        return {'max_depth':4,
                   'use_label_encoder':False,
                   'objective':'multi:softmax',
                   'eval_metric': 'merror',
                   'seed': 42,
                   'nthread': nthread,
                   'num_class': 9}

    @classmethod
    def xgbClassifier(self, variance_threshold, nthread):
        return Pipeline([
            ('variance_threshold', VarianceThreshold(threshold=variance_threshold)),
            ('classification', xgb.XGBClassifier(**Models.xgb_params(nthread)))
        ])

    @classmethod
    def xgbClassifierNoSelection(self, nthread):
        return xgb.XGBClassifier(**Models.xgb_params(nthread))


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
            y_pred = clf.predict(X_test)
            scores.append(accuracy_score(y_test, y_pred))
        return model_desc, np.mean(scores)
