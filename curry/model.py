import logging
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from curry.features import Extractor
from curry.loader import Loader


class Models:
    @classmethod
    def xgbClassifier(self, variance_threshold):
        return Pipeline([
            ('variance_threshold', VarianceThreshold(threshold=variance_threshold)),
            ('classification', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'))
        ])


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
        (model_name, args) = model_desc
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
        return np.mean(scores)
