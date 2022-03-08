import logging

import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from curry.features import Extractor
from curry.loader import Loader


class Scorer:
    confusion_matrix = 'confusion_matrix'
    accuracy = 'accuracy'
    all = [confusion_matrix, accuracy]

    @classmethod
    def score(self, y_true, y_pred):
        return {
            Scorer.confusion_matrix: confusion_matrix(y_true, y_pred),
            Scorer.accuracy: accuracy_score(y_true, y_pred)
        }

    @classmethod
    def aggregate(self, score_type, score_values):
        if score_type == Scorer.confusion_matrix:
            return np.sum(np.array(score_values), axis=0)
        elif score_type == Scorer.accuracy:
            return np.mean(score_values)
        else:
            raise Exception("Unimplemented!")


class _XGBClassifier:
    def __init__(self, params):
        self.bst = None
        self.params = params

    def fit(self, X, y):
        self.bst = xgb.train(self.params, xgb.DMatrix(X, label=y))

    def score(self, X, y):
        y_pred = self.bst.predict(xgb.DMatrix(X, label=y))
        return Scorer.score(y, y_pred)

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

    @classmethod
    def randomForest(self, n_estimators):
        return RandomForestClassifier(n_estimators=n_estimators)


class Trainer:
    def __init__(self, data_dir, filter_multi_grade):
        self.filter_multi_grade = filter_multi_grade
        self.loader = Loader(data_dir)
        self.extractor = Extractor()

    def get_X_y(self, vec_type):
        df, selected_ilocs = self.loader.sublessons_w_content(self.filter_multi_grade)
        X = self.extractor.join(df.content, df.land, vec_type)
        y = df.klass.astype('category').cat.codes.values
        if selected_ilocs:
            return X[selected_ilocs], y[selected_ilocs]
        else:
            return X, y

    def aggregate_scores(self, scores):
        out = dict()
        for score_type in Scorer.all:
            score_values = []
            for score in scores:
                score_values.append(score[score_type])
            out[score_type] = Scorer.aggregate(score_type, score_values)
        return out

    def train_score(self, model_desc):
        logging.info(f"train_score: {model_desc}")
        model_name = model_desc['name']
        vec_type = model_desc['vec_type']
        args = model_desc['args']
        clf = getattr(Models, model_name)(*args)
        X, y = self.get_X_y(vec_type)
        scores = []
        folder = StratifiedKFold(n_splits=3)
        for train_index, test_index in folder.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            scores.append(score)
        return model_desc, self.aggregate_scores(scores)
