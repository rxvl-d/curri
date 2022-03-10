import logging

import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import KBinsDiscretizer

from curry.features import Extractor
from curry.loader import Loader


class Scorer:
    confusion_matrix = 'confusion_matrix'
    accuracy = 'accuracy'
    empty = {confusion_matrix: None, accuracy: -1.0}
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
        self.clf = xgb.XGBClassifier(**params)

    def fit(self, X, y):
        self.clf.fit(X, y)

    def score(self, X, y):
        y_pred = self.clf.predict(X)
        return Scorer.score(y, y_pred)


class _XGBMultilabelClassifer:
    def __init__(self, params):
        self.clf = MultiOutputClassifier(xgb.XGBClassifier(**params))

    def fit(self, X, y):
        self.clf.fit(X, y)

    def score(self, X, y):
        y_pred = self.clf.predict(X)
        return Scorer.score(y, y_pred)

class _XGBCoarseGrained:
    def __init__(self, params, bins):
        self.clf = xgb.XGBClassifier(**params)
        self.binner = KBinsDiscretizer(n_bins=bins, encode='ordinal')

    def coarsify(self, y):
        return self.binner.fit_transform(y.reshape((-1, 1)))

    def fit(self, X, y):
        self.clf.fit(X, self.coarsify(y))

    def score(self, X, y):
        y_pred = self.clf.predict(X)
        return Scorer.score(self.coarsify(y), y_pred)


class _XGBOrdinalClassifer:
    # Code from https://stackoverflow.com/questions/57561189/multi-class-multi-label-ordinal-classification-with-sklearn
    def __init__(self, params):
        self.params = params
        self.clf = xgb.XGBClassifier
        self.clfs = {}

    def fit(self, X, y):
        self.unique_class = np.sort(np.unique(y))
        if self.unique_class.shape[0] > 2:
            for i in range(self.unique_class.shape[0] - 1):
                # for each k - 1 ordinal value we fit a binary classification problem
                binary_y = (y > self.unique_class[i]).astype(np.uint8)
                clf = self.clf(**self.params)
                clf.fit(X, binary_y)
                self.clfs[i] = clf

    def predict_proba(self, X):
        clfs_predict = {k: self.clfs[k].predict_proba(X) for k in self.clfs}
        predicted = []
        for i, y in enumerate(self.unique_class):
            if i == 0:
                # V1 = 1 - Pr(y > V1)
                predicted.append(1 - clfs_predict[i][:, 1])
            elif i in clfs_predict:
                # Vi = Pr(y > Vi-1) - Pr(y > Vi)
                predicted.append(clfs_predict[i - 1][:, 1] - clfs_predict[i][:, 1])
            else:
                # Vk = Pr(y > Vk-1)
                predicted.append(clfs_predict[i - 1][:, 1])
        return np.vstack(predicted).T

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y):
        _, indexed_y = np.unique(y, return_inverse=True)
        return Scorer.score(indexed_y, self.predict(X))


class Models:
    @classmethod
    def xgb_params(self, nthreads):
        return {'max_depth': 4,
                'use_label_encoder': False,
                'seed': 42,
                'nthread': nthreads}

    @classmethod
    def xgbClassifier(self, nthreads):
        params = Models.xgb_params(nthreads)
        params['num_class'] = 9
        params['objective'] = 'multi:softmax'
        params['eval_metric'] = 'merror'
        return _XGBClassifier(params)

    @classmethod
    def xgbOrdinalClassifier(self, nthreads):
        params = Models.xgb_params(nthreads)
        params['objective'] = 'binary:logistic'
        params['eval_metric'] = 'error'
        return _XGBOrdinalClassifer(params)

    @classmethod
    def xgbMultilabelClassifer(self, nthreads):
        params = Models.xgb_params(nthreads)
        params['objective'] = 'binary:logistic'
        params['eval_metric'] = 'error'
        return _XGBMultilabelClassifer(params)

    @classmethod
    def xgbCoarseGrainedClassifier(self, nthreads):
        params = Models.xgb_params(nthreads)
        params['objective'] = 'multi:softmax'
        params['eval_metric'] = 'merror'
        return _XGBCoarseGrained(params, bins=3)

    @classmethod
    def randomForest(self, n_estimators):
        return RandomForestClassifier(n_estimators=n_estimators)


class Trainer:
    def __init__(self, data_dir):
        self.loader = Loader(data_dir)
        self.extractor = Extractor()

    def get_X_y(self, vec_type, filter_multi_grade, land):
        df, selected_lesson_ilocs, selected_land_ilocs = self.loader.sublessons_w_content(filter_multi_grade, land)
        selected_ilocs = list(set(selected_land_ilocs).intersection(selected_land_ilocs))
        X = self.extractor.join(df.content, df.land if land else None, vec_type)
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

    def train_score(self, job_desc):
        logging.info(f"train_score: {job_desc}")
        model_name = job_desc['name']
        vec_type = job_desc['vec_type']
        filter_multi_grade = job_desc['filtered']
        args = job_desc['args']
        land = job_desc.get('land')
        clf = getattr(Models, model_name)(*args)
        X, y = self.get_X_y(vec_type, filter_multi_grade, land)
        scores = []
        try:
            folder = StratifiedKFold(n_splits=3)
            for train_index, test_index in folder.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                clf.fit(X_train, y_train)
                score = clf.score(X_test, y_test)
                scores.append(score)
            return self.aggregate_scores(scores)
        except Exception as e:
            logging.error(e)
            return Scorer.empty
