import argparse

from curry.features import Extractor
from curry.loader import Loader
from curry.model import Trainer, Models
from multiprocessing import Pool

class Runner:
    def __init__(self, n_par):
        self.pool = Pool(n_par)
        self.trainer = Trainer()

    def run(self):
        models = [
            ('xgbClassifier', (0.01,)),
            ('xgbClassifier', (0.02,)),
            ('xgbClassifier', (0.03,)),
            ('xgbClassifier', (0.04,)),
            ('xgbClassifier', (0.05,))
        ]
        return self.pool.map(self.trainer.train_score, models)

def parse():
    parser = argparse.ArgumentParser(description='Train Level Prediction.')
    parser.add_argument('n_par', type=int)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse()
    for i in Runner(n_par=args.n_par).run():
        print(i)