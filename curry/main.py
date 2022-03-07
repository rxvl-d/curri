import sys
sys.path.append('.')
import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

import argparse
from tqdm import tqdm
from curry.model import Trainer
from multiprocessing import Pool

class Runner:
    def __init__(self, n_par, data_dir):
        self.pool = Pool(n_par)
        self.trainer = Trainer(data_dir)

    def run(self):
        models = [
            ('xgbClassifier', (0.01,)),
            ('xgbClassifier', (0.02,)),
            ('xgbClassifier', (0.03,)),
            ('xgbClassifier', (0.04,)),
            ('xgbClassifier', (0.05,))
        ]
        return list(tqdm(self.pool.imap(self.trainer.train_score, models), total=len(models)))

def parse():
    parser = argparse.ArgumentParser(description='Train Level Prediction.')
    parser.add_argument('n_par', type=int)
    parser.add_argument('data_dir', type=str)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse()
    for i in Runner(n_par=args.n_par, data_dir=args.data_dir).run():
        print(i)
