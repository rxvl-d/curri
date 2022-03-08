import sys
sys.path.append('.')
import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
import json
import argparse
from curry.model import Trainer
from curry.results import ConsoleWriter, FileWriter
from multiprocessing import Pool


class Runner:
    def __init__(self, n_par=2, data_dir='../data'):
        self.n_par = n_par
        self.trainer = Trainer(data_dir)

    def run(self, model_conf):
        if len(model_conf) == 1:
            return [self.trainer.train_score(model_conf[0])]
        else:
            pool = Pool(self.n_par)
            return pool.map(self.trainer.train_score, model_conf)


def parse():
    parser = argparse.ArgumentParser(description='Train Level Prediction.')
    parser.add_argument('n_par', type=int)
    parser.add_argument('data_dir', type=str)
    parser.add_argument('model_conf', type=str)
    args = parser.parse_args()
    with open(args.model_conf) as f:
        model_conf = json.load(f)
    return args, model_conf


if __name__ == '__main__':
    args, model_conf = parse()
    results = Runner(n_par=args.n_par, data_dir=args.data_dir).run(model_conf)
    ConsoleWriter.write(results)
    FileWriter.write(results)