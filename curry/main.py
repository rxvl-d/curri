import sys
sys.path.append('.')
import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
import json
import argparse
from curry.model import Trainer
from curry.results import ConsoleWriter, FileWriter


class Runner:
    def __init__(self, data_dir='../data'):
        self.trainer = Trainer(data_dir)

    def run(self, job_descs):
        for job_desc in job_descs:
            result = self.trainer.train_score(job_desc)
            ConsoleWriter.write(job_desc, result)
            yield job_desc, result


def parse():
    parser = argparse.ArgumentParser(description='Train Level Prediction.')
    parser.add_argument('data_dir', type=str)
    parser.add_argument('job_desc', type=str)
    args = parser.parse_args()
    with open(args.job_desc) as f:
        job_descs = json.load(f)
    return args, job_descs


if __name__ == '__main__':
    args, job_descs = parse()
    results = Runner(
        data_dir=args.data_dir
    ).run(job_descs)
    FileWriter.write(list(results))