import sys

from tqdm import tqdm

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
        for job_desc in tqdm(job_descs):
            result = self.trainer.train_score(job_desc)
            ConsoleWriter.write(job_desc, result)
            yield job_desc, result


def parse(args):
    parser = argparse.ArgumentParser(description='Train Level Prediction.')
    parser.add_argument('data_dir', type=str)
    parser.add_argument('job_desc', type=str)
    args = parser.parse_args(args)
    with open(args.job_desc) as f:
        job_descs = json.load(f)
    return args, job_descs


def main(args):
    parsed_args, job_descs = parse(args)
    results = Runner(
        data_dir=parsed_args.data_dir
    ).run(job_descs)
    FileWriter.write(list(results))


if __name__ == '__main__':
    main(sys.argv[1:])
