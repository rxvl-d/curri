import sys

from tqdm import tqdm

from curry.features import SentenceTransformer

sys.path.append('.')
import logging

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
import json
import argparse
from curry.model import Trainer
from curry.results import ConsoleWriter, FileWriter


class Runner:
    def __init__(self, data_dir):
        self.st = SentenceTransformer(data_dir + '/cache/')
        self.trainer = Trainer(data_dir)

    def run_train(self, job_descs):
        for job_desc in tqdm(job_descs):
            result = self.trainer.train_score(job_desc)
            ConsoleWriter.write(job_desc, result)
            yield job_desc, result

    def run_sentence_transformers(self):
        self.st.write_cache()

def parse(args):
    parser = argparse.ArgumentParser(description='Train Level Prediction.')
    parser.add_argument('job', type=str)
    parser.add_argument('data_dir', type=str)
    parser.add_argument('job_desc', type=str, default=None)
    args = parser.parse_args(args)
    if args.job == 'train':
        with open(args.job_desc) as f:
            job_descs = json.load(f)
        return args, job_descs
    else:
        return args, None


def main(args):
    parsed_args, job_descs = parse(args)
    if args.job == 'train':
        results = Runner(
            data_dir=parsed_args.data_dir
        ).run_train(job_descs)
        FileWriter.write(list(results))
    elif args.job == 'gen_st':
        Runner(parsed_args.data_dir).run_sentence_transformers()


if __name__ == '__main__':
    main(sys.argv[1:])
