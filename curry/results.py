import logging
import os

from curry.model import Scorer


class ConsoleWriter:
    @classmethod
    def write(cls, job_desc, score):
        logging.info("=" * 40)
        logging.info(str(job_desc))
        logging.info("-" * 40)
        for score_type in score:
            logging.info(f"Score type: {score_type}")
            logging.info(f"Value:\n{score[score_type]}")
            logging.info(f"Result written to:\n{result_file_name(job_desc)}")

class FileWriter:
    @classmethod
    def write(cls, results):
        if not os.path.isdir('results'):
            os.mkdir('results')
        for job_desc, score in results:
            with open('results/' + result_file_name(job_desc) + "_confusion_matrix.txt", 'w') as f:
                f.write(str(score[Scorer.confusion_matrix]))
        accuracies = [(result_file_name(model_desc), score[Scorer.accuracy]) for model_desc, score in results]
        with open("results/accuracies", 'a') as f:
            for model_name, acc in accuracies:
                f.write(f"{model_name}\t{acc}\n")

    @classmethod
    def write_features(cls, job_desc, top_features):
        with open('results/' + result_file_name(job_desc) + "_top_features.txt", 'w') as f:
            f.write('\n'.join(top_features))


def result_file_name(job_desc):
    return f'{job_desc["name"]}_{job_desc["vec_type"]}_{job_desc["filtered"]}_{job_desc.get("land", "allstates")}'