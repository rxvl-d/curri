import logging
import os

from curry.model import Scorer


class ConsoleWriter:
    @classmethod
    def write(cls, model_desc, score):
        logging.info("=" * 40)
        logging.info(str(model_desc))
        logging.info("-" * 40)
        for score_type in score:
            logging.info(f"Score type: {score_type}")
            logging.info(f"Value:\n{score[score_type]}")

class FileWriter:
    @classmethod
    def result_file_name(cls, model_desc):
        return f'{model_desc["name"]}_{model_desc["vec_type"]}_{"-".join([str(i) for i in model_desc["args"]])}'

    @classmethod
    def write(cls, results):
        if not os.path.isdir('results'):
            os.mkdir('results')
        for model_desc, score in results:
            with open('results/' + cls.result_file_name(model_desc) + ".confusion_matrix", 'w') as f:
                f.write(str(score[Scorer.confusion_matrix]))
        accuracies = [(cls.result_file_name(model_desc), score[Scorer.accuracy]) for model_desc, score in results]
        with open("results/accuracies", 'a') as f:
            for model_name, acc in accuracies:
                f.write(f"{model_name}\t{acc}\n")

