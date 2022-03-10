import os
import sys
from curry.main import main
import shutil

sys.path.append('.')

def test_main():
    shutil.rmtree('results', ignore_errors=True)
    main(['data/', 'test/job_descs/simple.json'])
    assert os.path.isdir('results')
    assert 'accuracies' in os.listdir('results')
    assert 'xgbClassifier_babelkw_True.confusion_matrix' in os.listdir('results')
    with open('results/accuracies') as f:
        assert 'xgbClassifier_babelkw_True' in f.read()


def test_combination_features():
    shutil.rmtree('results', ignore_errors=True)
    main(['data/', 'test/job_descs/combination_features.json'])
    assert os.path.isdir('results')
    assert 'accuracies' in os.listdir('results')
    assert 'xgbClassifier_babelkw+wikikw_True.confusion_matrix' in os.listdir('results')
    with open('results/accuracies') as f:
        assert 'xgbClassifier_babelkw+wikikw_True' in f.read()


def test_combination_features():
    shutil.rmtree('results', ignore_errors=True)
    main(['data/', 'test/job_descs/combination_features.json'])
    assert os.path.isdir('results')
    assert 'accuracies' in os.listdir('results')
    assert 'xgbClassifier_st+wikikw_True.confusion_matrix' in os.listdir('results')
    with open('results/accuracies') as f:
        assert 'xgbClassifier_babelkw+wikikw_True' in f.read()

