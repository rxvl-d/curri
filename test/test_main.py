import os
import sys
sys.path.append('.')
from curry.main import main
import shutil

def test_main():
    shutil.rmtree('results')
    main(['data/', 'test/job_descs/simple.json'])
    assert os.path.isdir('results')
    assert 'accuracies' in os.listdir('results')
    assert 'xgbClassifier_babelkw_True.confusion_matrix' in os.listdir('results')