from src.experiment.main import *
import pytest
import os
FOLDER_PATH = os.getcwd()
BAD_DATA = 'hsgj.csv'
GOOD_DATA = 'tests/clean_test.csv'
BAD_MODEL = 'hsgj.h5'
GOOD_MODEL = 'tfidf_coronatweetmodel.h5'
BAD_NETWORK = 'hsjdgf'

GOOD_DATA_PATH= os.path.join(FOLDER_PATH, GOOD_DATA)
BAD_DATA_PATH= os.path.join(FOLDER_PATH, BAD_DATA)
GOOD_MODEL_PATH = os.path.join(FOLDER_PATH, GOOD_MODEL)
BAD_MODEL_PATH= os.path.join(FOLDER_PATH, BAD_MODEL)
params_bad = {
        'units' : 'a',
        'epochs' : 'b',
        'batch_size': 'c',
        'seq_length': 'd',
        'bidirectional': 'yes'
        }
params_good = {
        'units' : 1,
        'epochs' : 2,
        'batch_size': 3,
        'seq_length': 4,
        'bidirectional': True
        }


class TestModel():
    def test_network(self):
        with pytest.raises(ValueError) as e:
            RMSE = run_experiment(GOOD_DATA_PATH, GOOD_DATA_PATH, BAD_NETWORK, params_good)
        err_msg = 'Please select only \'tfidf\', \'lstm\', \
                            or \'gru\' for model_type.'
        assert e.match(err_msg)
    
    def test_train_path(self):
        
        with pytest.raises(OSError) as e:
            RMSE = run_experiment(BAD_DATA_PATH, GOOD_DATA_PATH, 'gru', params_good)
        err_msg = '.csv file not found for train'
        assert e.match(err_msg), 'Did not throw error with invalid csv file'

    def test_val_path(self):
        
        with pytest.raises(OSError) as e:
            RMSE = run_experiment(GOOD_DATA_PATH, BAD_DATA_PATH, 'gru', params_good)
        err_msg = '.csv file not found for val'
        assert e.match(err_msg), 'Did not throw error with invalid csv file'
    
    def test_params(self):
        RMSE = run_experiment(GOOD_DATA_PATH, GOOD_DATA_PATH, 'gru', params_good)
        assert isinstance(params_good, dict)
        assert isinstance(params_good['units'], int)
        assert isinstance(params_good['epochs'], int)
        assert isinstance(params_good['batch_size'], int)
        assert isinstance(params_good['seq_length'], int)
        assert isinstance(params_good['bidirectional'], bool)
