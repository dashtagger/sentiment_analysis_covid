import os

import pytest

from src.datapipeline.loader import *

CSV_PATH = os.getcwd() + '/tests/'
BAD_FILE = 'hsgj.csv'
GOOD_FILE = 'clean_test.csv'
DATA_PATH = os.path.join(CSV_PATH, BAD_FILE)
TEST_RATIO = 0.1
VAL_RATIO = 0.1


class TestLoader:
    def test_url(self):
        with pytest.raises(OSError) as e:
            df = Loader(DATA_PATH)
        err_msg = 'Please provide a valid csv file'
        assert e.match(err_msg), 'Did not throw error with invalid csv file'

    def test_split(self):
        df = Loader(os.path.join(CSV_PATH, GOOD_FILE))
        X_train, y_train, X_test, y_test, X_val, y_val = df.split_data(df, TEST_RATIO, VAL_RATIO)
        assert isinstance(TEST_RATIO, float)
        assert isinstance(VAL_RATIO, float)
        assert (TEST_RATIO < 1.0), 'Enter a number less than 1.0'
        assert (VAL_RATIO < 1.0), 'Enter a number less than 1.0'
        assert (TEST_RATIO > 0.0), 'Enter a number more than 0.0'
        assert (VAL_RATIO > 0.0), 'Enter a number more than 0.0'

