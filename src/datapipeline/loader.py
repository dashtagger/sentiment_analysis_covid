import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class Loader:
    def __init__(self, data_path):
        """ Load the covid tweets dataset.

        Args:
            data_path (url/path): path to tweet dataset and should end with .csv.

        Returns:
            df (pandas df): Covid tweets dataset
        """
        self.df = pd.read_csv(data_path)


    def split_data(self, df, test_ratio, val_ratio):
        """ Split the covid tweets dataset

        Args:
            df (pandas df): Covid tweets dataset
            test_ratio (float): the ratio for test data set
            val_ratio (float): the ratio for validation data set

        Returns:
            X_train, y_train, X_test, y_test, X_val, y_val (pandas df): Split covid tweets dataset
        """
        X = self.df.drop(columns=['sentiment'])
        y = self.df['sentiment']
        
        bins = np.linspace(0, len(y), 10)
        y_binned = np.digitize(y, bins)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_ratio, shuffle = True, stratify=y_binned)
        
        ratio_remaining = 1 - test_ratio
        ratio_val_adjusted = val_ratio / ratio_remaining
        train_bins = np.linspace(0, len(y_train), 10)
        
        y_train_binned = np.digitize(y_train, train_bins)

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=ratio_val_adjusted, shuffle = True, stratify=y_train_binned)

        return X_train, y_train, X_test, y_test, X_val, y_val
