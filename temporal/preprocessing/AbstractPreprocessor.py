import numpy as np

from abc import ABC, abstractmethod


class Preprocessor(ABC):
    def __init__(self, data):
        self._data = data

        self._train = None
        self._val = None
        self._test = None

        self._input_window = None
        self._offset = None

        self._X_train = None
        self._y_train = None
        self._X_val = None
        self._y_val = None
        self._X_test = None
        self._y_test = None

        self._val_ratio = None
        self._test_ratio = None

        self._train_snapshots = None
        self._val_snapshots = None

    @abstractmethod
    def train_val_test_split(self, val_ratio=0.1, test_ratio=0.1):
        pass

    @abstractmethod
    def standardize_data(self):
        pass

    @abstractmethod
    def _create_features_and_targets(self, data):
        pass

    @abstractmethod
    def create_features_and_targets(self, input_window: int, offset: int):
        pass

    @abstractmethod
    def inverse_transform(self, array: np.array()):
        pass

    def get_feature_and_target_datasets(self):
        return self._X_train, self._y_train, self._X_val, self._y_val, self._X_test, self._y_test

    @property
    def train_snapshots(self):
        return self._train_snapshots

    @property
    def val_snapshots(self):
        return self._val_snapshots
