import numpy as np

from abc import ABC, abstractmethod


class AbstractPreprocessor(ABC):
    def __init__(self, data):
        self._data = data

        self._train = None
        self._val = None
        self._test = None

        self._input_window = None
        self._offset = None

        self._features = None
        self._features_to_std = None
        self._means = None
        self._stds = None

        self._X_train = None
        self._y_train = None
        self._X_val = None
        self._y_val = None
        self._X_test = None
        self._y_test = None

        self._val_ratio = None
        self._test_ratio = None

        self._snapshots = None
        self._train_snapshots = None
        self._val_snapshots = None

    @abstractmethod
    def train_val_test_split(self, val_ratio=0.1, test_ratio=0.1):
        pass

    @abstractmethod
    def standardize_data(self, features: list):
        pass

    @abstractmethod
    def _create_features_and_targets(self, data):
        pass

    def _get_idx_of_features_to_standardize(self):
        features_to_std = {"adultFemaleLice": True,
                           "mobileLice": True,
                           "stationaryLice": True,
                           "totalLice": True,
                           "probablyNoFish": False,
                           "hasCountedLice": False,
                           "liceLimit": False,
                           "aboveLimit": False,
                           "seaTemperature": True,
                           "temperature_norkyst": True,
                           "salinity_norkyst": True}
        return [i for i, feature in enumerate(self._features) if features_to_std[feature]]

    def create_features_and_targets(self, input_window: int, offset: int):
        self._input_window = input_window
        self._offset = offset

        self._X_train, self._y_train = self._create_features_and_targets(self._train)
        self._X_val, self._y_val = self._create_features_and_targets(self._val)
        self._X_test, self._y_test = self._create_features_and_targets(self._test)

    def inverse_transform(self, array: np.array):
        """
        Assumes that the variable of interest is the first variable.
        Args:
            array:

        Returns:

        """
        return np.multiply(array, self._stds[0]) + self._means[0]

    def get_feature_and_target_datasets(self):
        return self._X_train, self._y_train, self._X_val, self._y_val, self._X_test, self._y_test

    @property
    def snapshots(self):
        return self._snapshots

    @property
    def train_snapshots(self):
        return self._train_snapshots

    @property
    def val_snapshots(self):
        return self._val_snapshots
