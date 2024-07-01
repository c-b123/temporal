import numpy as np

from .AbstractPreprocessor import AbstractPreprocessor


class SinglePreprocessor(AbstractPreprocessor):
    def __init__(self, data):
        super().__init__(data)

    def train_val_test_split(self, val_ratio=0.1, test_ratio=0.1):
        self._val_ratio = val_ratio
        self._test_ratio = test_ratio

        self._train_snapshots = int((1 - self._val_ratio - self._test_ratio) * self._data.shape[1])
        self._val_snapshots = int((1 - self._test_ratio) * self._data.shape[1])

        self._train = self._data[:, 0:self._train_snapshots].T
        self._val = self._data[:, self._train_snapshots:self._val_snapshots].T
        self._test = self._data[:, self._val_snapshots:].T

    def standardize_data(self, features: list):
        self._features = features
        self._features_to_std = self._get_idx_of_features_to_standardize()
        self._means = np.mean(self._train, axis=0)
        self._stds = np.std(self._train, axis=0)

        train_copy = np.copy(self._train)
        train_copy[:, self._features_to_std] = ((self._train[:, self._features_to_std] -
                                                 self._means[self._features_to_std]) /
                                                self._stds[self._features_to_std])
        self._train = train_copy

        val_copy = np.copy(self._val)
        val_copy[:, self._features_to_std] = ((self._val[:, self._features_to_std] -
                                               self._means[self._features_to_std]) /
                                              self._stds[self._features_to_std])
        self._val = val_copy

        test_copy = np.copy(self._test)
        test_copy[:, self._features_to_std] = ((self._test[:, self._features_to_std] -
                                                self._means[self._features_to_std]) /
                                               self._stds[self._features_to_std])
        self._test = test_copy

    def _create_features_and_targets(self, data):
        X, y = [], []
        for i in range(len(data) - self._input_window - self._offset + 1):
            X.append(data[i:i + self._input_window])
            y.append([data[i + self._input_window + self._offset - 1][0]])
        return np.array(X), np.array(y)


if __name__ == "__main__":
    arr = np.array([[0.3, 0.3, 0.3, 0.4, 0., 0., 0., 0., 0., 0.]])
    preprocessor = SinglePreprocessor(arr)
    preprocessor.train_val_test_split()
    # preprocessor.standardize_data()
    preprocessor.create_features_and_targets(2, 1)
    X_train, y_train, X_val, y_val, X_test, y_test = preprocessor.get_feature_and_target_datasets()
    print(X_train)
