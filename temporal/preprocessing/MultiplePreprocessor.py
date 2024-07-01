import numpy as np

from .AbstractPreprocessor import AbstractPreprocessor


class MultiplePreprocessor(AbstractPreprocessor):
    def __init__(self, data):
        super().__init__(data)

    def train_val_test_split(self, val_ratio=0.1, test_ratio=0.1):
        self._val_ratio = val_ratio
        self._test_ratio = test_ratio

        self._snapshots = self._data.shape[2]
        self._train_snapshots = int((1 - self._val_ratio - self._test_ratio) * self._snapshots)
        self._val_snapshots = int((1 - self._test_ratio) * self._snapshots)

        self._train = self._data[:, :, 0:self._train_snapshots].transpose(0, 2, 1)
        self._val = self._data[:, :, self._train_snapshots:self._val_snapshots].transpose(0, 2, 1)
        self._test = self._data[:, :, self._val_snapshots:].transpose(0, 2, 1)

    def standardize_data(self, features: list):
        self._features_to_std = features

        self._means = np.mean(self._train, axis=(0, 1))
        self._stds = np.std(self._train, axis=(0, 1))

        # Standardize the training data
        train_copy = np.copy(self._train)
        train_copy[:, :, self._features_to_std] = (
                (self._train[:, :, self._features_to_std] -
                 self._means[self._features_to_std]) /
                self._stds[self._features_to_std]
        )
        self._train = train_copy

        # Standardize the validation data
        val_copy = np.copy(self._val)
        val_copy[:, :, self._features_to_std] = (
                (self._val[:, :, self._features_to_std] -
                 self._means[self._features_to_std]) /
                self._stds[self._features_to_std]
        )
        self._val = val_copy

        # Standardize the test data
        test_copy = np.copy(self._test)
        test_copy[:, :, self._features_to_std] = (
                (self._test[:, :, self._features_to_std] -
                 self._means[self._features_to_std]) /
                self._stds[self._features_to_std]
        )
        self._test = test_copy

    def _create_features_and_targets(self, data):
        X_all, y_all = [], []
        for site in data:
            X, y = [], []
            for i in range(len(site) - self._input_window - self._offset + 1):
                X.append(site[i:i + self._input_window])
                y.append([site[i + self._input_window + self._offset - 1][0]])
            X_all.append(np.array(X))
            y_all.append(np.array(y))
        return np.concatenate(X_all, axis=0), np.concatenate(y_all, axis=0)


if __name__ == "__main__":
    arr = np.array([
        [[0.3, 0.3, 0.3, 0.4, 0., 0., 0., 0., 0., 0.],
         [0.0, 0.0, 0.0, 0.0, 1., 1., 1., 1., 1., 1.]],
        [[0.1, 0.1, 0.2, 0.6, 0., 0., 0., 0., 0., 0.],
         [0.0, 0.0, 0.0, 0.0, 0., 0., 0., 1., 1., 1.]]
    ])
    preprocessor = MultiplePreprocessor(arr)
    preprocessor.train_val_test_split()
    # preprocessor.standardize_data()
    preprocessor.create_features_and_targets(2, 1)
    X_train, y_train, X_val, y_val, X_test, y_test = preprocessor.get_feature_and_target_datasets()
    print(X_train)
    print(X_train.shape)
