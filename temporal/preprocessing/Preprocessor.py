import math
import numpy as np

from sklearn.preprocessing import StandardScaler


class Preprocessor:
    def __init__(self, data):
        self.__data = data

        self.__train = None
        self.__val = None
        self.__test = None

        self.input_window = None
        self.offset = None

        self.__X_train = None
        self.__y_train = None
        self.__X_val = None
        self.__y_val = None
        self.__X_test = None
        self.__y_test = None

        self.val_ratio = None
        self.test_ratio = None

        self.__train_snapshots = None
        self.__val_snapshots = None

    def train_val_test_split(self, val_ratio=0.1, test_ratio=0.1):
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        self.__train_snapshots = int((1 - self.val_ratio - self.test_ratio) * self.__data.shape[1])
        self.__val_snapshots = int((1 - self.test_ratio) * self.__data.shape[1])

        self.__train = self.__data[:, 0:self.__train_snapshots].T
        self.__val = self.__data[:, self.__train_snapshots:self.__val_snapshots].T
        self.__test = self.__data[:, self.__val_snapshots:].T

    def standardize_data(self):
        self.__scaler = StandardScaler()
        self.__train = self.__scaler.fit_transform(self.__train)
        self.__val = self.__scaler.transform(self.__val)
        self.__test = self.__scaler.transform(self.__test)

    def __create_features_and_targets(self, data):
        X, y = [], []
        for i in range(len(data) - self.input_window - self.offset):
            X.append(data[i:i + self.input_window])
            y.append([data[i + self.input_window + self.offset - 1][0]])
        return np.array(X), np.array(y)

    def create_features_and_targets(self, input_window, offset):
        self.input_window = input_window
        self.offset = offset

        self.__X_train, self.__y_train = self.__create_features_and_targets(self.__train)
        self.__X_val, self.__y_val = self.__create_features_and_targets(self.__val)
        self.__X_test, self.__y_test = self.__create_features_and_targets(self.__test)

    def inverse_transform(self, array):
        return np.multiply(array, math.sqrt(self.__scaler.var_[0])) + self.__scaler.mean_[0]

    def get_feature_and_target_datasets(self):
        return self.__X_train, self.__y_train, self.__X_val, self.__y_val, self.__X_test, self.__y_test

    def get_train_snapshots(self):
        return self.__train_snapshots

    def get_val_snapshots(self):
        return self.__val_snapshots


if __name__ == "__main__":
    data = np.array([[0.35, 0.3, 0.3, 0.45, 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 1., 1., 1., 1., 1., 1.]])
    preprocessor = Preprocessor(data)
    preprocessor.train_val_test_split()
    preprocessor.standardize_data()
    print(preprocessor.get_train_snapshots())
    # preprocessor.create_features_and_targets(2, 1)
