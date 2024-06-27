import numpy as np

from .MultiplePreprocessor import MultiplePreprocessor


class GlobalPreprocessor(MultiplePreprocessor):
    def __init__(self, data):
        super().__init__(data)

    def _create_features_and_targets(self, data):
        n_sites, n_time_steps, n_features = data.shape
        n_snapshots = n_time_steps - self._input_window - self._offset + 1

        X_all = []
        y_all = []
        for snapshot in range(n_snapshots):
            concatenated = np.concatenate(data, axis=1)
            snapshot_data = concatenated[snapshot:snapshot + self._input_window, :]
            X_all.append(snapshot_data)
            y_all.append(data[:, snapshot + self._input_window + self._offset - 1, 0])
            a = 5

        return np.array(X_all), np.array(y_all)

    def inverse_transform(self, array: np.array):
        # TODO overwrite this method to allow inverse transformation of global model
        return np.multiply(array, math.sqrt(self._scaler.var_[0])) + self._scaler.mean_[0]


if __name__ == "__main__":
    arr = np.array([
        [[0.3, 0.3, 0.3, 0.4, 0., 0., 0., 0., 0., 0.],
         [0.0, 0.0, 0.0, 0.0, 1., 1., 1., 1., 1., 1.]],
        [[0.1, 0.1, 0.2, 0.6, 0., 0., 0., 0., 0., 0.],
         [0.0, 0.0, 0.0, 0.0, 0., 0., 0., 1., 1., 1.]]
    ])

    arr = np.array([
        [
            [0.3, 0.],
            [0.3, 0.],
            [0.3, 0.],
            [0.4, 0.],
            [0., 1.],
            [0., 1.],
            [0., 1.],
            [0., 1.]
        ],
        [
            [0.1, 0.],
            [0.1, 0.],
            [0.2, 0.],
            [0.6, 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 1.]
        ]
    ])

    print(arr.shape)

    concatenated = np.concatenate(arr, axis=1)
    print(concatenated)

    snapshot_data = concatenated[0:0 + 2, :]
    print(snapshot_data)

    print(concatenated[0 + 2 + 1][range(0, 2 * 2, 2)])
    print(arr[:, 3, 0])

    # snapshot_data = arr[:, :, 0:0 + 2]
    # print(snapshot_data)
    #
    # snapshot_data = snapshot_data.transpose(0, 2, 1)
    # print(snapshot_data)
    #
    # snapshot_data = np.concatenate(snapshot_data, axis=1)
    # print(snapshot_data)

    # preprocessor = GlobalPreprocessor(arr)
    # preprocessor.train_val_test_split()
    # # preprocessor.standardize_data()
    # preprocessor.create_features_and_targets(2, 1)
    # X_train, y_train, X_val, y_val, X_test, y_test = preprocessor.get_feature_and_target_datasets()
    # print(X_train)
    # print(X_train.shape)
