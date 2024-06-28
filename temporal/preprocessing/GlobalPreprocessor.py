import math

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

        return np.array(X_all), np.array(y_all)

    def inverse_transform(self, array: np.array):
        return np.multiply(array, math.sqrt(self._scaler.var_[0])) + self._scaler.mean_[0]


if __name__ == "__main__":
    pass
    # dc = DatasetCreator(colab=False)
    # # dc.create_dataset_single_site(12011, wandb.config["features"])
    # dc.create_dataset_multiple_sites(feature_list=["adultFemaleLice", "probablyNoFish"])
    # timeseries = dc.get_dataset()
    #
    # preprocessor = GlobalPreprocessor(timeseries)
    # preprocessor.train_val_test_split()
    # preprocessor.standardize_data()
    # preprocessor.create_features_and_targets(input_window=7, offset=1)
    # X_train, y_train, X_val, y_val, X_test, y_test = preprocessor.get_feature_and_target_datasets()
    # print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)
    #
    # preprocessor.inverse_transform(np.random.rand(56, 80))
