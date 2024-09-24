import numpy as np

from .MultiplePreprocessor import MultiplePreprocessor


class GlobalPreprocessor(MultiplePreprocessor):
    """
    This class provides some basic preprocessing and allows to create feature-target pairs when dealing with multiple
    timeseries. In contrast to the MultiplePreprocessor class this class provides parallel features and targets.
    This class is used to create feature-target pairs when dealing with the global model.
    """
    def __init__(self, data):
        super().__init__(data)

    def _create_features_and_targets(self, data):
        """
        Overriding the create_features_and_targets method of the MultiplePreprocessor class. This
        method creates the features and targets using the sliding window approach.
        Args:
            data: The timeseries data for which to create the features and targets.

        Returns:
            Two numpy arrays containing the features and targets.
        """
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
