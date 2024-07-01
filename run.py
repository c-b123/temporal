from temporal import MultiplePreprocessor
from temporal.dataset import DatasetCreator

dc = DatasetCreator(colab=False)
# dc.create_dataset_single_site(["adultFemaleLice", "probablyNoFish"], 12011)
dc.create_dataset_multiple_sites(["adultFemaleLice", "probablyNoFish"])
timeseries = dc.get_dataset()


preprocessor = MultiplePreprocessor(timeseries)
preprocessor.train_val_test_split()
preprocessor.standardize_data(["adultFemaleLice", "probablyNoFish"])
preprocessor.create_features_and_targets(input_window=26, offset=1)
X_train, y_train, X_val, y_val, X_test, y_test = preprocessor.get_feature_and_target_datasets()
print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)
