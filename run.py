from temporal.preprocessing import Preprocessor
from temporal.dataset import DatasetCreator

dc = DatasetCreator(colab=False)
dc.create_dataset_single_site(12011, ["adultFemaleLice"])
timeseries = dc.get_dataset()

preprocessor = Preprocessor(timeseries)
preprocessor.train_val_test_split()
preprocessor.standardize_data()
preprocessor.create_features_and_targets(input_window=7, offset=1)
X_train, y_train, X_val, y_val, X_test, y_test = preprocessor.get_feature_and_target_datasets()
a = 5
