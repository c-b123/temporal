from preprocessing.AbstractProcessingStrategy import ProcessingStrategy


class MultipleProcessingStrategy(ProcessingStrategy):
    def train_val_test_split(self, val_ratio=0.1, test_ratio=0.1):
        pass

    def standardize_data(self):
        pass

    def create_features_and_targets(self, input_window, offset):
        pass

    def inverse_transform(self, array):
        pass
