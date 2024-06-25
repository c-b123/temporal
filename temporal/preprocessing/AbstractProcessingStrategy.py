from abc import ABC, abstractmethod


class ProcessingStrategy(ABC):

    @abstractmethod
    def train_val_test_split(self, val_ratio=0.1, test_ratio=0.1):
        pass

    @abstractmethod
    def standardize_data(self):
        pass

    @abstractmethod
    def __create_features_and_targets(self, data):
        pass

    @abstractmethod
    def create_features_and_targets(self, input_window, offset):
        pass

    @abstractmethod
    def inverse_transform(self, array):
        pass
