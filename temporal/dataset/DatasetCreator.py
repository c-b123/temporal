import numpy as np
import pandas as pd

from temporal import GitHubDataloader
from pathlib import Path


class DatasetCreator:
    def __init__(self, colab=True):
        self.colab = colab

        loader = GitHubDataloader(self.colab)
        self.__aq = pd.read_csv(loader.get_data(Path("Resources/aquaculture_registry.csv")))
        self.__lc = pd.read_csv(loader.get_data(Path("Resources/lice_counts_norway_2012-2023_updated.csv")))
        self.__ts = loader.get_data(Path("Resources/timesteps.json"))

        self.feature_list = None
        self.startDate = None
        self.endDate = None

        self.__dataset = None

        self.__encode_dummy_features()

    def __encode_dummy_features(self):
        for col in ["probablyNoFish", "aboveLimit", "hasCountedLice"]:
            self.__lc[col] = self.__lc[col].map({"yes": 1, "no": 0, "Ukjent": 0})

    def create_dataset_single_site(self, localityNo, feature_list, startDate="2012 - 1", endDate="2023 - 52"):
        """
        adultFemaleLice,mobileLice,stationaryLice,totalLice,probablyNoFish,
        hasCountedLice,liceLimit,aboveLimit,seaTemperature
        :param localityNo:
        :param endDate:
        :param startDate:
        :param feature_list:
        :return:
        """

        self.feature_list = feature_list
        self.startDate = startDate
        self.endDate = endDate

        liceData = self.__lc[self.__lc["localityNo"] == localityNo]

        self.__dataset = np.zeros((len(self.feature_list), len(self.__ts.keys())))

        for index, row in liceData.iterrows():
            t = self.__ts[f"{int(row['year'])} - {int(row['week'])}"]
            for i, feature in enumerate(self.feature_list):
                if pd.notnull(row[feature]):
                    self.__dataset[i][t] = row[feature]

        idx1 = self.__ts[self.startDate]
        idx2 = self.__ts[self.endDate] + 1
        assert idx1 < idx2, "Start date must be less than end date"
        self.__dataset = self.__dataset[:, idx1:idx2]

    def get_dataset(self):
        return self.__dataset


if __name__ == "__main__":
    dc = DatasetCreator(colab=False)
    dc.create_dataset_single_site(12011,
                                  ["adultFemaleLice", "probablyNoFish"],
                                  "2023 - 1",
                                  "2023 - 20")
    print(dc.get_dataset().shape)
    print(dc.get_dataset())
