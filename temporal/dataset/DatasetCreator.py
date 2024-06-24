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

    def create_dataset_single_site(self, localityNo: int, feature_list):
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

        # Create empty vector
        n_features = len(feature_list)
        n_timesteps = len(self.__ts.keys())
        self.__dataset = np.zeros((n_features, n_timesteps))

        # Fill vector with observations from lice data file
        lice_data = self.__lc[self.__lc["localityNo"] == localityNo]
        for index, row in lice_data.iterrows():
            t = self.__ts[f"{int(row['year'])} - {int(row['week'])}"]
            for i, feature in enumerate(self.feature_list):
                if pd.notnull(row[feature]):
                    self.__dataset[i][t] = row[feature]

    def create_dataset_multiple_sites(self, feature_list: list, productionAreaNo=2):
        self.feature_list = feature_list

        # Get all localityNo for specified production area
        sites = sorted(self.__lc[self.__lc["productionAreaNo"] == productionAreaNo]["localityNo"].unique())

        # Create empty vector
        n_sites = len(sites)
        n_features = len(feature_list)
        n_timesteps = len(self.__ts.keys())
        self.__dataset = np.zeros((n_sites, n_features, n_timesteps))

        # Fill vector with observations from lice data file
        for s, site in enumerate(sites):
            lice_data = self.__lc[self.__lc["localityNo"] == site]
            for index, row in lice_data.iterrows():
                t = self.__ts[f"{int(row['year'])} - {int(row['week'])}"]
                for f, feature in enumerate(self.feature_list):
                    if pd.notnull(row[feature]):
                        self.__dataset[s][f][t] = row[feature]

    def trim_time_horizon(self, startDate="2012 - 1", endDate="2023 - 52"):
        idx1 = self.__ts[startDate]
        idx2 = self.__ts[endDate] + 1

        assert idx1 < idx2, "Start date must be less than end date"

        if self.__dataset.ndim == 3:
            # For multiple sites
            self.__dataset = self.__dataset[:, :, idx1:idx2]
        elif self.__dataset.ndim == 2:
            # For single site
            self.__dataset = self.__dataset[:, idx1:idx2]
        else:
            raise ValueError("Unsupported dataset shape")

    def get_dataset(self):
        return self.__dataset


if __name__ == "__main__":
    dc = DatasetCreator(colab=False)
    # dc.create_dataset_single_site(12011, ["adultFemaleLice", "probablyNoFish"])
    dc.create_dataset_multiple_sites(["adultFemaleLice", "probablyNoFish"])
    dc.trim_time_horizon("2015 - 5", "2022 - 4")
    print(dc.get_dataset().shape)
    print(dc.get_dataset())
