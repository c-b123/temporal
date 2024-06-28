import unittest
import numpy as np
import pandas as pd

from pathlib import Path

from dataset import DatasetCreator


class TestDatasetCreator(unittest.TestCase):

    def setUp(self):
        self.localityNo = 12965
        self.fill_up = 196

        # Read external lice dataset
        file_path = Path("C:/Users/chris/OneDrive - Universität Zürich UZH/Master of Science/MasterThesis"
                         "/Resources/lice_counts_norway_2012-2023.csv")
        df = pd.read_csv(file_path, on_bad_lines="warn")

        # Filter for localityNo of interest
        df = df[df["Lokalitetsnummer"] == self.localityNo]

        # Create numpy array for adult female lice
        df["Voksne hunnlus"] = df["Voksne hunnlus"].fillna(0)
        adult_female_lice = np.flip(df["Voksne hunnlus"].to_numpy())
        adult_female_lice = np.concatenate([adult_female_lice, np.zeros(self.fill_up)])
        self.adult_female_lice = adult_female_lice

        # Create numpy array for mobile lice
        df["Lus i bevegelige stadier"] = df["Lus i bevegelige stadier"].fillna(0)
        mobile_lice = np.flip(df["Lus i bevegelige stadier"].to_numpy())
        mobile_lice = np.concatenate([mobile_lice, np.zeros(self.fill_up)])
        self.mobile_lice = mobile_lice

        # Create numpy array for probably no fish
        mapping = {"Ja": 1, "Nei": 0}
        df['Trolig uten fisk'] = df['Trolig uten fisk'].map(mapping)
        probably_no_fish = np.flip(df["Trolig uten fisk"].to_numpy())
        probably_no_fish = np.concatenate([probably_no_fish, np.zeros(self.fill_up)])
        self.probably_no_fish = probably_no_fish

    def test_create_dataset_single_site_1(self):
        dc = DatasetCreator(colab=False)
        dc.create_dataset_single_site(["adultFemaleLice"], self.localityNo)

        actual = dc.get_dataset()
        expected = np.array([self.adult_female_lice])

        self.assertEqual(expected.shape, actual.shape)
        np.testing.assert_almost_equal(actual, expected, decimal=3)

    def test_create_dataset_single_site_2(self):
        dc = DatasetCreator(colab=False)
        dc.create_dataset_single_site(["adultFemaleLice", "mobileLice", "probablyNoFish"], self.localityNo)

        actual = dc.get_dataset()
        expected = np.array([self.adult_female_lice, self.mobile_lice, self.probably_no_fish])

        self.assertEqual(expected.shape, actual.shape)
        np.testing.assert_almost_equal(actual, expected, decimal=3)

    def test_create_dataset_multiple_sites_1(self):
        dc = DatasetCreator(colab=False)
        dc.create_dataset_multiple_sites(["adultFemaleLice"])

        actual = dc.get_dataset()
        expected_shape = (80, 1, 626)
        expected = np.array([self.adult_female_lice])

        self.assertEqual(expected_shape, actual.shape)
        np.testing.assert_almost_equal(actual[35], expected, decimal=3)

    def test_create_dataset_multiple_sites_2(self):
        dc = DatasetCreator(colab=False)
        dc.create_dataset_multiple_sites(["adultFemaleLice", "mobileLice", "probablyNoFish"])

        actual = dc.get_dataset()
        expected_shape = (80, 3, 626)
        expected = np.array([self.adult_female_lice, self.mobile_lice, self.probably_no_fish])

        self.assertEqual(expected_shape, actual.shape)
        np.testing.assert_almost_equal(actual[35], expected, decimal=3)


if __name__ == '__main__':
    unittest.main()
