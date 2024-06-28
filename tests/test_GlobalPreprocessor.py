import unittest

import numpy as np

from preprocessing.GlobalPreprocessor import GlobalPreprocessor


class MyTestCase(unittest.TestCase):
    def test_output_uni_1(self):
        arr = np.array([
            [[0.3, 0.3, 0.3, 0.4, 0., 0., 0., 0., 0., 0.]],
            [[0.1, 0.1, 0.2, 0.6, 0., 0., 0., 0., 0., 0.]]
        ])
        preprocessor = GlobalPreprocessor(arr)
        preprocessor.train_val_test_split()
        # preprocessor.standardize_data()
        preprocessor.create_features_and_targets(2, 1)
        X_train, y_train, X_val, y_val, X_test, y_test = preprocessor.get_feature_and_target_datasets()

        expected_X = np.array([
            [
                [0.3, 0.1],
                [0.3, 0.1]
            ],
            [
                [0.3, 0.1],
                [0.3, 0.2]
            ],
            [
                [0.3, 0.2],
                [0.4, 0.6]
            ],
            [
                [0.4, 0.6],
                [0.0, 0.0]
            ],
            [
                [0.0, 0.0],
                [0.0, 0.0]
            ],
            [
                [0.0, 0.0],
                [0.0, 0.0]
            ],
        ])

        expected_y = np.array([
            [0.3, 0.2],
            [0.4, 0.6],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ])

        np.testing.assert_almost_equal(expected_X, X_train, decimal=3)
        self.assertEqual(expected_X.shape, X_train.shape)

        np.testing.assert_almost_equal(expected_y, y_train, decimal=3)
        self.assertEqual(expected_y.shape, y_train.shape)

    def test_output_multi_1(self):
        arr = np.array([
            [[0.3, 0.3, 0.3, 0.4, 0., 0., 0., 0., 0., 0.],
             [0.0, 0.0, 0.0, 0.0, 1., 1., 1., 1., 1., 1.]],
            [[0.1, 0.1, 0.2, 0.6, 0., 0., 0., 0., 0., 0.],
             [0.0, 0.0, 0.0, 0.0, 0., 0., 0., 1., 1., 1.]]
        ])
        preprocessor = GlobalPreprocessor(arr)
        preprocessor.train_val_test_split()
        # preprocessor.standardize_data()
        preprocessor.create_features_and_targets(2, 1)
        X_train, y_train, X_val, y_val, X_test, y_test = preprocessor.get_feature_and_target_datasets()
        print(X_train)
        expected_X = np.array([
            [
                [0.3, 0.0, 0.1, 0.0],
                [0.3, 0.0, 0.1, 0.0]
            ],
            [
                [0.3, 0.0, 0.1, 0.0],
                [0.3, 0.0, 0.2, 0.0]
            ],
            [
                [0.3, 0.0, 0.2, 0.0],
                [0.4, 0.0, 0.6, 0.0]
            ],
            [
                [0.4, 0.0, 0.6, 0.0],
                [0.0, 1.0, 0.0, 0.0]
            ],
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0]
            ],
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0]
            ],
        ])

        expected_y = np.array([
            [0.3, 0.2],
            [0.4, 0.6],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ])

        np.testing.assert_almost_equal(X_train, expected_X, decimal=3)
        self.assertEqual(expected_X.shape, X_train.shape)

        np.testing.assert_almost_equal(expected_y, y_train, decimal=3)
        self.assertEqual(expected_y.shape, y_train.shape)


if __name__ == '__main__':
    unittest.main()
