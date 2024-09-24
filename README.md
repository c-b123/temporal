# Temporal
Temporal is a Python library for transforming the salmon lice data provided by BarentsWatch into a supervised machine 
learning problem.

## Installation on Google Colab
```python
url = f"https://github.com/c-b123/temporal.git"
!pip install git+$url
```

## Requirements
All requirements are listed in the requirements.txt.

## Example Usage
The following shows how timeseries data can be loaded with the class DatasetCreator.
```python
from temporal.dataset import DatasetCreator

# Create an instance of the DatasetCreator
dc = DatasetCreator()

# Fetch and create parallel time series for all 58 aquaculture sites in production area 2
dc.create_dataset_multiple_sel_sites(feature_list = ["adultFemaleLice", "probablyNoFish", "seaTemperature"])

# Get parallel timeseries
timeseries = dc.get_dataset()
```
The following shows how the loaded timeseries is transformed into a supervised machine learning problem using the
sliding window approach.
```python
from temporal.preprocessing import GlobalPreprocessor

# Create an instance of the GlobalPreprocessor class
pp = GlobalPreprocessor(timeseries)

# Split the data into training, validation, and test data
pp.train_val_test_split(val_ratio=0.05, test_ratio=0.05)

# Standardize the data
pp.standardize_data()

# Create features and targets with the sliding window approach
pp.create_features_and_targets(input_window=8, offset=1)

# Get the datasets
X_train, y_train, X_val, y_val, X_test, y_test = pp.get_feature_and_target_datasets()
```
The classes SinglePreprocessor and MultiplePreprocessor work similar to the GlobalPreprocessor class. The 
SinglePreprocessor class can be use if interested only in data from a single aquaculture site. The MultiplePreprocessor
class is suitable if interested in data from all aquaculture sites but in a sequential manner (hybrid model).