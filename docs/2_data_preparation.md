# Data Preparation
Here, we will proceed with a GCN model training tutorial using [Cora Dataset](https://relational.fit.cvut.cz/dataset/CORA) dataset.
Please refer to the following instructions to utilize custom datasets.


### 1. Cora
If you want to train on the Cora dataset, simply set the `cora_dataset_train` value in the `config/config.yaml` file to `True` as follows.
```yaml
cora_dataset_train: True                 # if True, TU dataset will be loaded automatically.
cora_dataset:
    path: data/
CUSTOM:
    train_data_path: null
    validation_data_path: null
    test_data_path: null
```
<br>

### 2. Custom Data
If you want to train on the custom dataset, simply set the `cora_dataset_train` value in the `config/config.yaml` file to `False` as follows.
You have to set your custom training/validation/test datasets.
```yaml
cora_dataset_train: False                # if True, TU dataset will be loaded automatically.
cora_dataset:
    path: data/
CUSTOM:
    train_data_path: null
    validation_data_path: null
    test_data_path: null
```
<br>