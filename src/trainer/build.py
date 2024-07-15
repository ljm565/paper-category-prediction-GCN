
import os

import torch

from models import GCN
from utils import LOGGER, RANK, colorstr
from utils.data_utils import make_gnn_data, split_data

PIN_MEMORY = str(os.getenv('PIN_MEMORY', True)).lower() == 'true'  # global pin_memory for dataloaders



def get_model(config, device):
    model = GCN(config).to(device)
    return model


def build_dataset(config):
    if config.cora_dataset_train:
        cite_data_path = os.path.join(config.cora_dataset.path, 'coara.cites')
        content_data_path = os.path.join(config.cora_dataset.path, 'coara.content')
        adj, feature, label = make_gnn_data(cite_data_path, content_data_path, config.dynamic, config.directed)
        train_idx, val_idx, test_idx = split_data(feature.size(0))
        dataset_dict = {
            'adj': adj,
            'feature': feature,
            'label': label,
            'train': train_idx,
            'validation': val_idx,
            'test': test_idx
        }
        # add paramters to config
        config.input_dim = feature.size(1)
        config.class_num = len(set(label.tolist()))
    else:
        LOGGER.warning(colorstr('yellow', 'You have to implement data pre-processing code..'))
        raise NotImplementedError
    return dataset_dict



def get_data_loader(config, tokenizer, modes, is_ddp=False):
    datasets = build_dataset(config, tokenizer, modes)
    return datasets