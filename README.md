# Paper Category Classification using Graph Convolutional Networks
한국어 버전의 설명은 [여기](./docs/README_ko.md)를 참고하시기 바랍니다.

## Introduction
Node classification is one of the tasks that can be addressed using Graph Neural Networks (GNNs).
This code leverages the Graph Convolutional Networks (GCN) and the citation graph data from [Cora](https://relational.fit.cvut.cz/dataset/CORA) to predict the category of unlabeled papers (graph nodes).
Additionally, the distribution of the learned graph node features is visualized using t-SNE.
For a detailed explanation of the code, please refer to [Cora 데이터와 GCN을 이용한 노드 분류](https://ljm565.github.io/contents/gnn2.html).
<br><br><br>

## Supported Models
### Graph Convolutional Networks (GCN)
* A GCN using `nn.Linear` is implemented (This code is implemented in a naive manner without using PyTorch Geometric or the Deep Graph Library).
<br><br><br>


## Base Dataset
* [Cora Dataset](https://relational.fit.cvut.cz/dataset/CORA).
* Custom datasets can also be used by setting the path in the `config/config.yaml`.
However, implementing a custom dataloader may require additional coding work in `src/utils/data_utils.py`.
<br><br>

## Supported Devices
* CPU, GPU (DDP is not supported), MPS (for Mac and torch>=1.12.0)
<br><br><br>

## Quick Start
```bash
python3 src/run/train.py --config config/config.yaml --mode train
```
<br><br>


## Project Tree
This repository is structured as follows.
```
├── configs                         <- Folder for storing config files
│   └── *.yaml
│
└── src      
    ├── models
    |   └── gcn.py                  <- GCN model file
    |
    ├── run                   
    |   ├── train.py                <- Training execution file
    |   ├── validation.py           <- Trained model evaulation execution file
    |   └── vis_tsne.py             <- Trained model t-SNE visualization execuation file
    | 
    ├── tools                   
    |   ├── model_manager.py          
    |   └── training_logger.py      <- Training logger class file
    |
    ├── trainer                 
    |   ├── build.py                <- Codes for initializing dataset, etc.
    |   └── trainer.py              <- Class for training, evaluating, and visualizing with t-SNE
    |
    └── uitls                   
        ├── __init__.py             <- File for initializing the logger, versioning, etc.
        ├── data_utils.py           <- File defining the custom dataset dataloader
        ├── filesys_utils.py       
        └── training_utils.py     
```
<br><br>

## Tutorials & Documentations
Please follow the steps below to train the GCN.

1. [Getting Started](./docs/1_getting_started.md)
2. [Data Preparation](./docs/2_data_preparation.md)
3. [Training](./docs/3_trainig.md)
4. ETC
   * [Evaluation](./docs/4_model_evaluation.md)
   * [Predicted Feature Visualization using t-SNE](./docs/5_tsne_vis.md)

<br><br><br>


## Training Results
### Directed Graph Trained Results
* Loss History<br>
<img src="docs/figs/directed_loss.png" width="80%"><br><br>

* Accuracy History<br>
<img src="docs/figs/directed_acc.png" width="80%"><br><br>


* Test set accuracy: 0.8600 (87 epoch)<br>
(The test set results of the model when the highest accuracy on the validation set is achieved.)<br><br>

* Test set Feature Distribution<br>
<img src="docs/figs/dynamic_directed.png" width="80%"><br><br>


### Undirected Graph Trained Results
* Loss History<br>
<img src="docs/figs/undirected_loss.png" width="80%"><br><br>

* Accuracy History<br>
<img src="docs/figs/undirected_acc.png" width="80%"><br><br>

* Test set accuracy: 0.8771 (177 epoch)<br>
(The test set results of the model when the highest accuracy on the validation set is achieved.)<br><br>

* Test set Feature Distribution<br>
<img src="docs/figs/dynamic_undirected.png" width="80%"><br><br>

<br><br>

## ETC
This code implements the model in a naive way without using PyTorch Geometric or the Deep Graph Library.
Therefore, it does not include implementations for data loaders (or samplers) and does not train in batches.


<br><br><br>
