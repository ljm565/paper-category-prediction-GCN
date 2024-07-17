# Training GCN
여기서는 GCN 모델을 학습하는 가이드를 제공합니다.

### 1. Configuration Preparation
GCN 모델을 학습하기 위해서는 Configuration을 작성하여야 합니다.
Configuration에 대한 option들의 자세한 설명 및 예시는 다음과 같습니다.

```yaml
# base
seed: 0
deterministic: True

# environment config
device: cpu     # examples: [0], cpu, mps, not supported DDP 

# project config
project: outputs/GCN
name: cora

# model config
hidden_dim: 256
dropout: 0.1
directed: False                          # if True, directed graph will be constructed.
dynamic: True                            # if True, [D^-0.5 x A x D^-0.5] will be used to normalize graph or [D^-1 x A] will be used.

# data config
cora_dataset_train: True                 # if True, TU dataset will be loaded automatically.
cora_dataset:
    path: data/
CUSTOM:
    train_data_path: null
    validation_data_path: null
    test_data_path: null

# train config
epochs: 1000
lr: 1e-2

# logging config
common: ['train_loss', 'train_acc', 'validation_loss', 'validation_acc']
```


### 2. Training
#### 2.1 Arguments
`src/run/train.py`를 실행시키기 위한 몇 가지 argument가 있습니다.
* [`-c`, `--config`]: 학습 수행을 위한 config file 경로.
* [`-m`, `--mode`]: [`train`, `resume`] 중 하나를 선택.
* [`-r`, `--resume_model_dir`]: mode가 `resume`일 때 모델 경로. `${project}/${name}`까지의 경로만 입력하면, 자동으로 `${project}/${name}/weights/`의 모델을 선택하여 resume을 수행.
* [`-l`, `--load_model_type`]: [`metric`, `loss`, `last`] 중 하나를 선택.
    * `metric`(default): Valdiation metric이 최대일 때 모델을 resume.
    * `loss`: Valdiation loss가 최소일 때 모델을 resume.
    * `last`: Last epoch에 저장된 모델을 resume.


#### 2.2 Command
`src/run/train.py` 파일로 다음과 같은 명령어를 통해 모델을 학습합니다.
```bash
# training from scratch
python3 src/run/train.py --config configs/config.yaml --mode train

# training from resumed model
python3 src/run/train.py --config config/config.yaml --mode resume --resume_model_dir ${project}/${name}
```
모델 학습이 끝나면 `${project}/${name}/weights`에 체크포인트가 저장되며, `${project}/${name}/args.yaml`에 학습 config가 저장됩니다.