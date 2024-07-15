import torch


"""
common utils
"""
def make_dataset_path(base_path):
    dataset_path = {}
    dataset_path['cite'] = base_path+'data/cora.cites'
    dataset_path['content'] = base_path+'data/cora.content'
    return dataset_path


def save_checkpoint(file, model, optimizer):
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, file)
    print('model pt file is being saved\n')