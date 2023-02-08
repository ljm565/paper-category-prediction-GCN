import time
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from utils.config import Config
from utils.utils_func import *
from models.model import GCN



class Trainer:
    def __init__(self, config:Config, device:torch.device, mode:str, continuous:int):
        torch.manual_seed(999)
        self.config = config
        self.device = device
        self.mode = mode
        self.continuous = continuous

        # if continuous, load previous training info
        if self.continuous:
            with open(self.config.loss_data_path, 'rb') as f:
                self.loss_data = pickle.load(f)

        # path, data params
        self.base_path = self.config.base_path
        self.model_path = self.config.model_path
        self.data_path = self.config.dataset_path
 
        # train params
        self.directed = bool(self.config.directed)
        self.dynamic = bool(self.config.dynamic)
        self.epochs = self.config.epochs
        self.lr = self.config.lr

        # make necessary data for GCN training
        self.adj, self.feature, self.label = make_gnn_data(self.data_path['cite'], self.data_path['content'], self.dynamic, self.directed)

        # split train / val / test
        self.train_idx, self.val_idx, self.test_idx = split_data(self.feature.size(0))

        # model, optimizer, loss
        input_dim = self.feature.size(1)
        class_num = len(set(self.label.tolist()))
        self.model = GCN(self.config, input_dim, class_num).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        if self.mode == 'train':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=5e-4)
            total_steps = self.epochs
            pct_start = 5 / total_steps
            final_div_factor = self.lr / 25 / 1e-6
            self.scheduler = OneCycleLR(self.optimizer, max_lr=self.lr, total_steps=total_steps, pct_start=pct_start, final_div_factor=final_div_factor)
            if self.continuous:
                self.check_point = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(self.check_point['model'])
                self.optimizer.load_state_dict(self.check_point['optimizer'])
                del self.check_point
                torch.cuda.empty_cache()
        else:
            self.check_point = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(self.check_point['model'])    
            self.model.eval()
            del self.check_point
            torch.cuda.empty_cache()

        
    def training(self):
        early_stop = 0
        best_val_acc = 0 if not self.continuous else self.loss_data['best_val_acc']
        best_epoch_info = 0 if not self.continuous else self.loss_data['best_epoch']
        self.loss_data = {
            'train_history': {'loss': [], 'acc': []}, \
            'val_history': {'loss': [], 'acc': []}
            }

        idx_dict = {'train': self.train_idx, 'val': self.val_idx, 'test': self.test_idx}

        for epoch in range(self.epochs):
            start = time.time()
            print(epoch+1, '/', self.epochs)
            print('-'*10)
            for phase in ['train', 'val']:
                print('Phase: {}'.format(phase))

                idx = idx_dict[phase]
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                self.optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    adj, feature, label = self.adj.to(self.device), self.feature.to(self.device), self.label.to(self.device)
                    output = self.model(adj, feature)
                    loss = self.criterion(output[idx], label[idx])

                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()
                        self.scheduler.step()

                    acc = torch.sum(torch.argmax(output[idx], dim=-1).detach().cpu() == label[idx].detach().cpu()) / len(idx)

                print('Epoch {}: loss: {}, acc: {}'.format(epoch+1, loss.item(), acc.item()))
                
                if phase == 'train':
                    self.loss_data['train_history']['loss'].append(loss.item())
                    self.loss_data['train_history']['acc'].append(acc.item())
                elif phase == 'val':
                    self.loss_data['val_history']['loss'].append(loss.item())
                    self.loss_data['val_history']['acc'].append(acc.item())
            
                    # save best model
                    early_stop += 1
                    if  acc.item() > best_val_acc:
                        early_stop = 0
                        best_val_acc = acc.item()
                        best_epoch = best_epoch_info + epoch + 1
                        self.loss_data['best_epoch'] = best_epoch
                        self.loss_data['best_val_acc'] = best_val_acc
                        save_checkpoint(self.model_path, self.model, self.optimizer)

            print("time: {} s\n".format(time.time() - start))
            print('\n'*2)

            # early stopping
            if early_stop == self.config.early_stop_criterion:
                break

        print('best val acc: {:4f}, best epoch: {:d}\n'.format(best_val_acc, best_epoch))

        return self.loss_data


    def test(self, phase):
        with torch.no_grad():
            self.model.eval()
            idx_dict = {'train': self.train_idx, 'val': self.val_idx, 'test': self.test_idx}
            idx = idx_dict[phase]

            adj, feature, label = self.adj.to(self.device), self.feature.to(self.device), self.label.to(self.device)
            output = self.model(adj, feature)

            loss = self.criterion(output[idx], label[idx])
            acc = torch.sum(torch.argmax(output[idx], dim=-1).detach().cpu() == label[idx].detach().cpu()) / len(idx)

            print('{}set: loss: {}, acc: {}'.format(phase, loss.item(), acc.item()))

        # feature visualization
        pred = torch.argmax(output[idx], dim=-1).detach().cpu().numpy()
        label = label[idx].detach().cpu().numpy()
        output = output[idx].detach().cpu().numpy()
        
        tsne = TSNE()
        x_test_2D = tsne.fit_transform(output)
        x_test_2D = (x_test_2D - x_test_2D.min())/(x_test_2D.max() - x_test_2D.min())

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        plt.setp(ax, xticks=[], yticks=[])
        
        ax[0].scatter(x_test_2D[:, 0], x_test_2D[:, 1], s=10, cmap='tab10', c=label)
        ax[0].set_title("GT visualization")

        ax[1].scatter(x_test_2D[:, 0], x_test_2D[:, 1], s=10, cmap='tab10', c=pred)
        ax[1].set_title("Pred visualization")

        fig.tight_layout()
        plt.savefig(self.config.base_path+'result/'+ self.config.visualize_file_name + '.png')