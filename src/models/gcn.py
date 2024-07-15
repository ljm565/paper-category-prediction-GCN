import torch
import torch.nn as nn



class GraphConv(nn.Module):
    def __init__(self, in_size, out_size, dropout=None, activation=False):
        super(GraphConv, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dropout = dropout
        self.activation = activation
        
        self.layer = nn.Linear(self.in_size, self.out_size)
        if not self.dropout == None:
            self.dropout_layer = nn.Dropout(self.dropout)
        if self.activation:
            self.relu = nn.ReLU()

        self.init_wts()


    def init_wts(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                stdv = 1/m.weight.size(1)
                nn.init.uniform_(m.weight, -stdv, stdv)


    def forward(self, adj, x):
        x = torch.mm(adj, self.layer(x))
        if self.activation:
            x = self.relu(x)
        if not self.dropout == None:
            x = self.dropout_layer(x)
        return x



class GCN(nn.Module):
    def __init__(self, config):
        super(GCN, self).__init__()
        self.hidden_dim = config.hidden_dim
        self.dropout = config.dropout
        self.input_dim = config.input_dim
        self.class_num = config.class_num
        
        self.gc1 = self.get_layers(self.input_dim, self.hidden_dim, True, True)
        self.gc2 = self.get_layers(self.hidden_dim, self.class_num, False, False)


    def get_layers(self, in_size, out_size, activation=False, dropout=False):
        if dropout:
            layer_list = [GraphConv(in_size, out_size, self.dropout, activation)]
        else:
            layer_list = [GraphConv(in_size, out_size, None, activation)]
        layer_list = nn.ModuleList(layer_list)
        return layer_list
    

    def forward(self, adj, x):
        for l in self.gc1:
            x = l(adj, x)

        for l in self.gc2:
            x = l(adj, x)

        return x