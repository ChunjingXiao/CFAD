import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class SGC(nn.Module):

    def __init__(self, in_feature, out_feature):
        super(SGC, self).__init__()
        self.fc1 = nn.Linear(in_feature, 512)
        self.out = nn.Linear(512, out_feature)

    def forward(self, x):
        x = self.fc1(x)
        x = self.out(x)
        return x

class GAE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GAE, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, input_dim)

    def forward(self, x, edge_index,is_test,mask=None):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        if is_test:
            noise = torch.randn_like(x) * 0.1
            noise[mask] = 0
            x = torch.relu(x+noise)
        else:
            x = torch.relu(x)
        x = self.conv3(x, edge_index)
        return x