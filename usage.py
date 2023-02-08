import torch
import os.path as osp
from dig.xgraph.dataset import SynGraphDataset
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dataset = SynGraphDataset('./datasets', 'BA_shapes')
dataset.data.x = dataset.data.x.to(torch.float32)
dataset.data.x = dataset.data.x[:, :1]
dim_node = dataset.num_node_features
dim_edge = dataset.num_edge_features
num_classes = dataset.num_classes
