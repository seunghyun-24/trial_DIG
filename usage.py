import json
from dig.ggraph.dataset import ZINC250k
from torch_geometric.loader import DenseDataLoader
conf = json.load(open('config/rand_gen_zinc250k_config_dict.json'))
dataset = ZINC250k(one_shot=False, use_aug=True)
loader = DenseDataLoader(dataset, batch_size=conf['batch_size'], shuffle=True)
