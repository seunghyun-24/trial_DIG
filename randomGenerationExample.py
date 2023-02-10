import json
from dig.ggraph.dataset import ZINC250k
from torch_geometric.loader import DenseDataLoader
conf = json.load(open('config/rand_gen_zinc250k_config_dict.json'))
dataset = ZINC250k(one_shot=False, use_aug=True)
loader = DenseDataLoader(dataset, batch_size=conf['batch_size'], shuffle=True)

import torch
USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("다음 기기로 학습합니다:", device)

from dig.ggraph.method import GraphDF
runner = GraphDF()
lr = 0.001
wd = 0
max_epochs = 10
save_interval = 1
save_dir = 'rand_gen_zinc250k'
runner.train_rand_gen(loader=loader, lr=lr, wd=wd, max_epochs=max_epochs,
    model_conf_dict=conf['model'], save_interval=save_interval, save_dir=save_dir)
model_conf_dict.to(f'cuda:{model.device_ids[0]}')

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
ckpt_path = 'rand_gen_zinc250k/rand_gen_ckpt_10.pth'
n_mols = 100
mols, _ = runner.run_rand_gen(model_conf_dict=conf['model'], checkpoint_path=ckpt_path,
    n_mols=n_mols, atomic_num_list=conf['atom_list'])

from dig.ggraph.evaluation import RandGenEvaluator
evaluator = RandGenEvaluator()
input_dict = {'mols': mols}
print('Evaluating...')
evaluator.eval(input_dict)
