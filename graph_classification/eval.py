import pickle
from fast_language import *
from fast_bottom_up_learning import *
import sys


with open("MUTAG/tr.pickle", 'rb') as f:
  train_graphs = pickle.load(f)

with open("MUTAG/va.pickle", 'rb') as f:
  val_graphs = pickle.load(f)

with open("MUTAG/te.pickle", 'rb') as f:
  test_graphs = pickle.load(f)


graph_to_edges = {}
graph_to_nodes = {}
node_to_graph = {}

with open("MUTAG/MUTAG_graph_indicator.txt") as file:
  i = 0 
  for line in file.readlines():
    graph_idx = line.strip()
    idx = int(graph_idx) - 1
    node_to_graph[i] = idx
    if not idx in graph_to_nodes:
      graph_to_nodes[idx] = []
    graph_to_nodes[idx].append(i)
    graph_to_edges[idx] = []
    i = i+1


A = []
with open("MUTAG/MUTAG_A.txt") as file:
  i = 0
  j = 0
  for line in file.readlines():
    edge = line.strip().split(', ')
    fr_node = int(edge[0]) - 1
    to_node = int(edge[1]) - 1
    A.append((fr_node, to_node))
    if fr_node in graph_to_nodes[j]:
      graph_to_edges[j].append(i)
    elif fr_node in graph_to_nodes[j+1]:
      j = j + 1
      graph_to_edges[j] = [i]
    i = i + 1


graph_to_label = {}
graph_label = {}
with open("MUTAG/MUTAG_graph_labels.txt") as file:
  i = 0 
  for line in file.readlines():
    label = line.strip()
    graph_to_label[i] = int(label)
    if graph_to_label[i] == 1:
      graph_label[i] = 1
    else:
      graph_label[i] = 0
    i = i+1
      

node_to_label = {}
with open("MUTAG/MUTAG_node_labels.txt") as file:
  i = 0 
  for line in file.readlines():
    label = line.strip()
    int_label = int(label) 
    node_to_label[i] = int_label
    i = i+1


X_node = []
for i in range(len(node_to_label)):
  X_node.append([node_to_label[i]])



edge_to_label = {}
with open("MUTAG/MUTAG_edge_labels.txt") as file:
  i = 0 
  for line in file.readlines():
    label = line.strip()
    int_label = int(label)
    edge_to_label[i] = int_label
    i = i+1



X_edge = []
for i in range(len(edge_to_label)):
  X_edge.append([edge_to_label[i]])


labeled_graphs = [set(),set()]

for i, val in enumerate(graph_to_label):
  if graph_to_label[i] == -1:
    labeled_graphs[0].add(i)
  if graph_to_label[i] == 1:
    labeled_graphs[1].add(i)

graphs = []

for i in range(len(graph_to_label)):
  new_graph = [] 
  new_graph.append(graph_to_nodes[i])
  new_graph.append(graph_to_edges[i])
  graphs.append(new_graph)  




target_graphs = labeled_graphs
covered_graphs = set()





myMaps = MyMaps()
myMaps.A = A
myMaps.X_node = X_node
myMaps.X_edge = X_edge

succ_node_to_nodes = {}
pred_node_to_nodes = {}
nodes_to_edge = {}

for idx, val in enumerate(A):
  fr_node = val[0]
  to_node = val[1]
  nodes_to_edge[(fr_node, to_node)] = idx
  if not fr_node in succ_node_to_nodes:
    succ_node_to_nodes[fr_node] = set()
  if not to_node in pred_node_to_nodes:
    pred_node_to_nodes[to_node] = set()
  succ_node_to_nodes[fr_node].add((idx, to_node))
  pred_node_to_nodes[to_node].add((idx, fr_node))

myMaps.succ_node_to_nodes = succ_node_to_nodes
myMaps.pred_node_to_nodes = pred_node_to_nodes
myMaps.nodes_to_edge = nodes_to_edge




val_test_graphs = val_graphs | test_graphs
label_len = 2


graph_scores = {}
for _, node in enumerate(val_test_graphs):
  graph_scores[node] = []
  for i in range(label_len):
    graph_scores[node].append([(0, 0)])






for label in range(2):
  with open('MUTAG/learned_parameters_for_{}.pickle'.format(label), 'rb') as f:
    learned_abs_graphs = pickle.load(f)

  for _, abs_graph in enumerate(learned_abs_graphs):
    print()
    print()
    print("Learned Abstraph")
    print(abs_graph.absNodes)
    print(abs_graph.absEdges)
    chosen_graphs = eval_abs_graph_on_graphs_exist(abs_graph, graphs, myMaps)
    chosen_train_graphs = chosen_graphs & train_graphs
    score = len(chosen_train_graphs & labeled_graphs[label]) / (len(chosen_train_graphs) + 1)
    #print(total_chosen_graphs)
    print("Score : {}".format(score))
    print()
    print()
    print("Chosen val_graphs : {}".format(chosen_graphs & val_graphs))
    chosen_val_test_graphs = chosen_graphs & val_test_graphs

    for _, graph in enumerate(chosen_val_test_graphs):
      graph_scores[graph][label].append((score, abs_graph))

for _, val_test_graph in enumerate(val_test_graphs):
  graph_scores[val_test_graph][0].sort(key=lambda x: x[0], reverse=True)
  graph_scores[val_test_graph][1].sort(key=lambda x: x[0], reverse=True)


left_graphs = set()


def find_max(my_list):
  max_idx = -1
  max_val = 0
  for i in range(len(my_list)):
    if my_list[i] >= max_val:
      max_idx = i
      max_val = my_list[i]
  return max_idx

correct = 0

for _, val_graph in enumerate(val_graphs):
  score0 =  graph_scores[val_graph][0][0][0]
  score1 =  graph_scores[val_graph][1][0][0]
  if score0 == 0 and score1 == 0:
    left_graphs.add(val_graph)
    continue
  score_list = [score0, score1] 
  max_idx = find_max(score_list) 
  if max_idx == graph_label[val_graph]:
    correct = correct + 1

my_lst = [0, 0]

for _, val_graph in enumerate(left_graphs):
  my_lst[graph_label[val_graph]] = my_lst[graph_label[val_graph]] + 1 

default = find_max(my_lst)
correct = correct + my_lst[default]
print()
print()
print("==========================")
print("Val Accuracy : {}".format(float(correct/len(val_graphs))))





correct = 0

for _, test_graph in enumerate(test_graphs):
  score0 =  graph_scores[test_graph][0][0][0]
  score1 =  graph_scores[test_graph][1][0][0]
  if score0 == 0 and score1 == 0:
    if default == graph_label[test_graph]:
      correct = correct + 1
    continue
  score_list = [score0, score1] 
  max_idx = find_max(score_list) 
  if max_idx == graph_label[test_graph]:
    correct = correct + 1
print()
print()
print("Test Accuracy : {}".format(float(correct/len(test_graphs))))
print("==========================")



''' 
print("==========================")
#print(total_chosen_graphs)
print(total_chosen_graphs & test_graphs)
print(total_chosen_graphs & test_graphs & labeled_graphs)
print(test_graphs & labeled_graphs)
'''





