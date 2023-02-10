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
with open("MUTAG/MUTAG_graph_labels.txt") as file:
  i = 0 
  for line in file.readlines():
    label = line.strip()
    graph_to_label[i] = int(label)
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



labeled_graphs = set()
for i, val in enumerate(graph_to_label):
  #if graph_to_label[i] == 1:
  if graph_to_label[i] == -1:
    labeled_graphs.add(i)

graphs = []

for i in range(len(graph_to_label)):
  new_graph = [] 
  new_graph.append(graph_to_nodes[i])
  new_graph.append(graph_to_edges[i])
  graphs.append(new_graph)  




target_graphs = labeled_graphs
covered_graphs = set()


parameter = Parameter()
parameter.train_graphs = set()
parameter.labeled_graphs = labeled_graphs & train_graphs
parameter.graphs = graphs
parameter.left_graphs = labeled_graphs & train_graphs 
parameter.train_graphs = train_graphs


parameter.expected = 1.2





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

abs_graph = AbstractGraph ()

learned_parameters = learn_abs_graphs_bottom_up(parameter, myMaps)
with open('MUTAG/learned_parameters_for_0.pickle', 'wb') as f:
  pickle.dump(learned_parameters, f)



