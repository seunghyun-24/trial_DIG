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
  #if graph_to_label[i] == -1:
  if graph_to_label[i] == 1:
    labeled_graphs.add(i)

graphs = []

for i in range(len(graph_to_label)):
  new_graph = [] 
  new_graph.append(graph_to_nodes[i])
  new_graph.append(graph_to_edges[i])
  graphs.append(new_graph)  



#print(len(labeled_graphs))
#print(len(labeled_graphs & train_graphs))




target_graphs = labeled_graphs
covered_graphs = set()


parameter = Parameter()
parameter.train_graphs = set()
parameter.labeled_graphs = labeled_graphs & train_graphs
parameter.graphs = graphs
parameter.left_graphs = labeled_graphs & train_graphs 
parameter.train_graphs = train_graphs







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



print(train_graphs & labeled_graphs)
print(len(train_graphs & labeled_graphs))


#with open('MUTAG/learned_parameters_for_0_MUTAG_bu_choose_middle.pickle', 'rb') as f:
with open('MUTAG/learned_parameters_for_1.pickle', 'rb') as f:
  learned_abs_graphs_for_1 = pickle.load(f)
print(len(learned_abs_graphs_for_1))


s2 = set([0,3,5,7,9,10,11,12,14,15,17,19,21,22,23,24,26,27,28,29,30,32,34,40,42,43,45,46,47,48,50,51,52,53,55,57,58,60,63,66,67,68,70,71,73,78,79,81,82,84,85,86,89,90,91,92,93,95,96,98,100,101,102,103,104,105,106,107,108,111,117,120,121,124,125,127,132,133,135,136,141,147,148,151,152,156,157,158,160,162,163,164,165,166,169,173,174,176,179,182,183,186,187])



#print(len(learned_abs_graphs))
total_chosen_graphs = set()
for _, abs_graph in enumerate(learned_abs_graphs_for_1):
  print()
  print()
  print("Learned Abstraph")
  print(abs_graph.absNodes)
  print(abs_graph.absEdges)
  chosen_graphs = eval_abs_graph_on_graphs_exist(abs_graph, parameter.graphs, myMaps)
  print("BB")
  chosen_train_graphs = chosen_graphs & train_graphs
  score = len(chosen_train_graphs & labeled_graphs) / (len(chosen_train_graphs) + 1)
  total_chosen_graphs = total_chosen_graphs | chosen_graphs
  #print(total_chosen_graphs)
  print("Score : {}".format(score))
  print()
  print()
  print("Chosen train graphs : {}".format(chosen_graphs & train_graphs))
  print("Chosen train labeled graphs : {}".format(chosen_graphs & train_graphs & labeled_graphs))
  print("Chosen train graphs len : {}".format(len(chosen_graphs & train_graphs)))
  print("Chosen train labeled graphs len : {}".format(len(chosen_graphs & train_graphs & labeled_graphs)))
  print()
  print()
  print("Score : {}".format(score))
  print("Chosen test graphs : {}".format(chosen_graphs & test_graphs))
  print("Chosen test labeled graphs : {}".format(chosen_graphs & test_graphs & labeled_graphs))
  print()
  print()

  print("Chosen graphs len : {}".format(len(chosen_graphs)))
  print("Chosen labeled graphs len : {}".format(len(chosen_graphs & labeled_graphs)))
  
  print("Chosen graphs len2 : {}".format(len(chosen_graphs - s2)))
  print("Chosen graphs len2 : {}".format(len(chosen_graphs - s2)))
  print("Chosen labeled graphs len2 : {}".format(((chosen_graphs & labeled_graphs) - s2)))
 
print("==========================")
#print(total_chosen_graphs)
print(total_chosen_graphs & test_graphs)
print(total_chosen_graphs & test_graphs & labeled_graphs)
print(test_graphs & labeled_graphs)


print()
print()
print()
print(total_chosen_graphs)
print(total_chosen_graphs & val_graphs)
print(total_chosen_graphs & val_graphs & labeled_graphs)
print(val_graphs & labeled_graphs)

sys.exit()
print(len(train_graphs & labeled_graphs))





print()
print()
print("Human Insight")
print("Labeled graphs : {}".format(len(labeled_graphs)))
print("Chosen graphs : {}".format(len(s2)))
print("Chosen labeled graphs : {}".format(len(labeled_graphs & s2)))
print()

print()
print()
print("Training set")
print("Train graphs : {}".format(len(train_graphs)))
print("Train labeled graphs : {}".format(len(train_graphs & labeled_graphs)))
print()



sys.exit()

with open('MUTAG/learned_parameters_for_1_MUTAG_bu_choose_middle.pickle', 'rb') as f:
  learned_abs_graphs_for_1 = pickle.load(f)
print(len(learned_abs_graphs_for_1))

with open('MUTAG/learned_parameters_for_0_MUTAG_bu_choose_middle.pickle', 'rb') as f:
  learned_abs_graphs_for_0 = pickle.load(f)
print(len(learned_abs_graphs_for_0))
for _, abs_graph in enumerate(learned_abs_graphs_for_0):
  print()
  print()
  print("Learned Abstraph")
  print(abs_graph.absNodes)
  print(abs_graph.absEdges)
 

