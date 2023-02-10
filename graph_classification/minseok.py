import networkx as nx
import matplotlib.pyplot as plt
import pickle
from language import *
#from learn_abstract_graphs_for_GC import *
from top_down_learning import *
from bottom_up_learning import *
import sys



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


max_node_label = 0
node_to_label = {}
with open("MUTAG/MUTAG_node_labels.txt") as file:
  i = 0 
  for line in file.readlines():
    label = line.strip()
    int_label = int(label) 
    node_to_label[i] = int_label
    i = i+1
    if int_label > max_node_label:
      max_node_label = int_label


X_node = []
for i in range(len(node_to_label)):
  X_node.append([])
  for _ in range(max_node_label+1):
    X_node[i].append(0)


for _, val in enumerate(node_to_label):
  X_node[val][node_to_label[val]] = 1

#print(X_node)



max_edge_label = 0
edge_to_label = {}
with open("MUTAG/MUTAG_edge_labels.txt") as file:
  i = 0 
  for line in file.readlines():
    label = line.strip()
    int_label = int(label)
    edge_to_label[i] = int_label
    i = i+1
    if int_label > max_edge_label:
      max_edge_label = int_label



X_edge = []
for i in range(len(edge_to_label)):
  X_edge.append([])
  for _ in range(max_edge_label+1):
    X_edge[i].append(0)


for _, val in enumerate(edge_to_label):
  X_edge[val][edge_to_label[val]] = 1


labeled_graphs = set()
for i, val in enumerate(graph_to_label):
  #if graph_to_label[i] == -1:
  if graph_to_label[i] == 1:
    labeled_graphs.add(i)



graphs = []

my_graph_to_edges = {}
my_graph_to_nodes = {}

for i in range(len(graph_to_label)):
  new_graph = [] 
  new_graph.append(graph_to_nodes[i])
  new_graph.append(graph_to_edges[i])
  graphs.append(new_graph)  
  my_graph_to_edges[i] = [0,0,0,0]
  my_graph_to_nodes[i] = [0,0,0,0,0,0,0]
  for j, edge in enumerate(graph_to_edges[i]):
    my_graph_to_edges[i][edge_to_label[edge]] = my_graph_to_edges[i][edge_to_label[edge]] + 1 
  for j, node in enumerate(graph_to_nodes[i]):
    my_graph_to_nodes[i][node_to_label[node]] = my_graph_to_nodes[i][node_to_label[node]] + 1 

target_graphs = labeled_graphs
#covered_graphs = {3, 5, 133, 135, 136, 11, 12, 14, 15, 151, 156, 30, 158, 162, 163, 164, 165, 166, 42, 173, 46, 182, 183, 66, 79, 81, 91, 93, 95, 102, 105, 106, 107, 120, 124}
covered_graphs = set()


parameter = Parameter()
parameter.train_graphs = set()
parameter.X_node = X_node
parameter.X_edge = X_edge
parameter.A = A
parameter.covered_graphs = covered_graphs 
parameter.labeled_graphs = labeled_graphs 
parameter.chosen_depth = 4 
parameter.graphs = graphs
parameter.left_graphs = labeled_graphs - covered_graphs
parameter.node_to_label = node_to_label
parameter.edge_to_label = edge_to_label
parameter.gamma = 5 
parameter.my_gamma = 10
parameter.my_cache = set()

#print(labeled_graphs)
#print(graphs[128])
#abs_graph = constructAbsGraphUndirected(parameter, 128)
#print(abs_graph.absNodes)
#print(abs_graph.absEdges)
#sys.exit()



#learned_parameters = learn_abs_graphs_top_down(parameter)
#with open('learned_parameters_for_1.pickle', 'wb') as f:
#  pickle.dump(learned_parameters, f)

#'''
#24
abs_graph = AbstractGraph ()
#abs_graph.absNodes = [{0: (1.0, 1.0)}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
#abs_graph.absEdges = [({0: (1.0, 1.0)}, 3, 4), ({0: (1.0, 1.0)}, 4, 5), ({0: (1.0, 1.0)}, 4, 6), ({}, 6, 7), ({}, 7, 8), ({}, 8, 9), ({0: (1.0, 1.0)}, 9, 10)]


abs_graph.absNodes =[{0: (1.0, 1.0)}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
abs_graph.absEdges =[({0: (1.0, 1.0)}, 0, 1), ({}, 1, 2), ({}, 2, 3), ({}, 3, 4), ({}, 4, 5), ({}, 0, 5), ({}, 4, 6), ({}, 6, 7), ({}, 7, 8), ({}, 8, 9), ({}, 9, 10), ({}, 10, 11), ({}, 11, 12), ({}, 7, 12)]
#abs_graph.absNodes = [{2: (0.5, 1.0)}, {1: (0.5, 1.0)}, {2: (0.5, 1.0)}, {0: (0.5, 1.0)}, {0: (0.5, 1.0)}, {0: (0.5, 1.0)}, {0: (0.5, 1.0)}, {0: (0.5, 1.0)}, {0: (0.5, 1.0)}]
#abs_graph.absEdges = [({}, 0, 1), ({}, 1, 2), ({}, 1, 3), ({}, 3, 4), ({}, 4, 5), ({}, 5, 6), ({}, 6, 7), ({}, 7, 8)]
#print(my_connect(abs_graph))

#abs_graph.absNodes = [{}, {}]
#abs_graph.absEdges = [({}, 0, 1)]

#abs_graph.absNodes = [{0: (1.0, 1.0)}, {0: (1.0, 1.0)}, {0: (1.0, 1.0)}, {0: (1.0, 1.0)}, {0: (1.0, 1.0)}, {0: (1.0, 1.0)}, {}, {}, {}, {2: (1.0, 1.0)}, {0: (1.0, 1.0)}, {}, {}, {}, {}, {}, {}, {}, {}]
#abs_graph.absEdges = [({0: (1.0, 1.0)}, 0, 1), ({}, 1, 2), ({}, 2, 3), ({}, 3, 4), ({}, 4, 5), ({}, 0, 5), ({}, 9, 10), ({}, 10, 11)]

#subgraphs = eval_abs_graph(abs_graph, graph[0], graph[1], A, X_node, X_edge)
graphs = eval_abs_graph_on_graphs_exist(abs_graph, parameter.graphs, parameter.A, parameter.X_node, parameter.X_edge)

print(graphs)
print(len(graphs))
print(graphs & labeled_graphs)
print(len(graphs & labeled_graphs))

#'''
'''
abs_graph = AbstractGraph ()
abs_graph.absNodes = [{0: (0.5, 1.0)}, {0: (0.5, 1.0)}, {0: (0.5, 1.0)}]
abs_graph.absEdges = [({}, 0, 1), ({}, 0, 2)]


#subgraphs = eval_abs_graph(abs_graph, graph[0], graph[1], A, X_node, X_edge)
graphs = eval_abs_graph_on_graphs_val_set2(abs_graph, parameter.graphs, parameter.A, parameter.X_node, parameter.X_edge, parameter.labeled_graphs, parameter.labeled_graphs)

#print(graphs)
#print(len(graphs))
'''
abs_g = constructAbsGraphUndirected(parameter, 84)

print(abs_g.absNodes)
print(abs_g.absEdges)


