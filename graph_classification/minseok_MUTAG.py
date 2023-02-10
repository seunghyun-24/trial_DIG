import networkx as nx
import matplotlib.pyplot as plt
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
covered_graphs = set()


parameter = Parameter()
parameter.train_graphs = set()
parameter.X_node = X_node
parameter.X_edge = X_edge
parameter.A = A
parameter.labeled_graphs = labeled_graphs & train_graphs
parameter.graphs = graphs
parameter.left_graphs = labeled_graphs & train_graphs 
parameter.node_to_label = node_to_label
parameter.edge_to_label = edge_to_label
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





max_len = 0
min_len = 999

nodes_len = 0
edges_len = 0

graphs_len_list = []
for i, graph in enumerate(parameter.graphs):
  nodes = graph[0]
  edges = graph[1]
  graphs_len_list.append( (i, len(edges)) )

graphs_len_list_sorted = sorted(graphs_len_list, key=lambda tup: tup[1])



graphs = [14,9,36,71,169,43,104]


#print()
#print(graphs)
#print()
#print(graphs_len_list_sorted)










print(train_graphs & labeled_graphs)
print(len(train_graphs & labeled_graphs))



#56
abs_graph = AbstractGraph ()


#abs_graph.absNodes = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
#abs_graph.absEdges = [({}, 0, 1), ({}, 1, 2), ({0: (1.0, 1.0)}, 2, 3), ({}, 3, 4), ({}, 4, 5), ({}, 0, 5), ({}, 4, 6), ({}, 2, 7), ({}, 7, 8), ({}, 8, 9), ({}, 9, 10), ({}, 10, 11), ({}, 11, 12), ({}, 7, 12), ({}, 9, 13)]

abs_graph.absNodes =[{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
abs_graph.absEdges =[({0: (1.0, 1.0)}, 0, 1), ({}, 1, 2), ({}, 2, 3), ({}, 3, 4), ({}, 4, 5), ({}, 0, 5), ({}, 4, 6), ({}, 6, 7), ({}, 7, 8), ({}, 8, 9), ({}, 9, 10), ({}, 10, 11), ({}, 11, 12), ({}, 7, 12)]

#abs_graph.absNodes = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {0: (1.0, 1.0)}, {}, {0: (1.0, 1.0)}]
#abs_graph.absEdges = [({}, 0, 1), ({}, 1, 2), ({}, 2, 3), ({}, 3, 4), ({}, 4, 5), ({}, 0, 5), ({}, 5, 6), ({}, 6, 7), ({}, 6, 8), ({}, 3, 9), ({}, 9, 10), ({}, 10, 11), ({}, 2, 11)]


#abs_graph.absNodes = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
#abs_graph.absEdges = [({}, 0, 1), ({}, 1, 2), ({}, 2, 3), ({}, 3, 4), ({}, 4, 5), ({}, 5, 6), ({}, 6, 7), ({}, 2, 7), ({}, 7, 8), ({}, 8, 9), ({}, 8, 10), ({}, 3, 11), ({}, 11, 12), ({}, 0, 12), ({}, 11, 13), ({}, 13, 14), ({}, 13, 15)]


#abs_graph.absNodes = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
#abs_graph.absEdges = [({}, 0, 1), ({}, 1, 2), ({}, 2, 3), ({}, 3, 4), ({}, 4, 5), ({}, 5, 6), ({}, 6, 7), ({}, 2, 7), ({}, 7, 8), ({}, 8, 9), ({}, 8, 10), ({}, 3, 11), ({}, 11, 12), ({}, 0, 12), ({}, 11, 13), ({}, 13, 14), ({}, 13, 15)]



#abs_graph.absNodes = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
#abs_graph.absEdges = [({}, 0, 1), ({}, 1, 2), ({}, 2, 3), ({}, 3, 4), ({}, 4, 5), ({}, 5, 6), ({}, 6, 7), ({}, 7, 8), ({}, 8, 9), ({}, 9, 10), ({}, 5, 10), ({}, 10, 11), ({}, 2, 11), ({}, 11, 12), ({}, 12, 13), ({}, 0, 13), ({}, 4, 14), ({}, 14, 15), ({}, 14, 16)]

#abs_graph.absNodes = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
#abs_graph.absEdges = [({}, 0, 1), ({}, 1, 2), ({}, 2, 3), ({}, 3, 4), ({}, 4, 5), ({}, 0, 5), ({}, 4, 6), ({}, 2, 7), ({}, 7, 8), ({}, 8, 9), ({}, 9, 10), ({}, 10, 11), ({}, 11, 12), ({}, 7, 12), ({}, 10, 13), ({}, 13, 14), ({}, 13, 15)]


#abs_graph.absNodes = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
#abs_graph.absEdges = [({}, 0, 1), ({}, 1, 2), ({}, 2, 3), ({}, 3, 4), ({}, 4, 5), ({}, 0, 5), ({}, 3, 6), ({}, 6, 7), ({}, 7, 8), ({}, 8, 9), ({}, 9, 10), ({}, 10, 11), ({}, 11, 12), ({}, 12, 13), ({}, 8, 13), ({}, 11, 14), ({}, 14, 15), ({}, 14, 16)]


#abs_graph.absNodes = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
#abs_graph.absEdges = [({}, 0, 1), ({}, 1, 2), ({}, 2, 3), ({}, 3, 4), ({}, 4, 5), ({}, 0, 5), ({}, 4, 6), ({}, 6, 7), ({}, 7, 8), ({}, 8, 9), ({}, 3, 9), ({}, 9, 10), ({}, 10, 11), ({}, 11, 12), ({}, 12, 13), ({}, 8, 13), ({}, 12, 14), ({}, 14, 15), ({}, 14, 16)]



#abs_graph.absNodes = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
#abs_graph.absEdges = [({}, 0, 1), ({}, 1, 2), ({}, 2, 3), ({}, 3, 4), ({}, 4, 5), ({}, 5, 6), ({}, 2, 6), ({}, 6, 7), ({}, 7, 8), ({}, 0, 8), ({}, 7, 9), ({}, 9, 10), ({}, 9, 11)]


#abs_graph.absNodes = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
#abs_graph.absEdges = [({}, 0, 1), ({}, 1, 2), ({}, 2, 3), ({}, 3, 4), ({}, 4, 5), ({}, 0, 5), ({}, 4, 6), ({}, 6, 7), ({}, 7, 8), ({}, 8, 9), ({}, 3, 9), ({}, 9, 10), ({}, 10, 11), ({}, 11, 12), ({}, 12, 13), ({}, 8, 13), ({}, 12, 14), ({}, 14, 15), ({}, 14, 16)]

#10
#34
#subgraphX

abs_graph.absNodes = [{}, {0: (1.0, 1.0)}, {}, {}, {}, {}, {1: (1.0, 1.0)}, {}, {}, {0: (1.0, 1.0)}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
abs_graph.absEdges = [({}, 1, 2), ({}, 2, 3), ({}, 3, 4), ({}, 4, 6), ({}, 6, 7), ({}, 2, 9), ({}, 9, 10), ({}, 10, 11), ({}, 11, 12), ({}, 12, 13), ({}, 13, 15), ({}, 15, 16)]



#best
abs_graph.absNodes = [{}, {0: (1.0, 1.0)}, {0: (1.0, 1.0)}, {}, {}, {}, {}, {0: (1.0, 1.0)}, {}, {}, {}, {0: (1.0, 1.0)}, {0: (1.0, 1.0)}, {}, {}, {}]
abs_graph.absEdges = [({}, 1, 2), ({}, 2, 3), ({0: (1.0, 1.0)}, 3, 4), ({}, 4, 5), ({}, 5, 6), ({}, 6, 7), ({}, 7, 8), ({}, 3, 11), ({}, 11, 12)]


abs_graph = sort_abs_graph_edges(abs_graph)
chosen_graphs = eval_abs_graph_on_graphs_exist(abs_graph, parameter.graphs, myMaps)

print()
print()
print()
print(len(chosen_graphs))
print(len(chosen_graphs & labeled_graphs))
print(len(labeled_graphs))


print()
print()
print()
print(len(chosen_graphs & train_graphs))
print(len(chosen_graphs & labeled_graphs & train_graphs))
print(len(labeled_graphs & train_graphs))

print()
print()
print()
print(len(chosen_graphs & test_graphs))
print(len(chosen_graphs & labeled_graphs & test_graphs))
print(len(labeled_graphs & test_graphs))


left_graphs = train_graphs | val_graphs | test_graphs
graph = btm_up_graph_chooser(parameter.left_graphs, parameter.graphs) 
sys.exit()



abs_graph = constructAbsGraphUndirected(parameter, 34)

print()
print()
print()
print()
print()
print()
print("Given graph")
print("Nodes : {}".format(parameter.graphs[graph][0]))
print("Nodes Len : {}".format(len(parameter.graphs[graph][0])))
print("Edges : {}".format(parameter.graphs[graph][1]))
print("Edges Len : {}".format(len(parameter.graphs[graph][1])))
print()
print()
print("AbsGraph")
print("AbsNodes : {}".format(abs_graph.absNodes))
print("AbsEdges : {}".format(abs_graph.absEdges))
sys.exit()

#tests = [167, 172, 88, 106, 186, 154, 185, 141, 22, 142, 34, 29, 60, 146, 42, 124, 95, 33, 149, 10]

#for _, val in enumerate(tests):
#  print(graph_to_label[val])


sys.exit()


#with open('learned_parameters_for_1_MUTAG_bu_connected.pickle', 'rb') as f:
with open('learned_parameters_for_1_MUTAG_bu_connected_2.pickle', 'rb') as f:
  learned_abs_graphs = pickle.load(f)

print(len(learned_abs_graphs))



print(len(learned_abs_graphs))
total_chosen_graphs = set()
#my_else = {141, 146, 149, 88, 154}
my_else = {124, 10, 172}
print("AAAAA")
#'''
for _, abs_graph in enumerate(learned_abs_graphs):
  print()
  print()
  print("Learned Abstraph")
  print(abs_graph.absNodes)
  print(abs_graph.absEdges)
  chosen_graphs = eval_abs_graph_on_graphs_exist(abs_graph, parameter.graphs, myMaps)
  print("BB")
  chosen_train_graphs = chosen_graphs & train_graphs
  score = len(chosen_train_graphs & labeled_graphs) / (len(chosen_train_graphs) + 1)
  #total_chosen_graphs = total_chosen_graphs | chosen_graphs
  #print(total_chosen_graphs)
  print("Score : {}".format(score))
  print()
  print()
  print("Chosen train graphs : {}".format(chosen_graphs & train_graphs))
  print("Chosen train labeled graphs : {}".format(chosen_graphs & train_graphs & labeled_graphs))
  print()
  print()

  print("Chosen test graphs : {}".format(chosen_graphs & test_graphs))
  print("Chosen test labeled graphs : {}".format(chosen_graphs & test_graphs & labeled_graphs))
  #if score > 0.96:
  if score > 0.9:
    total_chosen_graphs = total_chosen_graphs | chosen_graphs
  if len(chosen_graphs & my_else) > 0:
    print("Here")
    print(chosen_graphs & my_else)
print("==========================")
#print(total_chosen_graphs)
print(total_chosen_graphs & test_graphs)
print(total_chosen_graphs & test_graphs & labeled_graphs)
print(test_graphs & labeled_graphs)
print(test_graphs - (test_graphs & labeled_graphs))
print(len(test_graphs - (test_graphs & labeled_graphs)))
sys.exit()
#'''


#chosen_graphs = {0, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 36, 37, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 55, 56, 57, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 73, 74, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 116, 117, 118, 119, 120, 121, 122, 124, 125, 126, 127, 128, 130, 131, 132, 133, 135, 136, 137, 139, 141, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 182, 183, 186, 187}
chosen_graphs = {0, 3, 5, 7, 9, 10, 11, 12, 14, 15, 17, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 34, 36, 40, 42, 43, 45, 46, 50, 51, 52, 56, 57, 58, 59, 60, 63, 65, 66, 67, 68, 70, 71, 73, 74, 79, 81, 84, 85, 86, 89, 90, 91, 92, 93, 95, 96, 98, 100, 101, 102, 103, 104, 105, 106, 107, 108, 111, 116, 117, 120, 121, 124, 125, 126, 127, 133, 135, 136, 139, 145, 147, 148, 151, 152, 156, 157, 158, 160, 161, 162, 163, 164, 165, 166, 169, 170, 172, 173, 176, 179, 182, 183, 186}


print("Test graphs : {}".format(len(test_graphs)))
print("Chosen test graphs : {}".format(len(test_graphs & chosen_graphs)))
print("Chosen labeled test graphs : {}".format(len(test_graphs & labeled_graphs & chosen_graphs)))

#print(test_graphs & labeled_graphs)
#print(test_graphs & chosen_graphs)
#print(test_graphs & labeled_graphs & chosen_graphs)



#print((chosen_graphs & test_graphs) - labeled_graphs)
#{141, 146, 149, 88, 154}
#learned_abs_graph = generalize(new_abs_graph, parameter)






