import networkx as nx
import matplotlib.pyplot as plt

from language import *
from learn_abstract_graphs_for_GC import *
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
  if graph_to_label[i] == 1:
    #if graph_to_label[i] == -1:
    labeled_graphs.add(i)



graphs = []

my_graph_to_edges = {}
my_graph_to_nodes = {}
my_graphs = set()
for i in range(len(graph_to_label)):
  my_graphs.add(i)
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



absGraph1 = AbstractGraph()
absGraph1.absNodes = [{0: (0.5, 1.0)}, {0: (0.5, 1.0)}, {}, {}] 
absGraph1.absEdges = [({}, 0, 1), ({1: (0.0, 0.5)}, 0, 2), ({}, 0, 3)]


absGraph = AbstractGraph()
absGraph.absNodes = [{}]
absGraph.absEdges = []



candidate_graphs = {139, 144, 145, 20, 150, 25, 31, 161, 36, 168, 169, 170, 43, 44, 171, 172, 174, 175, 49, 177, 53, 55, 56, 59, 71, 74, 78, 80, 82, 94, 114, 116, 126, 62}


#{169, 74, 43, 78, 20, 55} ([{}, {}] , [({3: (0.0, 0.5)}, 0, 1)]) 38
candidate_graphs = candidate_graphs - {169, 74, 43, 78, 20, 55}

absGraph2 = AbstractGraph()
absGraph2.absNodes = [{}, {}] 
absGraph2.absEdges = [({3: (0.0, 0.5)}, 0, 1)]


#{62} ([{}], []), 23
candidate_graphs = candidate_graphs - {62}

absGraph3 = AbstractGraph()
absGraph3.absNodes = [{}] 
absGraph3.absEdges = []


#{169, 43, 53, 55} ([{2: (0.0, 0.5)}, {}], [({3: (0.0, 0.5)}, 0, 1)]) 36
candidate_graphs = candidate_graphs - {169, 43, 53, 55}

absGraph4 = AbstractGraph()
absGraph4.absNodes = [{2: (0.0, 0.5)}, {}] 
absGraph4.absEdges = [({3: (0.0, 0.5)}, 0, 1)]



#{59, 43, 78, 94} ([{}, {1: (0.0, 0.5)}, {}], [({}, 0, 1), ({}, 0, 2)]) 52
candidate_graphs = candidate_graphs - {59, 43, 78, 94}

absGraph5 = AbstractGraph()
absGraph5.absNodes = [{}, {1: (0.0, 0.5)}, {}] 
absGraph5.absEdges = [({}, 0, 1), ({}, 0, 2)]


#{36, 74, 43, 78, 20, 59, 94} ([{1: (0.0, 0.5)}], []) 16
candidate_graphs = candidate_graphs - {36, 74, 43, 78, 20, 59, 94}

absGraph6 = AbstractGraph()
absGraph6.absNodes = [{1: (0.0, 0.5)}]
absGraph6.absEdges = [] 


#{56, 116, 172, 126} ([{5: (0.0, 0.5)}, {}], []) 44
candidate_graphs = candidate_graphs - {56, 116, 172, 126}

absGraph7 = AbstractGraph()
absGraph7.absNodes = [{5: (0.0, 0.5)}] 
absGraph7.absEdges = []


#{161, 170, 31} ([{}, {}], [({}, 0, 1)]) 50
candidate_graphs = candidate_graphs - {161, 170, 31}

absGraph8 = AbstractGraph()
absGraph8.absNodes = [{}, {}] 
absGraph8.absEdges = [({}, 0, 1)]


#{74, 139, 44, 80, 49, 20, 25} ([{0: (0.0, 0.5), 5: (0.0, 0.5)}], []) 6
candidate_graphs = candidate_graphs - {74, 139, 44, 80, 49, 20, 25}

absGraph9 = AbstractGraph()
absGraph9.absNodes = [{0: (0.0, 0.5), 5: (0.0, 0.5)}] 
absGraph9.absEdges = []


#{71} ([{0: (0.5, 1.0)}], []) 13
candidate_graphs = candidate_graphs - {71}

absGraph10 = AbstractGraph()
absGraph10.absNodes = [{0: (0.5, 1.0)}] 
absGraph10.absEdges = []


# {168, 171, 174, 175, 144, 145, 82, 177, 114, 150} ([{0: (0.0, 0.5)}], []) 3
candidate_graphs = candidate_graphs - {168, 171, 174, 175, 144, 145, 82, 177, 114, 150}
print(candidate_graphs)

absGraph11 = AbstractGraph()
absGraph11.absNodes = [{0: (0.0, 0.5)}] 
absGraph11.absEdges = []


'''
sys.exit()
'''
#{130, 132, 6, 8, 139, 141, 142, 144, 145, 149, 150, 25, 153, 155, 31, 161, 35, 36, 37, 38, 168, 170, 171, 44, 172, 174, 175, 49, 178, 53, 54, 181, 56, 184, 185, 59, 187, 62, 64, 65, 69, 71, 72, 75, 76, 77, 80, 82, 87, 94, 99, 109, 112, 113, 114, 115, 116, 118, 122, 126}


target_graphs = labeled_graphs



parameter = Parameter()
parameter.train_graphs = set()
parameter.X_node = X_node
parameter.X_edge = X_edge
parameter.A = A
parameter.covered_graphs = labeled_graphs - target_graphs
parameter.labeled_graphs = target_graphs 
parameter.chosen_depth = 3 
parameter.graphs = graphs
parameter.my_graphs = my_graphs
#parameter.my_graphs = candidate_graphs
parameter.gamma = 5


print(len(parameter.my_graphs))
#sys.exit()
#subgraphs = eval_abs_graph(absGraph, graphs[116][0], graphs[116][1], A, X_node, X_edge)
#print(len(subgraphs))

#print(len(labeled_graphs))
#my_score = score(absGraph1, parameter)
#print("MyScore : {}".format(my_score))
#(best_abs_graph, myscore) = specify_binary3(absGraph,parameter)
#score(best_abs_graph, parameter)


#print(candidate_graphs)

#sys.exit()


#[{}, {}]
#[({}, 0, 1)]
#map1 = {38: 18, 44: 21, 50: 18, 54: 14, 66: 3, 60: 4, 42: 3, 58: 2, 48: 3, 56: 5, 52: 1}
#map2 = {28: 15, 22: 14, 30: 4, 32: 2, 20: 2}


myset1 = set()
myset2 = set()
correct = 0
for i, graph in enumerate(graphs):
  subgraphs1 = eval_abs_graph(absGraph1, graph[0], graph[1], A, X_node, X_edge)
  subgraphs2 = eval_abs_graph(absGraph2, graph[0], graph[1], A, X_node, X_edge)
  subgraphs3 = eval_abs_graph(absGraph3, graph[0], graph[1], A, X_node, X_edge)
  subgraphs4 = eval_abs_graph(absGraph4, graph[0], graph[1], A, X_node, X_edge)
  subgraphs5 = eval_abs_graph(absGraph5, graph[0], graph[1], A, X_node, X_edge)
  subgraphs6 = eval_abs_graph(absGraph6, graph[0], graph[1], A, X_node, X_edge)
  subgraphs7 = eval_abs_graph(absGraph7, graph[0], graph[1], A, X_node, X_edge)
  subgraphs8 = eval_abs_graph(absGraph8, graph[0], graph[1], A, X_node, X_edge)
  subgraphs9 = eval_abs_graph(absGraph9, graph[0], graph[1], A, X_node, X_edge)
  subgraphs10 = eval_abs_graph(absGraph10, graph[0], graph[1], A, X_node, X_edge)
  subgraphs11 = eval_abs_graph(absGraph11, graph[0], graph[1], A, X_node, X_edge)
  #if not i in candidate_graphs:
  #  continue
  print()
  print("============================")
  print("Graph : {}".format(i)) 
  print("Graph Label : {}".format(graph_to_label[i]))
  print()
  print("Nodes : {}".format(my_graph_to_nodes[i]))
  print("Edges : {}".format(my_graph_to_edges[i]))
  print()
  print("============================")
  #if graph_to_label[i] == 1 and len(subgraphs) == 38:
  #if len(subgraphs) == 38 or len(subgraphs) == 28 or len(subgraphs) == 22:
  print()
  if len(subgraphs1) >= 20 or len(subgraphs2) == 38 or len(subgraphs3) == 23 or len(subgraphs4) == 36  or len(subgraphs5) == 52 or len(subgraphs6) == 16  or len(subgraphs7) == 44  or len(subgraphs8) == 500  or len(subgraphs9) == 6  or len(subgraphs10) == 13 or len(subgraphs11) == 3 :
    myset1.add(i)
  else:
    myset2.add(i)

correct1 = len(myset1 & labeled_graphs)
correct2 = len(myset2 - labeled_graphs)

print(correct1)
print(correct2)


print("Accuracy : {}".format(correct1 + correct2))
print()
print("Case 1")
print(myset1 - labeled_graphs)
print()
print("Case 1")
print(myset2 & labeled_graphs)














