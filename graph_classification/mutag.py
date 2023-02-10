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
#covered_graphs = {0, 7, 10, 17, 147, 21, 24, 152, 27, 28, 157, 32, 34, 40, 43, 47, 52, 60, 63, 68, 71, 96, 98, 101, 111, 127, 122, 3, 5, 133, 135, 136, 11, 12, 14, 15, 151, 156, 30, 158, 162, 163, 164, 165, 166, 42, 173, 46, 182, 183, 186, 66, 79, 81, 91, 93, 95, 102, 105, 106, 107, 120, 121, 124, 22, 26, 29, 160, 45, 50, 51, 57, 58, 67, 70, 73, 84, 85, 89, 90, 92, 100, 103, 11, 104, 172, 48, 176, 82, 116, 148, 86, 56, 126, 59, 125, 94}
#covered_graphs = {0, 7, 10, 17, 147, 21, 24, 152, 27, 28, 157, 32, 34, 40, 43, 47, 52, 60, 63, 68, 71, 96, 98, 101, 111, 127, 3, 5, 133, 135, 136, 11, 12, 14, 15, 151, 156, 30, 158, 162, 163, 164, 165, 166, 42, 173, 46, 182, 183, 186, 66, 79, 81, 91, 93, 95, 102, 105, 106, 107, 120, 121, 124, 22, 26, 29, 160, 45, 50, 51, 57, 58, 67, 70, 73, 84, 85, 89, 90, 92, 100, 103, 117, 104, 172, 48, 176, 82, 116, 148, 86, 56, 126, 59, 125, 94}
#covered_graphs = {0, 3, 5, 7, 9, 10, 11, 12, 14, 15, 17, 19, 20, 21, 22, 24, 26, 27, 28, 29, 30, 31, 32, 34, 36, 40, 42, 43, 45, 46, 47, 48, 50, 51, 52, 56, 57, 58, 59, 60, 62, 63, 66, 67, 68, 70, 71, 73, 74, 79, 81, 82, 84, 85, 86, 89, 90, 91, 92, 93, 94, 95, 96, 98, 100, 101, 102, 103, 104, 105, 106, 107, 111, 114, 116, 117, 120, 121, 124, 125, 126, 127, 133, 135, 136, 144, 145, 147, 148, 151, 152, 156, 157, 158, 160, 161, 162, 163, 164, 165, 166, 170, 172, 173, 175, 176, 182, 183, 186}
#covered_graphs = {0, 3, 5, 7, 9, 10, 11, 12, 14, 15, 17, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 36, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 55, 56, 57, 58, 59, 60, 62, 63, 66, 67, 68, 70, 71, 73, 74, 78, 79, 80, 81, 82, 84, 85, 86, 89, 90, 91, 92, 93, 94, 95, 96, 98, 100, 101, 102, 103, 104, 105, 106, 107, 108, 111, 114, 116, 117, 120, 121, 124, 125, 126, 127, 133, 135, 136, 139, 144, 145, 147, 148, 151, 152, 156, 157, 158, 160, 161, 162, 163, 164, 165, 166, 168, 169, 170, 172, 173, 175, 176, 177, 179, 182, 183, 186}
#covered_graphs = {0, 3, 5, 7, 9, 10, 11, 12, 14, 15, 17, 20, 21, 22, 24, 26, 27, 28, 29, 30, 32, 34, 40, 42, 43, 45, 46, 47, 50, 51, 52, 53, 55, 57, 58, 60, 63, 66, 67, 68, 70, 71, 73, 74, 78, 79, 81, 84, 85, 89, 90, 91, 92, 93, 95, 96, 100, 101, 102, 103, 104, 105, 106, 107, 117, 120, 121, 124, 133, 135, 136, 151, 156, 157, 158, 160, 162, 163, 164, 165, 166, 169, 173, 182, 183, 186, 34, 98, 40, 9, 10, 104, 111, 147, 21, 24, 27, 60, 152, 127}
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


learned_parameters = learn_abs_graphs_top_down(parameter)
with open('learned_parameters_for_1.pickle', 'wb') as f:
  pickle.dump(learned_parameters, f)

sys.exit()





#print(len(labeled_graphs))
#print(len(graph_to_label))



#print(len(labeled_graphs))
#print(len(covered_graphs))
#print(len(labeled_graphs - covered_graphs))
#print(labeled_graphs - covered_graphs)

#sys.exit()

absGraph = AbstractGraph()

absGraph.absNodes = [{}, {}] 
absGraph.absEdges = [({}, 0, 1)]


#topdown
#sys.exit()
#24, 36, 48 
#absGraph.absNodes = [{0: (0.5, 1.0)}, {0: (0.5, 1.0)}, {0: (0.5, 1.0)}, {0: (0.5, 1.0)}] 
#absGraph.absEdges = [({}, 0, 1), ({}, 1, 2), ({}, 1, 3)]

#24
#absGraph.absNodes = [{0: (0.5, 1.0)}, {0: (0.5, 1.0)}, {}, {}]
#absGraph.absEdges = [({}, 0, 1), ({}, 1, 2), ({}, 1, 3)]




#topdown 2
#56
#absGraph.absNodes = [{0: (0.5, 1.0)}, {0: (0.5, 1.0)}, {0: (0.5, 1.0)}]
#absGraph.absEdges = [({}, 0, 1), ({}, 1, 2)]


#48
#absGraph.absNodes = [{0: (0.5, 1.0)}, {0: (0.5, 1.0)}]
#absGraph.absEdges = [({}, 0, 1)]


#44
#absGraph.absNodes = [{0: (0.5, 1.0)}, {0: (0.5, 1.0)}, {}, {}]
#absGraph.absEdges = [({}, 0, 1), ({}, 1, 2), ({}, 1, 3)]

#38
#absGraph.absNodes = [{}, {}]
#absGraph.absEdges = [({}, 0, 1)]

#42?
#absGraph.absNodes = [{0: (0.5, 1.0)}, {0: (0.5, 1.0)}, {0: (0.5, 1.0)}]
#absGraph.absEdges = [({}, 0, 1), ({}, 0, 2)]

#42
#absGraph.absNodes = [{0: (0.5, 1.0)}, {}, {}, {}]
#absGraph.absEdges = [({}, 0, 1), ({}, 0, 2), ({}, 0, 3)]

#52
#absGraph.absNodes = [{}, {}, {}, {}, {}]
#absGraph.absEdges = [({}, 0, 1), ({}, 1, 2), ({}, 1, 3), ({}, 0, 4)]



#eval_abs_graph_on_graphs_val_set2(absGraph, parameter.graphs, parameter.A, parameter.X_node, parameter.X_edge, parameter.labeled_graphs, parameter.left_graphs)

#sys.exit()

print(labeled_graphs)
print(labeled_graphs - covered_graphs)
left_graphs = labeled_graphs - covered_graphs
print()
print(left_graphs)


specify(absGraph, parameter)


sys.exit()
'''
for i, val in enumerate(left_graphs):
  abs_graph = constructAbsGraph(parameter, val)
  print()
  print("AbstractGraph : {}".format(val))
  print(len(abs_graph.absNodes))
  print()
'''

abs_graph= constructAbsGraphUndirected(parameter, 23)


print("AbstractGraph")
print(abs_graph.absNodes)
print(abs_graph.absEdges)




(learned_abs_graph, learned_val) = generalize(abs_graph, parameter)

print("Learned AbstractGraph")
print(learned_abs_graph.absNodes)
print(learned_abs_graph.absEdges)







#eval_abs_graph_on_graphs_val_set2(abs_graph, parameter.graphs, parameter.A, parameter.X_node, parameter.X_edge, parameter.labeled_graphs, parameter.left_graphs)


sys.exit()

'''
absGraph.absNodes = [{0: (1.0, 1.0)}, {1: (1.0, 1.0)}, {1: (1.0, 1.0)}, {0: (1.0, 1.0)}, {0: (1.0, 1.0)}, {0: (1.0, 1.0)}, {0: (1.0, 1.0)}, {0: (1.0, 1.0)}, {0: (1.0, 1.0)}, {0: (1.0, 1.0)}, {1: (1.0, 1.0)}, {2: (1.0, 1.0)}, {2: (1.0, 1.0)}]

absGraph.absEdges = [({1: (1.0, 1.0)}, 0, 1), ({2: (1.0, 1.0)}, 1, 2), ({1: (1.0, 1.0)}, 2, 3), ({1: (1.0, 1.0)}, 3, 4), ({2: (1.0, 1.0)}, 4, 5), ({1: (1.0, 1.0)}, 5, 6), ({2: (1.0, 1.0)}, 6, 7), ({1: (1.0, 1.0)}, 7, 8), ({1: (1.0, 1.0)}, 3, 8), ({2: (1.0, 1.0)}, 8, 9), ({1: (1.0, 1.0)}, 1, 9), ({1: (1.0, 1.0)}, 5, 10), ({2: (1.0, 1.0)}, 10, 11), ({1: (1.0, 1.0)}, 10, 12)]



#absGraph.absEdges = [({}, 0, 1), ({2: (0.5, 1.0)}, 1, 2), ({}, 2, 3), ({}, 3, 4), ({}, 4, 5)]





eval_abs_graph_on_graphs_val_set(absGraph, graphs, A, X_node, X_edge, labeled_graphs, labeled_graphs)

sys.exit()

absGraph = AbstractGraph()
absGraph.absNodes = [{}]
absGraph.absEdges = []

absGraph_valSet = specify(absGraph, parameter)

chosen_graphs = set()
for _, (abs_graph, val_set) in enumerate(absGraph_valSet):
  print()
  print()
  print("AbsGraph")
  print("AbsGraph Nodes : {}".format(abs_graph.absNodes))
  print("AbsGraph Edges : {}".format(abs_graph.absEdges))
  print("Val Set : {}".format(val_set))
  myset = set()
  for _, val in enumerate(val_set):
    for i, graph in enumerate(graphs):
      subgraphs = eval_abs_graph(abs_graph, graph[0], graph[1], A, X_node, X_edge)
      if len(subgraphs) == val:
        chosen_graphs.add(i)
 

print()
print(chosen_graphs)
print(len(chosen_graphs))
print()
print(chosen_graphs & labeled_graphs)
print(len(chosen_graphs & labeled_graphs))
print(len(labeled_graphs))


print(labeled_graphs - chosen_graphs)
 
sys.exit()
'''





#parameter.my_graphs = {0, 128, 8, 139, 13, 142, 16, 17, 18, 19, 20, 144, 145, 150, 153, 154, 155, 159, 33, 35, 36, 37, 168, 169, 43, 44, 171, 174, 175, 49, 177, 178, 181, 54, 184, 185, 60, 62, 65, 68, 74, 80, 87, 88, 96, 99, 112, 114, 118, 119, 122, 125}

case1 = set()
case2 = set()
correct = 0

for i, graph in enumerate(graphs):
  subgraphs = eval_abs_graph(absGraph1, graph[0], graph[1], A, X_node, X_edge)
  #    set2 = set2 + 1
  print()
  print("============================")
  print("Graph : {}".format(i)) 
  print("Graph Label : {}".format(graph_to_label[i]))
  print()
  #print("Sum : {}".format(len(graph[0]) + len(graph[1])))
  print("Nodes : {}".format(my_graph_to_nodes[i]))
  print("Edges : {}".format(my_graph_to_edges[i]))
  print()
  #print("Subgraphs : {}".format(subgraphs))
  #print("Subgraphs1 : {}".format(len(subgraphs1)))
  print("Subgraphs2 : {}".format(len(subgraphs)))
  print("============================")
  print()
  continue
  if graph_to_label[i] == 1 and len(subgraphs) >= 20:
    correct = correct + 1
    print("Correct!!!") 
  elif graph_to_label[i] == -1 and len(subgraphs) < 20:
    correct = correct + 1
    print("Correct!!") 
  
  else:
    print("Not Correct!!") 
    #if len(subgraphs) > 0 :
    '''
    print()
    print("============================")
    print("Graph : {}".format(i)) 
    print("Graph Label : {}".format(graph_to_label[i]))
    print()
    print("Sum : {}".format(len(graph[0]) + len(graph[1])))
    print()
    #print("Subgraphs : {}".format(subgraphs))
    print("Subgraphs : {}".format(len(subgraphs)))
    print("============================")
    print()
    '''
    if graph_to_label[i] == 1:
      case1.add(i)
      #print("Edge_list2 : {}".format(my_graph_to_edges[i]))
    else:
      case2.add(i)

print (correct)
print (case1)
print (case2)
sys.exit()

#subgraphs = eval_abs_graph(absGraph, graphs[116][0], graphs[116][1], A, X_node, X_edge)
#print(len(subgraphs))

#print(len(labeled_graphs))
#my_score = score(absGraph1, parameter)
#print("MyScore : {}".format(my_score))
#(best_abs_graph, myscore) = specify_binary3(absGraph,parameter)
#sys.exit()

#specify_binary2(absGraph,parameter)
#score(best_abs_graph, parameter)
#sys.exit()

#map1 = {74: 5, 50: 0, 100: 3, 34: 0, 140: 5, 58: 0, 84: 5, 46: 3, 76: 11, 104: 5, 112: 6, 52: 1, 98: 4, 114: 4, 36: 2, 80: 3, 48: 1, 64: 4, 124: 7, 170: 1, 39: 1, 106: 2, 122: 3, 92: 4, 96: 4, 54: 1, 108: 3, 72: 11, 56: 2, 78: 4, 136: 2, 62: 0, 86: 2, 27: 0, 128: 1, 32: 0, 126: 7, 120: 1, 162: 2, 94: 0, 42: 1, 28: 0, 102: 2, 83: 0, 31: 0, 29: 0, 79: 0, 40: 0, 110: 1, 88: 1}
#map2 = {74: 1, 50: 7, 100: 0, 34: 6, 140: 0, 58: 7, 84: 0, 46: 6, 76: 0, 104: 0, 112: 0, 52: 3, 98: 0, 114: 0, 36: 3, 80: 1, 48: 3, 64: 1, 124: 0, 170: 0, 39: 0, 106: 0, 122: 0, 92: 0, 96: 0, 54: 3, 108: 0, 72: 1, 56: 3, 78: 0, 136: 0, 62: 2, 86: 1, 27: 1, 128: 0, 32: 4, 126: 0, 120: 0, 162: 0, 94: 1, 42: 2, 28: 1, 102: 0, 83: 1, 31: 1, 29: 2, 79: 1, 40: 1, 110: 0, 88: 0}

'''
myset1 = set()
myset2 = set()
myset3 = set()
gamma = 0.9
for _, val in enumerate(map1):
  if (map1[val] / (map1[val] + map2[val])) > gamma:
    myset1.add(val)
  elif (map2[val] / (map1[val] + map2[val])) > gamma:
    myset2.add(val)
  else:
    myset3.add(val)

print(myset1)
print(myset2)
print(myset3)
my_target_graphs = set()
#sys.exit()

correct = 0
case1 = set() 
case2 = set()

for i, graph in enumerate(graphs):
  subgraphs = eval_abs_graph(absGraph1, graph[0], graph[1], A, X_node, X_edge)
  print()
  print("============================")
  print("Graph : {}".format(i)) 
  print("Graph Label : {}".format(graph_to_label[i]))
  print()
  #print("Sum : {}".format(len(graph[0]) + len(graph[1])))
  print("Nodes : {}".format(my_graph_to_nodes[i]))
  print("Edges : {}".format(my_graph_to_edges[i]))
  print()
  print("Subgraphs : {}".format(len(subgraphs)))
  print("============================")
  print()
  if graph_to_label[i] == 1 and len(subgraphs) in myset1:
    correct = correct + 1
    print("Correct!!!") 
  elif graph_to_label[i] == -1 and len(subgraphs) in myset2:
    correct = correct + 1
    print("Correct!!") 
  if len(subgraphs) in myset3: # ToDo
    my_target_graphs.add(i)
    correct = correct + 1  
  else:
    print("Not Correct!!") 
print (correct)

print(my_target_graphs)
print(len(my_target_graphs))

sys.exit()
'''
print("Target graphs : {}".format(target_graphs))
one = 0
two = 0

#my_set = {144, 145, 20, 150, 25, 36, 168, 171, 44, 172, 175, 177, 53, 56, 62, 74, 78, 80, 82, 114, 116, 126}

#my_set = {139, 144, 145, 20, 150, 25, 36, 168, 169, 171, 44, 175, 49, 177, 55, 62, 74, 78, 80, 82, 114}

#set1 = 0
#set2 = 0

#my_cnt1 = 0
#my_cnt2 = 0
#my_map = {}
#score(absGraph, parameter)
for i, graph in enumerate(graphs):
  if not (i in my_set) and graph_to_label[i] == 1:
    continue
  #if graph_to_label[i] == -1:
  #  continue
  
  #subgraphs1 = eval_abs_graph(absGraph1, graph[0], graph[1], A, X_node, X_edge)
  subgraphs2 = eval_abs_graph(absGraph2, graph[0], graph[1], A, X_node, X_edge)
  #if len(subgraphs2) != 8:
  #  continue 
  #if not (i in set1) and graph_to_label[i] == 1:
  #if not (i in my_set) and graph_to_label[i] == 1:
  #  continue 
  if (len(subgraphs2), graph_to_label[i]) in my_map:
    my_map[(len(subgraphs2), graph_to_label[i])] = my_map[(len(subgraphs2), graph_to_label[i])] + 1
  else:
    my_map[(len(subgraphs2), graph_to_label[i])] = 1 
    

  if graph_to_label[i] == 1:
    my_cnt1 = my_cnt1 + 1
  else:
    my_cnt2 = my_cnt2 + 1


  #else:
  #  if len(subgraphs1) == 16 and graph_to_label[i] == 1:
  #    set1 = set1 + 1
  #  elif len(subgraphs1) == 16 and graph_to_label[i] == -1:
  #    set2 = set2 + 1
  print()
  print("============================")
  print("Graph : {}".format(i)) 
  print("Graph Label : {}".format(graph_to_label[i]))
  print()
  #print("Sum : {}".format(len(graph[0]) + len(graph[1])))
  print("Nodes : {}".format(my_graph_to_nodes[i]))
  print("Edges : {}".format(my_graph_to_edges[i]))
  print()
  #print("Subgraphs : {}".format(subgraphs))
  #print("Subgraphs1 : {}".format(len(subgraphs1)))
  print("Subgraphs2 : {}".format(len(subgraphs2)))
  print("============================")
  print()
  continue
  #if graph_to_label[i] == 1 and len(subgraphs) >= 23:
  if graph_to_label[i] == 1 and len(subgraphs1) >= 20:
  #if graph_to_label[i] == 1 and (len(subgraphs1) >= 15):
    correct = correct + 1
    #print("Edge_list1 : {}".format(my_graph_to_edges[i]))
    print("Correct!!!") 
  #elif graph_to_label[i] == -1 and len(subgraphs) < 23:
  elif graph_to_label[i] == -1 and len(subgraphs1) < 20:
  #elif graph_to_label[i] == -1 and (len(subgraphs1) < 15):
    correct = correct + 1
    print("Correct!!") 
  
  else:
    print("Not Correct!!") 
    #if len(subgraphs) > 0 :
    '''
    print()
    print("============================")
    print("Graph : {}".format(i)) 
    print("Graph Label : {}".format(graph_to_label[i]))
    print()
    print("Sum : {}".format(len(graph[0]) + len(graph[1])))
    print()
    #print("Subgraphs : {}".format(subgraphs))
    print("Subgraphs : {}".format(len(subgraphs)))
    print("============================")
    print()
    '''
    if graph_to_label[i] == 1:
      case1.add(i)
      #print("Edge_list2 : {}".format(my_graph_to_edges[i]))
    else:
      case2.add(i)

print (correct)
print (case1)
print (case2)

print()
print()
print(my_cnt1)
print(my_cnt2)
print()
print(my_map)

sys.exit()

print()
print()

print(one)
print(two)

print()

print(set1)
print(set2)









print("======================================Minseok======================================")
#'''
print()
print()
#25
graph = graphs[83]
nodes = []
for i, val in enumerate(graph[0]):
  nodes.append((val, node_to_label[val]))
print("Nodes : {}".format(nodes))

edges = []
for i, val in enumerate(graph[1]):
  edges.append((A[val], edge_to_label[val]))
print("Edges : {}".format(edges))
print()

sys.exit()
print()
print()
#25
graph = graphs[118]
nodes = []
for i, val in enumerate(graph[0]):
  nodes.append((val, node_to_label[val]))
print("Nodes : {}".format(nodes))

edges = []
for i, val in enumerate(graph[1]):
  edges.append((A[val], edge_to_label[val]))
print("Edges : {}".format(edges))
print()



sys.exit()
'''
absGraph = AbstractGraph()
absGraph.absNodes = [{0: (0.5, 1.0)}, {0: (0.5, 1.0)}]
absGraph.absEdges = [({1: (0.0, 0.5)}, 0, 1)]

subgraphs = eval_abs_graph(absGraph, graphs[20][0], graphs[20][1], A, X_node, X_edge)
print("============================================")
#print(graphs[0])
print(subgraphs)
'''
#

#sys.exit()


absGraph = AbstractGraph()
absGraph.absNodes = [{1: (0.5, 1.0)}, {2: (0.5, 1.0)}, {2: (0.5, 1.0)}]
absGraph.absEdges = [({}, 0, 1), ({}, 0, 2)]


correct = 0
case1 = 0
case2 = 0

#score(absGraph, parameter)
for i, graph in enumerate(graphs):
  subgraphs = eval_abs_graph(absGraph, graph[0], graph[1], A, X_node, X_edge)
  if len(subgraphs) >-1 :
   print()
   print("============================")
   print("Graph : {}".format(i)) 
   print("Graph Label : {}".format(graph_to_label[i]))
   print()
   print("Sum : {}".format(len(graph[0]) + len(graph[1])))
   print()
   #print("Subgraphs : {}".format(subgraphs))
   print("Subgraphs : {}".format(len(subgraphs)))
   print("============================")
   print()










