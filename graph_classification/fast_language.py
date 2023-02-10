import copy
import sys 
import json


class MyMaps:
  def __init__(self):
    succ_node_to_nodes = {}
    pred_node_to_nodes = {}
    nodes_to_edge = {}
    A = []
    X_node = []
    X_edge = []

class Parameter:
  def __init__(self):
    labeled_graphs = set()
    chosen_depth = 1
  

class Graph:
  def __init__(self):
    self.Nodes = set()
    self.Edges = {}
    self.Idx_node_map = {}
  #Nodes = set()
  #Edges = {}
  #Idx_node_map = {}




def takeNodes (graph, nodes):
  nodes = list(nodes)
  nodes.sort()
  for i in range(len(nodes)):
    graph.Idx_node_map[nodes[i]] = i

  for i in range(len(nodes)):
    graph.Nodes.add(graph.Idx_node_map[nodes[i]])
    graph.Edges[graph.Idx_node_map[nodes[i]]] = set()

def addEdge(graph, fr, to):
  if not (graph.Idx_node_map[fr] in graph.Edges):
    graph.Idx_node_map[fr] = set()
  if not (graph.Idx_node_map[to] in graph.Edges):
    graph.Idx_node_map[to] = set()

  graph.Edges[graph.Idx_node_map[fr]].add(graph.Idx_node_map[to])
  graph.Edges[graph.Idx_node_map[to]].add(graph.Idx_node_map[fr])


def DFS(src, edges):
  visited_nodes = set([src])
  for i in range(len(edges)):
    adj_nodes = set()
    for _,node in enumerate(visited_nodes):
      adj_nodes = adj_nodes | edges[node]
    visited_nodes = visited_nodes | adj_nodes
  return len(visited_nodes)

def IsConnected(graph):
  nodes = graph.Nodes
  edges = graph.Edges

  visited_nodes_len = DFS(0, edges)
  return (visited_nodes_len == len(nodes) )


class AbstractGraph:
  def __init__(self):
    self.absNodes = []
    self.absEdges = []


def eval_abs_node(abs_node, nodes, X_node):
  filtered_nodes = set()
  if len(abs_node) == 0:
    return nodes
  for _, node in enumerate(nodes):
    flag = True
    for _, feat_idx in enumerate(abs_node):
      (bot, top) = abs_node[feat_idx]
      if X_node[node][feat_idx] < bot or top < X_node[node][feat_idx]:
        flag = False
        break
    if flag == True:
      filtered_nodes.add(node)
  return filtered_nodes


def eval_abs_edge(abs_edge, edges, X_edge):
  filtered_edges = set()
  if len(abs_edge) == 0:
    return edges
  (itvs, p, q) = abs_edge 
  for _, edge in enumerate(edges):
    flag = True
    for _, feat_idx in enumerate(itvs):
      (bot, top) = itvs[feat_idx]
      if X_edge[edge][feat_idx] < bot or top < X_edge[edge][feat_idx]:
        flag = False
        break
    if flag == True:
      filtered_edges.add(edge)
  return filtered_edges

def abs_graph_size (abs_graph):
  size = 0
  size = size + len(abs_graph.absNodes)
  size = size + len(abs_graph.absEdges)
  return size


def eval_abs_graph_on_graphs(abs_graph, abs_graph_val, graphs, A, X_node, X_edge):
  chosen_graphs = set()
  for i, graph in enumerate(graphs):
    subgraphs = eval_abs_graph(abs_graph, graph[0], graph[1], A, X_node, X_edge)
    subgraphs_len = len(subgraphs)
    if subgraphs_len == abs_graph_val:
      chosen_graphs.add(i)
    sys.exit()
  return chosen_graphs




def eval_abs_graph_on_graphs_exist(abs_graph, graphs, my_maps):
  chosen_graphs = set()
  for i, graph in enumerate(graphs):
    #sys.exit()
    if eval_abs_graph_DFS(abs_graph, graph, my_maps):
      chosen_graphs.add(i)
  return chosen_graphs




def eval_abs_graph_Test(abs_graph, graphs, A, X_node, X_edge, labeled_graphs, my_graphs):

  correct_map = {}
  incorrect_map = {}
  val_set = set()
  
  max_score = 0
  max_score_val = 0
 
  for i, graph in enumerate(graphs):
    subgraphs = eval_abs_graph(abs_graph, graph[0], graph[1], A, X_node, X_edge)
    subgraphs_len = len(subgraphs)
    if subgraphs_len == 0:
      continue
    if i in labeled_graphs:
      if not subgraphs_len in incorrect_map:
        incorrect_map[subgraphs_len] = set()

      if subgraphs_len in correct_map:
        correct_map[subgraphs_len].add(i)
      else:
        correct_map[subgraphs_len] = set([i])

    else:
      if not subgraphs_len in correct_map:
        correct_map[subgraphs_len] = set() 
      if subgraphs_len in incorrect_map:
        incorrect_map[subgraphs_len].add(i)
      else:
        incorrect_map[subgraphs_len] = set([i])
  
  score_sum = 0 
  for _, val in enumerate(correct_map):
    #my_score = (len(correct_map[val]) / (len(incorrect_map[val]) + 1))
    if len(my_graphs & correct_map[val]) == 0:
      correct_map[val] = set([])
      my_score = 0
    else:
      #my_score = len(correct_map[val]) - len(incorrect_map[val])
      #F1 score
      correct_graphs_len = len(correct_map[val])
      incorrect_graphs_len = len(incorrect_map[val])
      accuracy =  correct_graphs_len/(correct_graphs_len + incorrect_graphs_len + 1)
      #recall =  correct_graphs_len/len(labeled_graphs)
      #my_score = (2 * (accuracy * recall))/(accuracy + recall)
      my_score = accuracy
      #my_score = len(correct_map[val]) - len(incorrect_map[val])
    
    if my_score > 0:
      val_set.add(val)
    if my_score > max_score:
      max_score = my_score
      max_score_val = val
  print()
  print()
  print("Correct Map:")
  print(correct_map)
  print()
  print("Incorrect Map:")
  print(incorrect_map)
  print()
  print()
  print("Max Score : {}".format(max_score)) 
  print("Max Score val : {}".format(max_score_val)) 
  print()
  print()
  return (max_score, max_score_val) 


def eval_abs_graph_on_graphs_GC(abs_graph, graphs, labeled_graphs, left_graphs, train_graphs, my_maps):
  correct_set = set()
  incorrect_set = set()
  #sys.exit() 
  max_score = 0

  #abs_nodes_len = len(abs_graph.absNodes)
  abs_edges_len = len(abs_graph.absEdges)
 
  for i, graph in enumerate(graphs):
    #print(i)
    if not (i in train_graphs):
      continue
    #nodes_len = len(graph[0])
    edges_len = len(graph[1])
    #print("Graph Idx : {}".format(i))
    if (abs_edges_len > edges_len):
      continue
    exists = eval_abs_graph_DFS(abs_graph, graph, my_maps)

    if exists:
      if (i in labeled_graphs):
        correct_set.add(i)
      else:
        incorrect_set.add(i)

  if len(left_graphs & correct_set) == 0:
    print("There is no new one")
    return 0.0

  correct_graphs_len = len(correct_set)
  incorrect_graphs_len = len(incorrect_set)
  accuracy =  correct_graphs_len/(correct_graphs_len + incorrect_graphs_len + 1)
  score = accuracy
  #print(score)
  #print(correct_graphs_len)
  #print(incorrect_graphs_len)
  #sys.exit()
    
  #if my_score > max_score:
  #  max_score = my_score
  '''
  print()
  print()
  print("Correct Map:")
  print(correct_map)
  print()
  print("Incorrect Map:")
  print(incorrect_map)
  print()
  print()
  print("Max Score : {}".format(max_score)) 
  print("Max Score val : {}".format(max_score_val)) 
  print()
  print()
  #'''
  #return max_score 
  return score 




def eval_abs_graph_on_graphs_val_top_down(abs_graph, graphs, A, X_node, X_edge, labeled_graphs, my_graphs):

  correct_map = {}
  incorrect_map = {}
  val_set = set()
  
  max_score = 0
  max_score_val = 0
 
  for i, graph in enumerate(graphs):
    subgraphs = eval_abs_graph(abs_graph, graph[0], graph[1], A, X_node, X_edge)
    subgraphs_len = len(subgraphs)
    if subgraphs_len == 0:
      continue
    if i in labeled_graphs:
      if not subgraphs_len in incorrect_map:
        incorrect_map[subgraphs_len] = set()

      if subgraphs_len in correct_map:
        correct_map[subgraphs_len].add(i)
      else:
        correct_map[subgraphs_len] = set([i])

    else:
      if not subgraphs_len in correct_map:
        correct_map[subgraphs_len] = set() 
      if subgraphs_len in incorrect_map:
        incorrect_map[subgraphs_len].add(i)
      else:
        incorrect_map[subgraphs_len] = set([i])
  
  score_sum = 0 
  for _, val in enumerate(correct_map):
    #my_score = (len(correct_map[val]) / (len(incorrect_map[val]) + 1))
    if len(my_graphs & correct_map[val]) == 0:
      correct_map[val] = set([])
      my_score = 0
    else:
      #my_score = len(correct_map[val]) - len(incorrect_map[val])
      #F1 score
      correct_graphs_len = len(correct_map[val])
      incorrect_graphs_len = len(incorrect_map[val])
      accuracy =  correct_graphs_len/(correct_graphs_len + incorrect_graphs_len + 1)
      #recall =  correct_graphs_len/len(labeled_graphs)
      #my_score = (2 * (accuracy * recall))/(accuracy + recall)
      my_score = accuracy
      #my_score = len(correct_map[val]) - len(incorrect_map[val])
    
    if my_score > 0:
      val_set.add(val)
    if my_score > max_score:
      max_score = my_score
      max_score_val = val
  #'''
  print()
  print()
  print("Correct Map:")
  print(correct_map)
  print()
  print("Incorrect Map:")
  print(incorrect_map)
  print()
  print()
  print("Max Score : {}".format(max_score)) 
  print("Max Score val : {}".format(max_score_val)) 
  print()
  print()
  #'''
  return (max_score, max_score_val) 




#graph_projection
def eval_abs_graph(abs_graph, nodes, edges, A, X_node, X_edge):
  
  subgraphs = [] 
  abs_node_idx_to_concrete_nodes = {}
  abs_edge_idx_to_concrete_edges = {}

  for idx, abs_node in enumerate(abs_graph.absNodes):
    abs_node_idx_to_concrete_nodes[idx] = eval_abs_node(abs_node, nodes, X_node)
  
  sub_abs_graph = [[],[]]
  for _, node in enumerate(abs_node_idx_to_concrete_nodes[0]):
    subgraphs.append(([],[]))

  for idx, abs_edge in enumerate(abs_graph.absEdges):
    abs_edge_idx_to_concrete_edges[idx] = eval_abs_edge(abs_edge, edges, X_edge)
   
  candidate_abs_edges = copy.deepcopy(abs_graph.absEdges) 
  while(len(candidate_abs_edges) > 0):
    (abs_edge, sub_abs_graph_edge, sub_abs_graph, case) = choose_an_abs_edge_and_update_sub_abs_graph(sub_abs_graph, candidate_abs_edges)
    del candidate_abs_edges[0]
    subgraphs = update_subgraphs(abs_edge, sub_abs_graph_edge, subgraphs, sub_abs_graph, case, abs_edge_idx_to_concrete_edges, abs_node_idx_to_concrete_nodes, A)
    
  return subgraphs 



def choose_an_abs_edge_and_update_sub_abs_graph(sub_abs_graph, candidate_abs_edges):
  (_, p, q) = candidate_abs_edges[0]
  idx = len(sub_abs_graph[1]) 
  if (p in sub_abs_graph[0]) and (q in sub_abs_graph[0]):
    sub_abs_graph[1].append((p, q))
    case = 2
     
  elif (q in sub_abs_graph[0]): 
    sub_abs_graph[0].append(p)
    sub_abs_graph[1].append((p, q))
    case = 1 
 
  elif (p in sub_abs_graph[0]): 
    sub_abs_graph[0].append(q)
    sub_abs_graph[1].append((p, q))
    case = 0
    
  else:
    sub_abs_graph[0].append(p)    
    sub_abs_graph[0].append(q) 
    sub_abs_graph[1].append((p, q))
    case = 3
 
  return ((p, q, idx), (sub_abs_graph[0].index(p), sub_abs_graph[0].index(q)), sub_abs_graph, case)      


#check below implementation

def update_subgraphs(abs_edge, sub_graph_node_indices, subgraphs, sub_abs_graph, case, abs_edge_idx_to_concrete_edges, abs_node_idx_to_concrete_nodes, A):
  (p_abs, q_abs, abs_edge_idx) = abs_edge
  (p_sub_abs, q_sub_abs) = sub_graph_node_indices
  
  #ToDo
  my_set = set()
  new_subgraphs = []
  candidate_concrete_edges = abs_edge_idx_to_concrete_edges[abs_edge_idx]
  for _, [nodes, edges] in enumerate(subgraphs):
    for _, val in enumerate(candidate_concrete_edges):
      (p_con, q_con) = A[val]
      if case == 0 and p_con == nodes[p_sub_abs] and (q_con in abs_node_idx_to_concrete_nodes[q_abs]) and not (q_con in nodes):
        my_new_subgraph = copy.deepcopy([nodes,edges])
        my_new_subgraph[1].append((p_con, q_con))
        my_new_subgraph[0].append(q_con)
        
      elif case == 1 and q_con == nodes[q_sub_abs] and (p_con in abs_node_idx_to_concrete_nodes[p_abs]) and not (p_con in nodes):
        my_new_subgraph = copy.deepcopy([nodes,edges])
        my_new_subgraph[1].append((p_con, q_con))
        my_new_subgraph[0].append(p_con)

      elif case == 2 and p_con == nodes[p_sub_abs] and q_con == nodes[q_sub_abs]:
        my_new_subgraph = copy.deepcopy([nodes,edges])
        my_new_subgraph[1].append((p_con, q_con))
      
      elif case == 3 and (p_con in abs_node_idx_to_concrete_nodes[p_abs]) and (q_con in abs_node_idx_to_concrete_nodes[q_abs]): 
        my_new_subgraph = copy.deepcopy([nodes,edges])
        my_new_subgraph[1].append((p_con, q_con))
        my_new_subgraph[0].append(p_con)
        my_new_subgraph[0].append(q_con)

      else:
        continue
     
      #Why we need this?
      key = (json.dumps(my_new_subgraph))
      if not (key in my_set):
        my_set.add(key)
        new_subgraphs.append(my_new_subgraph)

  return new_subgraphs




def concrete_node_belong_abs_node(abs_node, node, X_node):
  for _, feat_idx in enumerate(abs_node):
    (bot, top) = abs_node[feat_idx]
    if X_node[node][feat_idx] < bot or top < X_node[node][feat_idx]:
      return False
  return True 


def concrete_edge_belong_abs_edge(abs_edge, edge, X_edge):
  #print(abs_edge)
  (itvs, p, q) = abs_edge 
  for _, feat_idx in enumerate(itvs):
    (bot, top) = itvs[feat_idx]
    if X_edge[edge][feat_idx] < bot or top < X_edge[edge][feat_idx]:
      return False
  return True 




#sub_graph_exist
#def eval_abs_graph_DFS(abs_graph, graph, A, X_node, X_edge, node_to_nodes):
def eval_abs_graph_DFS(abs_graph, graph, my_maps):
  #nodes = graph[0]
  edges = graph[1]
  #print(edges)
  abs_edge_first = abs_graph.absEdges[0]
  abs_node_fr = abs_edge_first[1]
  abs_node_to = abs_edge_first[2]
  candidate_edges = set()
  for _, edge in enumerate(edges):
    condition1 = concrete_edge_belong_abs_edge(abs_edge_first, edge, my_maps.X_edge)
    condition2 = concrete_node_belong_abs_node(abs_graph.absNodes[abs_node_fr], my_maps.A[edge][0], my_maps.X_node)
    condition3 = concrete_node_belong_abs_node(abs_graph.absNodes[abs_node_to], my_maps.A[edge][1], my_maps.X_node)
    if condition1 and condition2 and condition3:
      candidate_edges.add(edge) 

  #print(candidate_edges)

  for _, init_graph_edge in enumerate(candidate_edges):
    abs_node_idx_to_concrete_node = {}
    abs_edge_idx_to_concrete_edge = {}

    sub_abs_graph_edge = (abs_node_fr, abs_node_to)
    sub_abs_graph = [[abs_node_fr, abs_node_to],[sub_abs_graph_edge]]
    subgraph = [[],[]]
    subgraph[0].append(my_maps.A[init_graph_edge][0])
    subgraph[0].append(my_maps.A[init_graph_edge][1])
    subgraph[1].append(init_graph_edge)
    abs_node_idx_to_concrete_node[abs_node_fr] = my_maps.A[init_graph_edge][0]
    abs_node_idx_to_concrete_node[abs_node_to] = my_maps.A[init_graph_edge][1]
    abs_edge_idx_to_concrete_edge[0] = init_graph_edge
    
    if exist_subgraph_DFS(subgraph, sub_abs_graph, abs_graph, graph, 1, abs_node_idx_to_concrete_node, abs_edge_idx_to_concrete_edge, my_maps) == 0:
      return True
  return False


def exist_subgraph_DFS(subgraph, sub_abs_graph, abs_graph, graph, abs_edge_idx, abs_node_idx_to_concrete_node, abs_edge_idx_to_concrete_edge, my_maps):
  #print(subgraph) 
  #sys.exit()

  if len(abs_graph.absEdges) == abs_edge_idx:
    return 0 
  target_abs_edge = abs_graph.absEdges[abs_edge_idx]
  new_sub_abs_graph = copy.deepcopy(sub_abs_graph)
  (new_sub_abs_graph, case) = get_abs_edge_case_and_update_sub_abs_graph(new_sub_abs_graph, target_abs_edge)
  if case == 2:
    abs_node_fr = target_abs_edge[1]
    abs_node_to = target_abs_edge[2]
    fr_con = abs_node_idx_to_concrete_node[abs_node_fr]
    to_con = abs_node_idx_to_concrete_node[abs_node_to]
    if (fr_con, to_con) in my_maps.nodes_to_edge:
      con_edge = my_maps.nodes_to_edge[(fr_con, to_con)]
      #print(con_edge)
      if concrete_edge_belong_abs_edge(target_abs_edge, con_edge, my_maps.X_edge):
        new_abs_edge_idx_to_concrete_edge = copy.deepcopy(abs_edge_idx_to_concrete_edge)
        new_abs_edge_idx_to_concrete_edge[abs_edge_idx] = con_edge 
        new_subgraph = copy.deepcopy(subgraph)
        new_subgraph[1].append(con_edge)
        if exist_subgraph_DFS(new_subgraph, new_sub_abs_graph, abs_graph, graph, abs_edge_idx + 1, abs_node_idx_to_concrete_node, new_abs_edge_idx_to_concrete_edge, my_maps) == 0:
          return 0
  elif case == 1: 
    #new_fr
    abs_node_fr = target_abs_edge[1] # checkthis
    abs_node_to = target_abs_edge[2]
    to_con = abs_node_idx_to_concrete_node[abs_node_to]
    candidate_fr_nodes = my_maps.pred_node_to_nodes[to_con]
    for _, (con_edge, fr_con) in enumerate(candidate_fr_nodes):
      #condition1 = not (fr_node in subgraph[0])
      condition1 = not (fr_con in subgraph[0])
      condition2 = concrete_node_belong_abs_node(abs_graph.absNodes[abs_node_fr], my_maps.A[con_edge][0], my_maps.X_node)
      condition3 = concrete_edge_belong_abs_edge(target_abs_edge, con_edge, my_maps.X_edge)
      if condition1 and condition2 and condition3:
        new_abs_node_idx_to_concrete_node = copy.deepcopy(abs_node_idx_to_concrete_node)
        new_abs_edge_idx_to_concrete_edge = copy.deepcopy(abs_edge_idx_to_concrete_edge)
        new_abs_edge_idx_to_concrete_edge[abs_edge_idx] = con_edge
        new_abs_node_idx_to_concrete_node[target_abs_edge[1]] = fr_con

        new_subgraph = copy.deepcopy(subgraph)
        new_subgraph[0].append(fr_con)
        new_subgraph[1].append(con_edge)
        if exist_subgraph_DFS(new_subgraph, new_sub_abs_graph, abs_graph, graph, abs_edge_idx + 1, new_abs_node_idx_to_concrete_node, new_abs_edge_idx_to_concrete_edge, my_maps) == 0:
          return 0
  elif case == 0:
    #new_to
    abs_node_fr = target_abs_edge[1]
    abs_node_to = target_abs_edge[2]
    fr_con = abs_node_idx_to_concrete_node[abs_node_fr]
    #print(my_maps.succ_node_to_nodes)
    candidate_to_nodes = my_maps.succ_node_to_nodes[fr_con]
    for _, val  in enumerate(candidate_to_nodes):
      (con_edge, to_con) = val
      condition1 = not (to_con in subgraph[0])
      condition2 = concrete_node_belong_abs_node(abs_graph.absNodes[abs_node_to], my_maps.A[con_edge][1], my_maps.X_node)
      condition3 = concrete_edge_belong_abs_edge(target_abs_edge, con_edge, my_maps.X_edge)
      if condition1 and condition2 and condition3:
        new_abs_node_idx_to_concrete_node = copy.deepcopy(abs_node_idx_to_concrete_node)
        new_abs_node_idx_to_concrete_node[target_abs_edge[2]] = to_con
        new_abs_edge_idx_to_concrete_edge = copy.deepcopy(abs_edge_idx_to_concrete_edge)
        new_abs_edge_idx_to_concrete_edge[abs_edge_idx] = con_edge

        new_subgraph = copy.deepcopy(subgraph)
        new_subgraph[0].append(to_con)
        new_subgraph[1].append(con_edge)
        if exist_subgraph_DFS(new_subgraph, new_sub_abs_graph, abs_graph, graph, abs_edge_idx + 1, new_abs_node_idx_to_concrete_node, new_abs_edge_idx_to_concrete_edge, my_maps) == 0:
          return 0
  else:
    raise("Cannot be happened")
  return 1


def get_abs_edge_case_and_update_sub_abs_graph(sub_abs_graph, abs_edge):
  (_, p, q) = abs_edge
  sub_abs_nodes = sub_abs_graph[0]
  if (p in sub_abs_nodes) and (q in sub_abs_nodes):
    sub_abs_graph[1].append((p, q))
    case = 2
  elif (q in sub_abs_nodes): 
    sub_abs_graph[0].append(p)
    sub_abs_graph[1].append((p, q))
    case = 1 
  elif (p in sub_abs_nodes): 
    sub_abs_graph[0].append(q)
    sub_abs_graph[1].append((p, q))
    case = 0
  else:
    print("p : {}".format(p))
    print("q : {}".format(q))
    print("sub_abs_nodes : {}".format(sub_abs_nodes))
    print("sub_abs_nodes : {}".format(sub_abs_graph[1]))

    raise("Something wrong!")
  return (sub_abs_graph, case)




