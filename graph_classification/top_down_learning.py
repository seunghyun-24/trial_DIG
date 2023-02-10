from fast_language import *
#from language import *
import copy
import json
import sys

def learn_abs_graphs_top_down(parameter):
  print("Left graphs : {}".format(parameter.left_graphs))
  print("Left graphs len : {}".format(len(parameter.left_graphs)))
  print()
  learned_parameters = set()
  while(len(parameter.left_graphs) > 0):
    abs_graph = AbstractGraph()
    abs_graph.absNodes = [{}, {}]
    abs_graph.absEdges = [({}, 0, 1)]

    (learned_abs_graph, learned_val) = specify(abs_graph, parameter)

    learned_parameters.add((learned_abs_graph, learned_val))
    
    chosen_graphs = eval_abs_graph_on_graphs(learned_abs_graph, learned_val, parameter.graphs, parameter.A, parameter.X_node, parameter.X_edge) 
    parameter.left_graphs = parameter.left_graphs - chosen_graphs
    
    print("Left graphs : {}".format(parameter.left_graphs))
    print("Left graphs len : {}".format(len(parameter.left_graphs)))

  return learned_parameters


def specify(abs_graph, parameter, my_maps):
  best_score = eval_abs_graph_on_graphs_GC(abs_graph, parameter.graphs, parameter.labeled_graphs, parameter.train_graphs, my_maps)
  best_abs_graph = abs_graph
  
  my_abs_graphs_val_set = [] 

  new_candidate_abs_graphs = set([abs_graph])

  for depth in range(parameter.chosen_depth):
    print("Depth: {}".format(depth))
    candidate_abs_graphs = new_candidate_abs_graphs
    new_candidate_abs_graphs = set()
    for _, abs_graph in enumerate(candidate_abs_graphs):
      #print("Depth: {}".format(depth))
      #print()
      #print("Absgraphs Enumerats from")
      #print("AbsNodes : {}".format(abs_graph.absNodes))
      #print("AbsEdges : {}".format(abs_graph.absEdges))
      #print()
      #case_add_new_node
      #print()
      #print("===============Case add node===============")
      for exist_node1 in range(len(abs_graph.absNodes)):
        new_node = {}
        new_edge_idx = len(abs_graph.absNodes)
        new_edge = ({}, exist_node1, new_edge_idx)
        new_abs_graph = copy.deepcopy(abs_graph)
        new_abs_graph.absNodes.append(new_node)
        new_abs_graph.absEdges.append(new_edge)
        new_score = eval_abs_graph_on_graphs_GC(new_abs_graph, parameter.graphs, parameter.labeled_graphs, parameter.train_graphs, my_maps)
        if new_score >= best_score:
          best_abs_graph = new_abs_graph
          best_score = new_score
          print()
          print("NewAbsGraph")
          print(new_abs_graph.absNodes)
          print(new_abs_graph.absEdges)
          print()
          print("New Score : {}".format(new_score))
       
        if not ((len(new_abs_graph.absNodes), len(new_abs_graph.absEdges)) in parameter.my_cache):
          if (depth < parameter.chosen_depth - 1) and (new_score > 0.1):
            new_candidate_abs_graphs.add(new_abs_graph)
        else:
          parameter.my_cache.add((len(new_abs_graph.absNodes), len(new_abs_graph.absEdges), new_val))


      for abs_node_idx in range(len(abs_graph.absNodes)):

        flag = True
        for i in range(abs_node_idx):
          if len(abs_graph.absNodes[i]) == 0:
            flag = False
            break
        if flag == False:
          continue



        for abs_node_feat_idx in range(len(parameter.X_node[0])):
          abs_node = abs_graph.absNodes[abs_node_idx]
          if len(abs_node) > 0:
            continue
          if abs_node_feat_idx in abs_node:
            continue
          else:
            #case upper
            new_itv = (0.5, 1.0)
            new_abs_graph = copy.deepcopy(abs_graph)
            new_abs_graph.absNodes[abs_node_idx][abs_node_feat_idx] = new_itv
            #(new_score, new_val) = score(new_abs_graph, parameter)
            new_score = eval_abs_graph_on_graphs_GC(new_abs_graph, parameter.graphs, parameter.labeled_graphs, parameter.train_graphs, my_maps)

            if new_score >= best_score:
              best_abs_graph = new_abs_graph
              best_score = new_score
              print()
              print("NewAbsGraph")
              print(new_abs_graph.absNodes)
              print(new_abs_graph.absEdges)
              print()
              print("New Score : {}".format(new_score))
           
            if not ((len(new_abs_graph.absNodes), len(new_abs_graph.absEdges)) in parameter.my_cache):
              if (depth < parameter.chosen_depth - 1) and (new_score > 0.1):
                new_candidate_abs_graphs.add(new_abs_graph)
            else:
              parameter.my_cache.add((len(new_abs_graph.absNodes), len(new_abs_graph.absEdges), new_val))

      for abs_edge_idx in range(len(abs_graph.absEdges)):
        for abs_edge_feat_idx in range(len(parameter.X_edge[0])):
          abs_edge = abs_graph.absEdges[abs_edge_idx]
          if abs_edge_feat_idx in abs_edge:
            continue
          if len(abs_edge) > 0:
            continue
          else:
            #case upper
            new_itv = (0.5, 1.0)
            new_abs_graph = copy.deepcopy(abs_graph)
            new_abs_graph.absEdges[abs_edge_idx][0][abs_edge_feat_idx] = new_itv
 
            new_score = eval_abs_graph_on_graphs_GC(new_abs_graph, parameter.graphs, parameter.labeled_graphs, parameter.train_graphs, my_maps)

            if new_score >= best_score:
              best_abs_graph = new_abs_graph
              best_score = new_score
              print()
              print("NewAbsGraph")
              print(new_abs_graph.absNodes)
              print(new_abs_graph.absEdges)
              print()
              print("New Score : {}".format(new_score))
 
            if not ((len(new_abs_graph.absNodes), len(new_abs_graph.absEdges)) in parameter.my_cache):
              if (depth < parameter.chosen_depth - 1) and (new_score > 0.1):
                new_candidate_abs_graphs.add(new_abs_graph)
            else:
              parameter.my_cache.add((len(new_abs_graph.absNodes), len(new_abs_graph.absEdges), new_val))
  print()
  print("Learned best absGraph")
  print(best_abs_graph.absNodes)
  print(best_abs_graph.absEdges)
  print("Learned Score : {}".format(best_score))

  return best_abs_graph



