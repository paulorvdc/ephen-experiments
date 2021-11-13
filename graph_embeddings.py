import pandas as pd
import networkx as nx
import numpy as np
from scipy.spatial.distance import cosine
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import random
import pickle5 as pickle
import time
import scipy.sparse
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from ge import DeepWalk
from ge import Node2Vec
from ge import Struc2Vec
from ge import LINE

from ephen_utils import inner_connections_dateless
from ephen_utils import disturbed_hin
from ephen_utils import hide_nodes
from ephen_utils import find_nodes
from ephen_utils import regularization
from ephen_utils import restore_hin
from ephen_utils import embedding_graph
from ephen_utils import metapath2vec
from ephen_utils import gcn

def run_model(G_disturbed, cutted_dict, algorithm, network_name, path, iteration, split, restored_folder_name):
    if algorithm == 'regularization':
        G_disturbed = regularization(G_disturbed)
        restored = restore_hin(G_disturbed, cutted_dict)
        restored.to_csv('{}restored_{}/{}_{}_{}_{}.csv'.format(path, restored_folder_name, algorithm, network_name, iteration, split))
    
    elif algorithm == 'deep_walk':
        model_deep_walk = DeepWalk(G_disturbed,walk_length=10,num_walks=80,workers=1)
        model_deep_walk.train(window_size=5,iter=3,embed_size=512)# train model
        embeddings_deep_walk = model_deep_walk.get_embeddings()# get embedding vectors
        G_disturbed = embedding_graph(G_disturbed, embeddings_deep_walk)
        restored = restore_hin(G_disturbed, cutted_dict)    
        restored.to_csv('{}restored_{}/{}_{}_{}_{}.csv'.format(path, restored_folder_name, algorithm, network_name, iteration, split))
    
    elif algorithm == 'node2vec':
        model_node2vec = Node2Vec(G_disturbed, walk_length = 10, num_walks = 80, p = 0.5, q = 1, workers = 1)
        model_node2vec.train(window_size=5,iter=3,embed_size=512)# train model
        embeddings_node2vec = model_node2vec.get_embeddings()# get embedding vectors
        G_disturbed = embedding_graph(G_disturbed, embeddings_node2vec)
        restored = restore_hin(G_disturbed, cutted_dict)
        restored.to_csv('{}restored_{}/{}_{}_{}_{}.csv'.format(path, restored_folder_name, algorithm, network_name, iteration, split))
    
    elif algorithm == 'struc2vec':
        model_struc2vec = Struc2Vec(G_disturbed, 10, 80, workers=2, verbose=40) #init model
        model_struc2vec.train(window_size = 5, iter = 3, embed_size=512)# train model
        embeddings_struc2vec = model_struc2vec.get_embeddings()# get embedding vectors
        G_disturbed = embedding_graph(G_disturbed, embeddings_struc2vec)
        restored = restore_hin(G_disturbed, cutted_dict)    
        restored.to_csv('{}restored_{}/{}_{}_{}_{}.csv'.format(path, restored_folder_name, algorithm, network_name, iteration, split))

    elif algorithm == 'metapath2vec':
        
        # gdelt datasets
        user_metapaths = [
            ['event','date','event'],['event','theme','event'],['event','location','event'],
            ['event','person','event'],['event','org','event'],
        ]
        """
        # benchmark datasets
        user_metapaths = [
            ['event','when','event'],['event','what','event'],['event','where','event'],
            ['event','who','event'],['event','why','event'],['event','how','event'],
        ]"""
        embeddings_metapath2vec = metapath2vec(G_disturbed, user_metapaths=user_metapaths)
        G_disturbed = embedding_graph(G_disturbed, embeddings_metapath2vec)
        restored = restore_hin(G_disturbed, cutted_dict)
        restored.to_csv('{}restored_{}/{}_{}_{}_{}.csv'.format(path, restored_folder_name, algorithm, network_name, iteration, split))
    
    elif algorithm == 'line':
        model_line = LINE(G_disturbed,embedding_size=512, order='second') #init model,order can be ['first','second','all']
        model_line.train(batch_size=8,epochs=20,verbose=0)# train model
        embeddings_line = model_line.get_embeddings()# get embedding vectors 
        G_disturbed = embedding_graph(G_disturbed, embeddings_line)
        restored = restore_hin(G_disturbed, cutted_dict)    
        restored.to_csv('{}restored_{}/{}_{}_{}_{}.csv'.format(path, restored_folder_name, algorithm, network_name, iteration, split))
    
    elif algorithm == 'gcn':
        G_disturbed = gcn(G_disturbed, network_name, iteration, split)
        restored = restore_hin(G_disturbed, cutted_dict)
        restored.to_csv('{}restored_{}/{}_{}_{}_{}.csv'.format(path, restored_folder_name, algorithm, network_name, iteration, split))

def run_dynamic_test(G, target, path, iteration, split, restored_folder_name, edge_type):
    G_hidden, hidden_dict = hide_nodes(G, random_state=(1 + iteration))
    G_hidden, cutted_dict = disturbed_hin(G_hidden, split=split, random_state=(1 + iteration), edge_type=edge_type)
    G_hidden = regularization(G_hidden)
    G_hidden, hidden_dict = find_nodes(G_hidden, hidden_dict, percentual=0.5)
    G_hidden = regularization(G_hidden, iterations=5)
    G_hidden, hidden_dict = find_nodes(G_hidden, hidden_dict, percentual=1.0)
    G_hidden = regularization(G_hidden, iterations=5)
    restored = restore_hin(G_hidden, cutted_dict)
    restored.to_csv('{}restored_teste_dynamic_{}/{}_{}_{}.csv'.format(path, restored_folder_name, str(target), iteration, split))

def full_network_experiments(targets, splits, edge_type, interval, algorithms=None):
    for target in targets:
        with open(path + "graphs/graph_" + str(target) + ".gpickle", "rb") as fh:
            G = pickle.load(fh)
        for i in range(interval, 10):
            for split in splits:
                G_disturbed, cutted_dict = disturbed_hin(G, split=split, random_state=(1 + i), edge_type=edge_type)
                for algorithm in algorithms:
                    print('TEST: {}, {}, {}, {}, {}'.format(algorithm, target, i, split, sys.argv[1]))
                    start_time = time.time()
                    run_model(G_disturbed, cutted_dict, algorithm, target, path, i, split, sys.argv[1])
                    with open(f"{path}benchmark_graphs/execution_time.txt", 'a') as f:
                        f.write(f'Execution time for algorithm {algorithm}, in target {target} on scenario {sys.argv[1]}: {(time.time() - start_time)}.\n')

def dynamic_insert_network_experiments(targets, splits, edge_type, interval):
    for target in targets:
        with open(path + "graphs/graph_" + str(target) + ".gpickle", "rb") as fh:
            G = pickle.load(fh)
        for i in range(interval, 10):
            for split in splits:
                print('TEST: {}, {}, {}, {}'.format(target, i, split, sys.argv[1]))
                run_dynamic_test(G, target, path, i, split, sys.argv[1], edge_type)

import time
def full_network_experiments_bench(network_names, splits, edge_type, interval, algorithms=None):
    for network_name in network_names:
        with open(f"{path}benchmark_graphs/{network_name}.gpickle", "rb") as fh:
            G = pickle.load(fh)
        G = inner_connections_dateless(G)
        for i in range(interval, 10):
            for split in splits:
                G_disturbed, cutted_dict = disturbed_hin(G, split=split, random_state=(1 + i), edge_type=edge_type)
                for algorithm in algorithms:
                    print('TEST: {}, {}, {}, {}, {}'.format(algorithm, network_name, i, split, sys.argv[1]))
                    start_time = time.time()
                    run_model(G_disturbed, cutted_dict, algorithm, network_name, path, i, split, sys.argv[1])
                    with open(f"{path}benchmark_graphs/execution_time.txt", 'a') as f:
                        f.write(f'Execution time for algorithm {algorithm}, in network {network_name}: {(time.time() - start_time)}.\n')
#targets = [377904, 375777, 380274, 389293, 388224, 397968, 394909, 394491, 402610, 372939, 380994, 377199, 389118]
#edge_type = ['event_location', 'event_person', 'event_org', 'event_event']

import sys

targets = algorithms = edge_type = splits = interval = None 
path = "/media/pauloricardo/basement/projeto/"

if sys.argv[1] == 'location':
    targets = [375777, 388224]
    algorithms = ['regularization', 'deep_walk', 'node2vec', 'struc2vec', 'metapath2vec', 'line', 'gcn']
    edge_type = ['event_location']
    splits = [0.05, 0.1, 0.15, 0.2]
    interval = 0

elif sys.argv[1] == 'actor':
    targets = [375777, 388224]
    algorithms = ['regularization', 'deep_walk', 'node2vec', 'struc2vec', 'metapath2vec', 'line', 'gcn']
    edge_type = ['event_person', 'event_org']
    splits = [0.05, 0.1, 0.15, 0.2]
    interval = 0

elif sys.argv[1] == 'event':
    targets = [375777, 388224]
    algorithms = ['regularization', 'deep_walk', 'node2vec', 'struc2vec', 'metapath2vec', 'line', 'gcn']
    edge_type = ['event_event']
    splits = [0.05, 0.1, 0.15, 0.2]
    interval = 0

elif sys.argv[1] == 'where':
    network_names = ['40er_5w1h_graph_hin', 'bbc_5w1h_graph_hin', 'gold_standard_5w1h_graph_hin', 'google_news_5w1h_graph_hin', 'news_cluster_5w1h_graph_hin']
    algorithms = ['regularization', 'deep_walk', 'node2vec', 'struc2vec', 'metapath2vec', 'line', 'gcn']
    edge_type = ['event_where']
    splits = [0.2]
    interval = 0

elif sys.argv[1] == 'who':
    network_names = ['40er_5w1h_graph_hin', 'bbc_5w1h_graph_hin', 'gold_standard_5w1h_graph_hin', 'google_news_5w1h_graph_hin', 'news_cluster_5w1h_graph_hin']
    algorithms = ['regularization', 'deep_walk', 'node2vec', 'struc2vec', 'metapath2vec', 'line', 'gcn']
    edge_type = ['event_who']
    splits = [0.2]
    interval = 0

experiment_selection = {
    'full': full_network_experiments,
    'dynamic': dynamic_insert_network_experiments,
    'full_bench': full_network_experiments_bench
}

experiment_selection['full'](targets, splits, edge_type, interval, algorithms=algorithms)
#experiment_selection['dynamic'](targets, splits, edge_type, interval)

#experiment_selection['full_bench'](network_names, splits, edge_type, interval, algorithms=algorithms)