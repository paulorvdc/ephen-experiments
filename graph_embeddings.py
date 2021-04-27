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

from ge import DeepWalk
from ge import Node2Vec
from ge import Struc2Vec
from ge import LINE

from ephin_utils import disturbed_hin
from ephin_utils import regularization
from ephin_utils import get_knn_data
from ephin_utils import restore_hin
from ephin_utils import embedding_graph
from ephin_utils import masked_softmax_cross_entropy
from ephin_utils import masked_accuracy
from ephin_utils import gcn

def run_model_cpu(G_disturbed, cutted_dict, algorithm, target, path, iteration, split):
    if algorithm == 'regularization':
        G_disturbed = regularization(G_disturbed)
        G_restored, restored = restore_hin(G_disturbed, cutted_dict)
        restored.to_csv('{0}restored_new/{1}_{2}_{3}_{4}.csv'.format(path, algorithm, str(target), iteration, split))
    
    elif algorithm == 'deep_walk':
        model_deep_walk = DeepWalk(G_disturbed,walk_length=10,num_walks=80,workers=1)
        model_deep_walk.train(window_size=5,iter=3,embed_size=512)# train model
        embeddings_deep_walk = model_deep_walk.get_embeddings()# get embedding vectors
        G_disturbed = embedding_graph(G_disturbed, embeddings_deep_walk)
        G_restored, restored = restore_hin(G_disturbed, cutted_dict)    
        restored.to_csv('{0}restored_new/{1}_{2}_{3}_{4}.csv'.format(path, algorithm, str(target), iteration, split))
    
    elif algorithm == 'node2vec':
        model_node2vec = Node2Vec(G_disturbed, walk_length = 10, num_walks = 80, p = 0.5, q = 1, workers = 1)
        model_node2vec.train(window_size=5,iter=3,embed_size=512)# train model
        embeddings_node2vec = model_node2vec.get_embeddings()# get embedding vectors
        G_disturbed = embedding_graph(G_disturbed, embeddings_node2vec)
        G_restored, restored = restore_hin(G_disturbed, cutted_dict)
        restored.to_csv('{0}restored_new/{1}_{2}_{3}_{4}.csv'.format(path, algorithm, str(target), iteration, split))

def run_model_gpu(G_disturbed, cutted_dict, algorithm, target, path, iteration, split):
    if algorithm == 'struct2vec':
        model_struct2vec = Struc2Vec(G_disturbed, 10, 80, workers=2, verbose=40) #init model
        model_struct2vec.train(window_size = 5, iter = 3, embed_size=512)# train model
        embeddings_struct2vec = model_struct2vec.get_embeddings()# get embedding vectors
        G_disturbed = embedding_graph(G_disturbed, embeddings_struct2vec)
        G_restored, restored = restore_hin(G_disturbed, cutted_dict)    
        restored.to_csv('{0}restored_new/{1}_{2}_{3}_{4}.csv'.format(path, algorithm, str(target), iteration, split))    
    
    elif algorithm == 'line':
        model_line = LINE(G_disturbed,embedding_size=512, order='second') #init model,order can be ['first','second','all']
        model_line.train(batch_size=8,epochs=20,verbose=0)# train model
        embeddings_line = model_line.get_embeddings()# get embedding vectors 
        G_disturbed = embedding_graph(G_disturbed, embeddings_line)
        G_restored, restored = restore_hin(G_disturbed, cutted_dict)    
        restored.to_csv('{0}restored_new/{1}_{2}_{3}_{4}.csv'.format(path, algorithm, str(target), iteration, split))
    
    elif algorithm == 'gcn':
        G_disturbed = gcn(G_disturbed)
        G_restored, restored_df = restore_hin(G_disturbed, cutted_dict)
        restored.to_csv('{0}restored_new/{1}_{2}_{3}_{4}.csv'.format(path, algorithm, str(target), iteration, split))
        
import multiprocessing

def process_cpu(start, end, targets, algorithms, edge_type, splits, path):                                           
    for i in range(start, end):
        with open(path + "graphs/graph_" + str(targets[i]) + ".gpickle", "rb") as fh:
            G = pickle.load(fh)
        for j in range(10):
            for split in splits:
                G_disturbed, cutted_dict = disturbed_hin(G, split=split, random_state=(1 + j), edge_type=edge_type)
                for algorithm in algorithms:
                    print('CPU: {0}, {1}, {2}, {3}'.format(algorithm, targets[i], j, split))
                    run_model_cpu(G_disturbed, cutted_dict, algorithm, targets[i], path, j, split)

def process_gpu(total, targets, algorithms, edge_type, splits, path):                                           
    for i in range(total):
        with open(path + "graphs/graph_" + str(targets[i]) + ".gpickle", "rb") as fh:
            G = pickle.load(fh)
        for j in range(10):
            for split in splits:
                G_disturbed, cutted_dict = disturbed_hin(G, split=split, random_state=(1 + j), edge_type=edge_type)
                for algorithm in algorithms:
                    print('GPU: {0}, {1}, {2}, {3}'.format(algorithm, targets[i], j, split))
                    run_model_gpu(G_disturbed, cutted_dict, algorithm, targets[i], path, j, split)

def split_processing(num_thread, targets, algorithms_cpu, algorithms_gpu, edge_type, splits, path):                                      
    split_size = len(targets) // num_thread                                       
    threads = []                                                                
    for i in range(num_thread):                                                 
        # determine the indices of the list this thread will handle             
        start = i * split_size                                                  
        # special case on the last chunk to account for uneven splits           
        end = len(targets) if i+1 == num_thread else (i+1) * split_size                 
        # create the thread                                                     
        threads.append(                                                         
            multiprocessing.Process(target=process_cpu, args=(start, end, targets, algorithms_cpu, edge_type, splits, path)))
        print('Starting thread: {0}'.format(i))         
        threads[-1].start() # start the thread we just created
    threads.append(                                                         
        multiprocessing.Process(target=process_gpu, args=(len(targets), targets, algorithms_gpu, edge_type, splits, path)))
    print('Starting thread: {0}'.format(i))                

    # wait for all threads to finish                                            
    for t in threads:
        t.join()

num_thread = 5
targets = [377904, 375777,  380274, 377199, 389118, 389293, 388224, 397968, 394909, 394491, 372939, 402610, 380994]
algorithms_cpu = ['regularization', 'deep_walk', 'node2vec']
algorithms_gpu = ['gcn', 'line', 'struct2vec']
edge_type = ['location', 'person', 'org']
splits = [0.05, 0.1, 0.15, 0.2]
path = "/home/paulocarmo/graph_experiments/"
split_processing(num_thread, targets, algorithms_cpu, algorithms_gpu, edge_type, splits, path)
