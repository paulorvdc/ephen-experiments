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

from ephin_utils import disturbed_hin
from ephin_utils import regularization
from ephin_utils import get_knn_data
from ephin_utils import restore_hin
from ephin_utils import embedding_graph
from ephin_utils import masked_softmax_cross_entropy
from ephin_utils import masked_accuracy
from ephin_utils import metapath2vec
from ephin_utils import gcn

def run_model(G_disturbed, cutted_dict, algorithm, target, path, iteration, split):
    if algorithm == 'regularization':
        G_disturbed = regularization(G_disturbed)
        G_restored, restored = restore_hin(G_disturbed, cutted_dict)
        restored.to_csv('{0}restored_28.07/{1}_{2}_{3}_{4}.csv'.format(path, algorithm, str(target), iteration, split))
    
    elif algorithm == 'deep_walk':
        model_deep_walk = DeepWalk(G_disturbed,walk_length=10,num_walks=80,workers=1)
        model_deep_walk.train(window_size=5,iter=3,embed_size=512)# train model
        embeddings_deep_walk = model_deep_walk.get_embeddings()# get embedding vectors
        G_disturbed = embedding_graph(G_disturbed, embeddings_deep_walk)
        G_restored, restored = restore_hin(G_disturbed, cutted_dict)    
        restored.to_csv('{0}restored_28.07/{1}_{2}_{3}_{4}.csv'.format(path, algorithm, str(target), iteration, split))
    
    elif algorithm == 'node2vec':
        model_node2vec = Node2Vec(G_disturbed, walk_length = 10, num_walks = 80, p = 0.5, q = 1, workers = 1)
        model_node2vec.train(window_size=5,iter=3,embed_size=512)# train model
        embeddings_node2vec = model_node2vec.get_embeddings()# get embedding vectors
        G_disturbed = embedding_graph(G_disturbed, embeddings_node2vec)
        G_restored, restored = restore_hin(G_disturbed, cutted_dict)
        restored.to_csv('{0}restored_28.07/{1}_{2}_{3}_{4}.csv'.format(path, algorithm, str(target), iteration, split))
    
    elif algorithm == 'struc2vec':
        model_struc2vec = Struc2Vec(G_disturbed, 10, 80, workers=2, verbose=40) #init model
        model_struc2vec.train(window_size = 5, iter = 3, embed_size=512)# train model
        embeddings_struc2vec = model_struc2vec.get_embeddings()# get embedding vectors
        G_disturbed = embedding_graph(G_disturbed, embeddings_struc2vec)
        G_restored, restored = restore_hin(G_disturbed, cutted_dict)    
        restored.to_csv('{0}restored_28.07/{1}_{2}_{3}_{4}.csv'.format(path, algorithm, str(target), iteration, split))    

    elif algorithm == 'metapath2vec':
        embeddings_metapath2vec = metapath2vec(G_disturbed)
        G_disturbed = embedding_graph(G_disturbed, embeddings_metapath2vec)
        G_restored, restored = restore_hin(G_disturbed, cutted_dict)
        restored.to_csv('{0}restored_28.07/{1}_{2}_{3}_{4}.csv'.format(path, algorithm, str(target), iteration, split))
    
    elif algorithm == 'line':
        model_line = LINE(G_disturbed,embedding_size=512, order='second') #init model,order can be ['first','second','all']
        model_line.train(batch_size=8,epochs=20,verbose=0)# train model
        embeddings_line = model_line.get_embeddings()# get embedding vectors 
        G_disturbed = embedding_graph(G_disturbed, embeddings_line)
        G_restored, restored = restore_hin(G_disturbed, cutted_dict)    
        restored.to_csv('{0}restored_28.07/{1}_{2}_{3}_{4}.csv'.format(path, algorithm, str(target), iteration, split))
    
    elif algorithm == 'gcn':
        G_disturbed = gcn(G_disturbed, target, i, split)
        G_restored, restored = restore_hin(G_disturbed, cutted_dict)
        restored.to_csv('{0}restored_28.07/{1}_{2}_{3}_{4}.csv'.format(path, algorithm, str(target), iteration, split))

#targets = [377904, 375777, 380274, 389293, 388224, 397968, 394909, 394491, 402610, 372939, 380994, 377199, 389118]
targets = [389293, 388224, 397968, 394909, 394491, 402610]
algorithms = ['regularization', 'deep_walk', 'node2vec', 'struc2vec', 'metapath2vec', 'line', 'gcn']
#edge_type = ['event_location', 'event_person', 'event_org', 'event_event']
edge_type = ['event_location']
splits = [0.05, 0.1, 0.15, 0.2]
path = "/media/pauloricardo/basement/projeto/"

for target in targets:
    with open(path + "graphs/graph_" + str(target) + ".gpickle", "rb") as fh:
        G = pickle.load(fh)
    for i in range(10):
        for split in splits:
            G_disturbed, cutted_dict = disturbed_hin(G, split=split, random_state=(1 + i), edge_type=edge_type)
            for algorithm in algorithms:
                print('TEST: {0}, {1}, {2}, {3}'.format(algorithm, target, i, split))
                run_model(G_disturbed, cutted_dict, algorithm, target, path, i, split)
