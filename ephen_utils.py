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

def difference(start, end, interval):
    x = end - start
    r = {
            'week': int(x / np.timedelta64(1, 'W')),
            'fortnight': int(x / np.timedelta64(2, 'W')),
            'month': int(x / np.timedelta64(1, 'M'))
        }
    return r[interval]

def make_hin(X, df, id_feature='GKGRECORDID', date_feature='date_str', theme_feature='Themes', location_feature='Locations', person_feature='Persons', org_feature='Organizations'):
    G = nx.Graph()
    for index,row in df.iterrows():
        node_id = row[id_feature]
        # date conversion
        date_value = pd.to_datetime(row[date_feature], format='%Y-%m-%d')
        node_date = str(date_value.week) + '-' + str(date_value.year)
        #node_themes_array = ''
        node_locations_array = ''
        node_people_array = ''
        node_organizations_array = ''
        try:
            node_themes_array = row[theme_feature].split(';')
        except:
            node_themes_array = []
        try:
            node_locations_array = row[location_feature].split(';')
        except:
            node_locations_array = []
        try:
            node_people_array = row[person_feature].split(';')
        except:
            node_people_array = []
        try:
            node_organizations_array = row[org_feature].split(';')
        except:
            node_organizations_array = []
        
        # event <-> date
        G.add_edge(node_id,node_date,edge_type='event_date', edge_value=date_value)
        G.nodes[node_id]['node_type'] = 'event'
        G.nodes[node_date]['node_type'] = 'date'
        # event <-> theme
        for theme in node_themes_array:
            if len(theme) > 0:
                G.add_edge(node_id,theme,edge_type='event_theme')
                G.nodes[theme]['node_type'] = 'theme'
        # event <-> locations
        for location in node_locations_array:
            if len(location) > 0:
                G.add_edge(node_id,location,edge_type='event_location')
                G.nodes[location]['node_type'] = 'location'
        # event <-> persons
        for person in node_people_array:
            if len(person) > 0:
                G.add_edge(node_id,person,edge_type='event_person')
                G.nodes[person]['node_type'] = 'person'
        # event <-> organization
        for org in node_organizations_array:
            if len(org) > 0:
                G.add_edge(node_id,org,edge_type='event_org')
                G.nodes[org]['node_type'] = 'org'
        # embedding
        G.nodes[node_id]['embedding'] = X[index]
    return G

def inner_connections(G, interval='week', embedding_feature='embedding', type_feature='edge_type', desired_type_feature='event_date', value_feature='edge_value', return_type_feature='event_event'):
    edges_to_add = []
    for node1, neighbor1 in G.edges:
        if embedding_feature in G.nodes[node1]:
            if G[node1][neighbor1][type_feature] == desired_type_feature:
                for node2, neighbor2 in G.edges:
                    if embedding_feature in G.nodes[node2]:
                        if G[node2][neighbor2][type_feature] == desired_type_feature:
                            temp_cosine = cosine(G.nodes[node1][embedding_feature], G.nodes[node2][embedding_feature])
                            if temp_cosine <= 0.5 and temp_cosine != 0.0:
                                if abs(difference(G[node1][neighbor1][value_feature], G[node2][neighbor2][value_feature], interval)) <= 3:
                                    edges_to_add.append((node1,node2))
    for new_edge in edges_to_add:
        G.add_edge(new_edge[0],new_edge[1],edge_type=return_type_feature)
    return G

def is_equal(x, true_feature='true', restored_feature='restored'):
    if x[true_feature][0] == x[restored_feature][0] and x[true_feature][1] == x[restored_feature][1]:
        return 1
    elif x[true_feature][0] == x[restored_feature][1] and x[true_feature][1] == x[restored_feature][0]:
        return 1
    return 0

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
def get_metric(metric, pred):
    if metric == 'acc':
        return accuracy_score([1] * pred.shape[0], list(pred))
    elif metric == 'precision':
        return precision_score([1] * pred.shape[0], list(pred))
    elif metric == 'recall':
        return recall_score([1] * pred.shape[0], list(pred))
    elif metric == 'f1':
        return f1_score([1] * pred.shape[0], list(pred))

def disturbed_hin(G, split=0.1, random_state=None, edge_type=['event_date', 'event_event', 'event_location', 'event_person', 'event_org', 'event_theme'], type_feature='edge_type'):
    """
    G: hin;
    split: percentage to be cut from the hin;
    random_state: ;
    edge_type: listlike object of types of edges to be cut;
    type_feature: feature name of edge_type on your hin.
    """
    def keep_left(x, G):
        edge_split = x['type'].split('_')
        if G.nodes[x['node']]['node_type'] != edge_split[0]:
            x['node'], x['neighbor'] = x['neighbor'], x['node']
        return x
    # prepare data for type counting
    edges = list(G.edges)
    edge_types = []
    for node, neighbor in edges:
        edge_types.append(G[node][neighbor][type_feature])
    
    edges = pd.DataFrame(edges)
    edges = edges.rename(columns={0: 'node', 1: 'neighbor'})
    edges['type'] = edge_types
    edges = edges.apply(keep_left, G=G, axis=1)
    edges_group = edges.groupby(by=['type'], as_index=False).count().reset_index()

    # preparar arestas para eliminar
    edges = edges.sample(frac=1, random_state=random_state).reset_index(drop=True)
    edges_group = edges_group.rename(columns={'node': 'count', 'neighbor': 'to_cut_count'})
    edges_group['to_cut_count'] = edges_group['to_cut_count'].apply(lambda x:round(x * split))
    to_cut = {}
    for index, row in edges_group.iterrows():
        if row['type'] in edge_type:
            to_cut[row['type']] = edges[edges['type'] == row['type']].reset_index(drop=True).loc[0:row['to_cut_count']-1]
                    
    # eliminar arestas, salvar grafo e arestas retiradas para avaliação
    G_disturbed = deepcopy(G)
    for key, tc_df in to_cut.items():
        for index, row in tc_df.iterrows():
            G_disturbed.remove_edge(row['node'],row['neighbor'])
    return G_disturbed, to_cut

def regularization(G, dim=512, embedding_feature: str = 'embedding', iterations=15, mi=0.85):
    nodes = []
    # inicializando vetor f para todos os nodes
    for node in G.nodes():
        G.nodes[node]['f'] = np.array([0.0]*dim)
        if embedding_feature in G.nodes[node]:
            G.nodes[node]['f'] = G.nodes[node][embedding_feature]*1.0
        nodes.append(node)
    pbar = tqdm(range(0, iterations))
    for iteration in pbar:
        random.shuffle(nodes)
        energy = 0.0
        # percorrendo cada node
        for node in nodes:
            f_new = np.array([0.0]*dim)
            f_old = np.array(G.nodes[node]['f'])*1.0
            sum_w = 0.0
            # percorrendo vizinhos do onde
            for neighbor in G.neighbors(node):
                w = 1.0
                if 'weight' in G[node][neighbor]:
                    w = G[node][neighbor]['weight']
                w /= np.sqrt(G.degree[neighbor])
                f_new = f_new + w*G.nodes[neighbor]['f']
                sum_w = sum_w + w
            if sum_w == 0.0: sum_w = 1.0
            f_new /= sum_w
            G.nodes[node]['f'] = f_new*1.0
            if embedding_feature in G.nodes[node]:
                G.nodes[node]['f'] = G.nodes[node][embedding_feature] * \
                    mi + G.nodes[node]['f']*(1.0-mi)
            energy = energy + np.linalg.norm(f_new-f_old)
        iteration = iteration + 1
        message = 'Iteration '+str(iteration)+' | Energy = '+str(energy)
        pbar.set_description(message)
    return G

def get_knn_data(G, node, embedding_feature: str = 'f'):
    knn_data, knn_nodes = [], []
    for node in nx.non_neighbors(G, node):
        if embedding_feature in G.nodes[node]:
            knn_data.append(G.nodes[node][embedding_feature])
            knn_nodes.append(node)
    return pd.DataFrame(knn_data), pd.DataFrame(knn_nodes)

def run_knn(k, G_restored, row, knn_data, knn_nodes, node_feature='node', embedding_feature='f'):
    knn = NearestNeighbors(n_neighbors=k, metric='cosine')
    knn.fit(knn_data)
    indice = knn.kneighbors(G_restored.nodes[row[node_feature]][embedding_feature].reshape(-1, 512), return_distance=False)
    return [knn_nodes[0].iloc[indice[0][i]] for i in range(k)]

from annoy import AnnoyIndex
def run_annoy(k, G_restored, row, knn_data, knn_nodes, node_feature='node', embedding_feature='f', dim=512):
    knn = AnnoyIndex(dim, 'angular')
    for knn_index, knn_row in knn_data.iterrows():
        knn.add_item(knn_index, knn_row)
    knn.build(k)
    indice = knn.get_nns_by_vector(G_restored.nodes[row[node_feature]][embedding_feature], k, include_distances=False)
    return [knn_nodes[0].loc[indice[i]] for i in range(k)]

import multiprocessing
def restore_hin(G, cutted_dict, nn_method='knn', n_jobs=-1, k=5, node_feature='node', neighbor_feature='neighbor', node_type_feature='node_type', embedding_feature='f'):
    def process(start, end, G, nearest_neighbor_selector, key, value, return_dict, thread_id):
        value_thread = value.loc[start:(end-1)]
        restored_dict_thread = {'true': [], 'restored': [], 'edge_type': []}
        for index, row in tqdm(value_thread.iterrows(), total=value_thread.shape[0]):
            edge_to_add = key.split('_')
            edge_to_add[0] = row[node_feature]
            edge_to_add = [row[node_feature] if e == G.nodes[row[node_feature]][node_type_feature] and row[node_feature] != edge_to_add[0] else e for e in edge_to_add]
            knn_data, knn_nodes = get_knn_data(G, row[node_feature])
            knn_nodes['type'] = knn_nodes[0].apply(lambda x: G.nodes[x][node_type_feature])
            knn_data = knn_data[knn_nodes['type'].isin(edge_to_add)]
            knn_nodes = knn_nodes[knn_nodes['type'].isin(edge_to_add)]
            edge_to_add[1] = nearest_neighbor_selector[nn_method](k, G, row, knn_data, knn_nodes)
            restored_dict_thread['true'].append([row[node_feature], row[neighbor_feature]])
            restored_dict_thread['restored'].append(edge_to_add)
            restored_dict_thread['edge_type'].append(key)
        for key in restored_dict_thread.keys():
            _key = key + str(thread_id)
            return_dict[_key] = (restored_dict_thread[key])
    
    def split_processing(n_jobs, G, nearest_neighbor_selector, key, value, return_dict):
        split_size = round(len(value) / n_jobs)
        threads = []                                                                
        for i in range(n_jobs):                                                 
            # determine the indices of the list this thread will handle             
            start = i * split_size                                                  
            # special case on the last chunk to account for uneven splits           
            end = len(value) if i+1 == n_jobs else (i+1) * split_size                
            # create the thread
            threads.append(                                                         
                multiprocessing.Process(target=process, args=(start, end, G, nearest_neighbor_selector, key, value, return_dict, i)))
            threads[-1].start() # start the thread we just created                  

        # wait for all threads to finish                                            
        for t in threads:
            t.join()

    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    restored_dict = {'true': [], 'restored': [], 'edge_type': []}
    return_dict = multiprocessing.Manager().dict()
    nearest_neighbor_selector = {
        'knn': run_knn,
        'annoy': run_annoy
    }

    for key, value in cutted_dict.items():
        split_processing(n_jobs, G, nearest_neighbor_selector, key, value, return_dict)
        return_dict = dict(return_dict)
        for thread_key in restored_dict.keys():
            for job in range(n_jobs):
                for res in return_dict[thread_key + str(job)]:
                    restored_dict[thread_key].append(res)
    return pd.DataFrame(restored_dict)

def restore_edges(G, restored):
    G_restored = deepcopy(G)
    for idx, restored in enumerate(restored.restored.to_list()):
        G_restored.add_edge(restored[0],restored[1][0], edge_type=restored.edge_type.to_list()[idx])
    return G_restored

# put embeddings on graph
def embedding_graph(G, embeddings, embedding_feature='f'):
    for key, value in embeddings.items():
        G.nodes[key][embedding_feature] = value
    return G

def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)

import layers.graph as lg
import utils.sparse as us

def gcn(G, target, i, split, label_feature='node_type', label_number='type_code', embedding_feature='f'):
    node_list = []
    for node in G.nodes():
      node_list.append(node)
    
    label_codes = {}
    for node in node_list:
      if label_feature in G.nodes[node]:
          label = G.nodes[node][label_feature]
          if label not in label_codes: 
              label_codes[label] = len(label_codes)
              G.nodes[node][label_number] = label_codes[label]
          else:
              G.nodes[node][label_number] = -1
      else: 
          G.nodes[node][label_number] = -1
      
    adj = nx.adj_matrix(G,nodelist=node_list)
  
    # Get important parameters of adjacency matrix
    n_nodes = adj.shape[0]
  
    # Some preprocessing
    adj_tilde = adj + np.identity(n=adj.shape[0])
    d_tilde_diag = np.squeeze(np.sum(np.array(adj_tilde), axis=1))
    d_tilde_inv_sqrt_diag = np.power(d_tilde_diag, -1/2)
    d_tilde_inv_sqrt = np.diag(d_tilde_inv_sqrt_diag)
    adj_norm = np.dot(np.dot(d_tilde_inv_sqrt, adj_tilde), d_tilde_inv_sqrt)
    adj_norm_tuple = us.sparse_to_tuple(scipy.sparse.coo_matrix(adj_norm))
  
    # Features are just the identity matrix
    feat_x = np.identity(n=adj.shape[0])
    feat_x_tuple = us.sparse_to_tuple(scipy.sparse.coo_matrix(feat_x))
  
    # Preparing train data
    memberships = [m for m in nx.get_node_attributes(G, label_number).values()]
    nb_classes = len(set(memberships))
    targets = np.array([memberships], dtype=np.int32).reshape(-1)
    one_hot_targets = np.eye(nb_classes)[targets]
  
    labels_to_keep = [i for i in range(len(node_list))]
  
    y_train = np.zeros(shape=one_hot_targets.shape,
                      dtype=np.float32)
  
    train_mask = np.zeros(shape=(n_nodes,), dtype=np.bool)
  
    for l in labels_to_keep:
        y_train[l, :] = one_hot_targets[l, :]
        train_mask[l] = True
  
    # TensorFlow placeholders
    ph = {
        'adj_norm': tf.sparse_placeholder(tf.float32, name="adj_mat"),
        'x': tf.sparse_placeholder(tf.float32, name="x"),
        'labels': tf.placeholder(tf.float32, shape=(n_nodes, nb_classes)),
        'mask': tf.placeholder(tf.int32)}
  
    l_sizes = [1024, 1024, 512, nb_classes]
    
    name_text = str(target) + '_' + str(i) + '_' + str(split)
    
    o_fc1 = lg.GraphConvLayer(
        input_dim=feat_x.shape[-1],
        output_dim=l_sizes[0],
        name='fc1_'+name_text,
        activation=tf.nn.tanh)(adj_norm=ph['adj_norm'], x=ph['x'], sparse=True)
  
    o_fc2 = lg.GraphConvLayer(
        input_dim=l_sizes[0],
        output_dim=l_sizes[1],
        name='fc2_'+name_text,
        activation=tf.nn.tanh)(adj_norm=ph['adj_norm'], x=o_fc1)
  
    o_fc3 = lg.GraphConvLayer(
        input_dim=l_sizes[1],
        output_dim=l_sizes[2],
        name='fc3_'+name_text,
        activation=tf.nn.tanh)(adj_norm=ph['adj_norm'], x=o_fc2)
  
    o_fc4 = lg.GraphConvLayer(
        input_dim=l_sizes[2],
        output_dim=l_sizes[3],
        name='fc4_'+name_text,
        activation=tf.identity)(adj_norm=ph['adj_norm'], x=o_fc3)
  
  
    with tf.name_scope('optimizer'):
        loss = masked_softmax_cross_entropy(preds=o_fc4, labels=ph['labels'], mask=ph['mask'])
        accuracy = masked_accuracy(preds=o_fc4, labels=ph['labels'], mask=ph['mask'])
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
        opt_op = optimizer.minimize(loss)
  
    feed_dict_train = {ph['adj_norm']: adj_norm_tuple,
                      ph['x']: feat_x_tuple,
                      ph['labels']: y_train,
                      ph['mask']: train_mask}
  
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
  
    epochs = 20
    save_every = 50
  
    t = time.time()
    embedding_out = []
    # Train model
    for epoch in range(epochs):
        _, train_loss, train_acc = sess.run(
            (opt_op, loss, accuracy), feed_dict=feed_dict_train)
  
        if True:
            val_loss, val_acc = sess.run((loss, accuracy), feed_dict=feed_dict_train)
  
            # # Print results
            # #print("Epoch:", '%04d' % (epoch + 1),
            #       "train_loss=", "{:.5f}".format(train_loss),
            #       "time=", "{:.5f}".format(time.time() - t))
  
            feed_dict_output = {ph['adj_norm']: adj_norm_tuple,
                                ph['x']: feat_x_tuple}
  
            embeddings = sess.run(o_fc3, feed_dict=feed_dict_output)
            if epoch + 1 == epochs:
                embedding_out = embeddings
    for idx, node in enumerate(G.nodes()):
        G.nodes[node][embedding_feature] = embedding_out[idx]

    return G

from bs4 import BeautifulSoup
def decode_html_text(x):
    x = BeautifulSoup(x, 'html.parser')
    return x.get_text()

from gensim.models import Word2Vec
from stellargraph.data import UniformRandomMetaPathWalk
from stellargraph import StellarGraph

def metapath2vec(graph, dimensions = 512, num_walks = 1, walk_length = 100, context_window_size = 10, 
                           num_iter = 1, workers = 1, node_type='node_type', edge_type='edge_type',
                           user_metapaths=[
                                   ['event','date','event'],['event','what','event'],['event','where','event'],
                                   ['event','who','event'],['event','why','event'],['event','how','event'],
                                   ['event','date','event','trend','event'],['event','what','event','trend','event'],
                                   ['event','where','event','trend','event'],['event','who','event','trend','event'],
                                   ['event','why','event','trend','event'],['event','how','event','trend','event'],
                               ]
                           ):
    s_graph = StellarGraph.from_networkx(graph, node_type_attr=node_type, edge_type_attr=edge_type)
    rw = UniformRandomMetaPathWalk(s_graph)
    walks = rw.run(
        s_graph.nodes(), n=num_walks, length=walk_length, metapaths=user_metapaths
    )
    
    print(f"Number of random walks: {len(walks)}")

    model = Word2Vec(
        walks,
        size=dimensions,
        window=context_window_size,
        min_count=0,
        sg=1,
        workers=workers,
        iter=num_iter,
    )
    
    def get_embeddings(model, graph):
        if model is None:
            print("model not train")
            return {}

        _embeddings = {}
        for word in graph.nodes():
            try:
                _embeddings[word] = model.wv[word]
            except:
                _embeddings[word] = np.zeros(dimensions)

        return _embeddings
    return get_embeddings(model, graph)

def _ap(true, list_pred, at):
    ranking, aps = [], []
    for i in range(at):
        ranking.append(i+1)
    for index_t, t in enumerate(true):
        hit = False
        # get the list of predicteds that's on the secon argument
        for index_lp, lp in enumerate(list_pred[index_t][1]):
            if index_lp >= at:
                break
            if t[1] == lp:
                aps.append((1/at)*(at/ranking[index_lp]))
                hit = True
        if not(hit):
            aps.append(0)
    return aps

def _map(true, list_pred, at):
    return np.mean(_ap(true, list_pred, at))