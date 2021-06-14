import networkx as nx
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from textwrap import wrap
import pickle5 as pickle
from scipy.spatial.distance import cosine
from tqdm import tqdm

from ephin_utils import decode_html_text

df = pd.read_csv('/media/pauloricardo/basement/projeto/df11-03.csv').drop(columns='Unnamed: 0').reset_index(drop=True)
data = pd.read_csv('/media/pauloricardo/basement/projeto/data11-03.csv').drop(columns='Unnamed: 0').reset_index(drop=True)
targets = [377904, 375777,  380274, 389293, 388224, 397968, 394909, 394491, 372939, 402610, 380994, 377199, 389118]
path = "/media/pauloricardo/basement/projeto/"


for target in targets:
    with open(path + "graphs/graph_" + str(target) + ".gpickle", "rb") as fh:
        G = pickle.load(fh)
    precursor = {'event': [], 'date': [], 'week': [], 'year': [], 'cos': []}
    for node in tqdm(G.nodes):
        if G.nodes[node]['node_type'] == 'event':
            for neighbor in G.neighbors(node):
                if G.nodes[neighbor]['node_type'] == 'date':
                    precursor['event'].append(node)
                    precursor['date'].append(neighbor)
                    neighbor_split = neighbor.split('-')
                    precursor['week'].append(int(neighbor_split[0]))
                    precursor['year'].append(int(neighbor_split[1]))
                    precursor['cos'].append(1 - cosine(G.nodes[node]['embedding'], data.iloc[target]))
    precursor = pd.DataFrame(precursor)
    precursor = precursor.sort_values(by=['year', 'week']).reset_index(drop=True)
    precursor.to_csv(path + "/precursors/" + str(target) + ".csv")
    print(precursor)
    
    target_date = pd.to_datetime(df['DATE'].iloc[target], format='%Y-%m-%d')
    plt.figure(target)
    g = sns.lineplot(x="date", y="cos",
             data=precursor)
    g.set_title("\n".join(wrap('event: ' + decode_html_text(df['text'].iloc[target]) + '. happened at week ' + str(target_date.week) + ' of the year ' + str(target_date.year))), fontsize=18)
    g.set_xlabel('week-year', fontsize=14)
    g.set_ylabel('cosine', fontsize=14)
    for item in g.get_xticklabels():
        item.set_rotation(45)
    g.get_figure().set_size_inches(12,8)
    g.get_figure().savefig(path + "/precursors/" + str(target) + ".pdf") 