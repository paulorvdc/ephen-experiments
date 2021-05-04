import networkx as nx
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from textwrap import wrap
import pickle5 as pickle
from scipy.spatial.distance import cosine
from bs4 import BeautifulSoup

df = pd.read_csv('/media/pauloricardo/basement/projeto/df11-03.csv').drop(columns='Unnamed: 0').reset_index(drop=True)
data = pd.read_csv('/media/pauloricardo/basement/projeto/data11-03.csv').drop(columns='Unnamed: 0').reset_index(drop=True)
targets = [377904, 375777,  380274, 389293, 388224, 397968, 394909, 394491, 372939, 402610, 380994, 377199, 389118]
path = "/media/pauloricardo/basement/projeto/"


for target in targets:
    with open(path + "graphs/graph_" + str(target) + ".gpickle", "rb") as fh:
        G = pickle.load(fh)
    precursor = {'date': [], 'date_sum': [], 'cos': []}
    for node in G.nodes:
        if G.nodes[node]['node_type'] == 'event':
            for neighbor in G.neighbors(node):
                if G.nodes[neighbor]['node_type'] == 'date':
                    precursor['date'].append(neighbor)
                    neighbor_split = neighbor.split('-')
                    precursor['date_sum'].append(int(neighbor_split[0]) + int(neighbor_split[1]))
                    precursor['cos'].append(1 - cosine(G.nodes[node]['embedding'], data.iloc[target]))
    precursor = pd.DataFrame(precursor)
    precursor = precursor.groupby(by=['date', 'date_sum'], as_index=False).sum().sort_values(by=['date_sum']).reset_index(drop=True)
    print(precursor)
    plt.figure(target)
    g = sns.lineplot(x="date", y="cos",
             data=precursor)
    g.set_title("\n".join(wrap('event: ' + BeautifulSoup(df['text'].iloc[target], 'html.parser').get_text(), 100)), fontsize=18)
    g.set_xlabel('week-year', fontsize=14)
    g.set_ylabel('accmulated cosine', fontsize=14)
    for item in g.get_xticklabels():
        item.set_rotation(45)
    g.get_figure().set_size_inches(12,8)
    g.get_figure().savefig(path + "/precursors/" + str(target) + ".pdf")    