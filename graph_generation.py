import pandas as pd
import networkx as nx
import numpy as np
from scipy.spatial.distance import cosine
from copy import deepcopy

from ephen_utils import difference
from ephen_utils import make_hin
from ephen_utils import inner_connections

data = pd.read_csv('/media/pauloricardo/basement/projeto/data11-03.csv').drop(columns='Unnamed: 0').reset_index(drop=True)
df = pd.read_csv('/media/pauloricardo/basement/projeto/df11-03.csv').drop(columns='Unnamed: 0').reset_index(drop=True)
stats = pd.read_csv('/media/pauloricardo/basement/projeto/stats_filtered11-03.csv').drop(columns='Unnamed: 0').reset_index(drop=True)

df['date_str'] = df['DATE']
df['DATE'] = pd.to_datetime(df['DATE'], format='%Y-%m-%d')
   
for index, row in stats.iterrows():
    y = df['DATE'].apply(difference, end=df['DATE'].iloc[row['target']], interval='week')
    X = data[y > 0]
    filtered_df = df[y > 0]
    y = y[y > 0]
    filtered_df['dis_cos'] = data.apply(cosine, axis=1, v=data.iloc[row['target']])
    X = X[filtered_df['dis_cos'] <= 0.5]
    y = y[filtered_df['dis_cos'] <= 0.5]
    filtered_df = filtered_df[filtered_df['dis_cos'] <= 0.5]
    filtered_df = filtered_df.reset_index().drop(columns='index')
    
    G = make_hin(X.to_numpy(), filtered_df)

    G = inner_connections(G)
    
    nx.write_gpickle(G, "/media/pauloricardo/basement/projeto/graphs/graph_" + str(row['target']) + ".gpickle")
    