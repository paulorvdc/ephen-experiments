import pandas as pd
import networkx as nx
import numpy as np
from scipy.spatial.distance import cosine
from copy import deepcopy

from ephen_utils import difference
from ephen_utils import make_hin
from ephen_utils import inner_connections

df = pd.concat([pd.read_parquet('/media/pauloricardo/basement/projeto/df01-10_1.parquet'), pd.read_parquet('/media/pauloricardo/basement/projeto/df01-10_2.parquet')])

stats = pd.read_csv('/media/pauloricardo/basement/projeto/stats_filtered01-10.csv').drop(columns='Unnamed: 0').reset_index(drop=True)

df['date_str'] = df['DATE']
df['DATE'] = pd.to_datetime(df['DATE'], format='%Y-%m-%d')
   
for index, row in stats.iterrows():
    y = df['DATE'].apply(difference, end=df['DATE'].iloc[row['target']], interval='week')
    X = df[y > 0].embedding
    filtered_df = df[y > 0]
    y = y[y > 0]
    filtered_df['dis_cos'] = df.embedding.apply(cosine, v=df.embedding.iloc[row['target']])
    X = X[filtered_df['dis_cos'] <= 0.5]
    y = y[filtered_df['dis_cos'] <= 0.5]
    filtered_df = filtered_df[filtered_df['dis_cos'] <= 0.5]
    filtered_df = filtered_df.reset_index().drop(columns='index')
    
    G = make_hin(X.to_numpy(), filtered_df)

    G = inner_connections(G)
    
    nx.write_gpickle(G, "/media/pauloricardo/basement/projeto/graphs/graph_" + str(row['target']) + ".gpickle")
    