import pandas as pd
import numpy as np
from ast import literal_eval

from ephen_utils import modified_map
from ephen_utils import map

path = '/media/pauloricardo/basement/projeto/'

targets = [377904, 375777, 380274, 389293, 388224, 397968, 394909, 394491, 402610]
algorithms = ['regularization', 'deep_walk', 'node2vec', 'struc2vec', 'metapath2vec', 'line', 'gcn']
splits = [0.2]
types = ['actor', 'event', 'location']
ap_at = [1, 3, 5]

results_df = {'ap@': [], 'algorithm': [], 'target': [], 'iteration':[], 'split': [], 'type': [], 'value': []}
for algorithm in algorithms:
    for target in targets:
        for at in ap_at:
            for iteration in range(10):
                for split in splits:
                    for _type in types:
                        file_path = '{}restored_{}/{}_{}_{}_{}.csv'.format(path, _type, algorithm, target, iteration, split)
                        restored_df = pd.read_csv(file_path).drop(columns='Unnamed: 0').reset_index(drop=True)
                        restored_df['true'] = restored_df['true'].apply(literal_eval)
                        restored_df['restored'] = restored_df['restored'].apply(literal_eval)
                        results_df['ap@'].append(at)
                        results_df['algorithm'].append(algorithm)
                        results_df['target'].append(target)
                        results_df['iteration'].append(iteration)
                        results_df['split'].append(split)
                        results_df['type'].append(_type)
                        results_df['value'].append(modified_map(restored_df.true.to_list(), restored_df.restored.to_list(), at))
                        #results_df['value'].append(map(restored_df.true.to_list(), restored_df.restored.to_list(), at))
                        
results_df = pd.DataFrame(results_df)
results_df_mean = results_df.groupby(by=['ap@', 'algorithm', 'target', 'split', 'type'], as_index=False).mean()
results_df_var = results_df.groupby(by=['ap@', 'algorithm', 'target', 'split', 'type'], as_index=False).std()
results_df_mean['variance'] = results_df_var['value']
print(results_df_mean)
from datetime import datetime
results_df_mean.to_csv('{}new_restored_results/results_{}.csv'.format(path, datetime.now()))


"""
import seaborn as sns
import matplotlib.pyplot as plt
from textwrap import wrap
from ephin_utils import decode_html_text

df = pd.read_csv('/media/pauloricardo/basement/projeto/df11-03.csv').drop(columns='Unnamed: 0').reset_index(drop=True)

results_df['algorithm'] = results_df['algorithm'].apply(lambda x: x if x != 'regularization' else 'ephin')
results_df['value'] = results_df['value'].apply(lambda x: x * 100)
results_df['split'] = results_df['split'].apply(lambda x: str(x * 100))

for idxt, target in enumerate(targets):
    results_filtered = results_df[results_df['target'] == target]
    for idxe, edge_type in enumerate(edge_types):
        types_filtered = results_filtered[results_filtered['type'] == edge_type]
        if types_filtered.shape[0] >= 1:
            plt.figure(idxt + idxe)
            ax = sns.lineplot(x="split", y="value", hue="algorithm", marker="o", data=types_filtered)
            ax.set_title("\n".join(wrap('event: ' + decode_html_text(df['text'].iloc[target]) + ' (event to ' + edge_type.split('_')[1] + ' link prediction)')), fontsize=18)
            ax.set_xlabel('removed (%)', fontsize=14)
            ax.set_ylabel('accuracy (%)', fontsize=14)
            ax.get_figure().set_size_inches(12,8)
            ax.get_figure().savefig('/media/pauloricardo/basement/projeto/line_graphs/line_' + str(target) + edge_type + '.pdf')"""
