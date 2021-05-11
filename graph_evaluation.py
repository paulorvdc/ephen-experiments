import pandas as pd
import numpy as np
import glob
import os
from ast import literal_eval

from ephin_utils import is_equal
from ephin_utils import get_metric

path = '/media/pauloricardo/basement/projeto/restored_new/'
all_files = glob.glob(os.path.join(path, "*.csv"))

targets = [377904, 375777,  380274, 377199, 389118, 389293, 388224, 397968, 394909, 394491, 372939, 402610, 380994]
algorithms = ['regularization', 'deep_walk', 'node2vec', 'line', 'struct2vec', 'gcn']
splits = [0.05, 0.1, 0.15, 0.2]
metrics = ['recall']
edge_types = ['event_location', 'event_actor']

results_df = {'metric': [], 'algorithm': [], 'target': [], 'split': [], 'type': [], 'value': []}
for algorithm in algorithms:
    for target in targets:
        for i in range(10):
            for split in splits:
                file_path = '{0}{1}_{2}_{3}_{4}.csv'.format(path, algorithm, target, i, split)
                if file_path in all_files:
                    restored_df = pd.read_csv(file_path).drop(columns='Unnamed: 0').reset_index(drop=True)
                    restored_df['true'] = restored_df['true'].apply(literal_eval)
                    restored_df['restored'] = restored_df['restored'].apply(literal_eval)
                    pred = restored_df.apply(is_equal, axis=1)
                    for metric in metrics:
                        results_df['metric'].append(metric)
                        results_df['algorithm'].append(algorithm)
                        results_df['target'].append(target)
                        results_df['split'].append(split)
                        results_df['type'].append('all')
                        results_df['value'].append(get_metric(metric, pred))
                        for edge_type in edge_types:
                            results_df['metric'].append(metric)
                            results_df['algorithm'].append(algorithm)
                            results_df['target'].append(target)
                            results_df['split'].append(split)
                            results_df['type'].append(edge_type)
                            filtered_pred = pd.Series()
                            if edge_type == 'event_actor':
                                filtered_pred = pred[restored_df['edge_type'] == ('event_person' or 'event_org')]
                            else:
                                filtered_pred = pred[restored_df['edge_type'] == edge_type]
                            results_df['value'].append(get_metric(metric, filtered_pred))
                        
results_df = pd.DataFrame(results_df)
results_df = results_df.groupby(by=['metric', 'algorithm', 'target', 'split', 'type'], as_index=False).mean()
from datetime import date
results_df.to_csv('/media/pauloricardo/basement/projeto/restored_results/results_{}.csv'.format(data.today()))

"""
import seaborn as sns
import matplotlib.pyplot as plt

for idxt, target in enumerate(targets):
    for idxs, split in enumerate(splits):
        results_filtered = results_df[results_df['target'] == target]
        results_filtered = results_filtered[results_filtered['split'] == split]
        all_filtered = results_filtered[results_filtered['type'] == 'all']
        if all_filtered.shape[0] >= 1:
            plt.figure(idxt + idxs)
            ax = sns.barplot(x="algorithm", y="value", hue="metric", data=all_filtered).set_title(str(target) + '_' + str(split))
            plt.show()
        for idxe, edge_type in enumerate(edge_types): 
            type_filtered = results_filtered[results_filtered['type'] == edge_type]
            type_filtered.to_csv('/media/pauloricardo/basement/projeto/restored_results/{0}_{1}_{2}.csv'.format(target, split, edge_type))
            if type_filtered.shape[0] >= 1:
                plt.figure(idxt + idxs + idxe)
                ax = sns.barplot(x="algorithm", y="value", hue="metric", data=type_filtered).set_title(str(target) + '_' + str(split) + '_' + edge_type)
                plt.show()"""
