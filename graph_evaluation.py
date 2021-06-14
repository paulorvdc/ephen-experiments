import pandas as pd
import glob
import os
from ast import literal_eval

from ephin_utils import is_equal
from ephin_utils import get_metric

path = '/media/pauloricardo/basement/projeto/restored/'
all_files = glob.glob(os.path.join(path, "*.csv"))

targets = [377904, 375777,  380274, 389293, 388224, 397968, 394909, 394491, 402610]
algorithms = ['regularization', 'deep_walk', 'node2vec', 'line', 'struct2vec', 'gcn']
splits = [0.05, 0.1, 0.15, 0.2]
metrics = ['acc']
edge_types = ['event_location', 'event_actor', 'event_event']

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
                    if pred.shape[0] == 0:
                        pred = pd.Series()
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
results_df.to_csv('/media/pauloricardo/basement/projeto/restored_results/results_{}.csv'.format(date.today()))


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
            ax.get_figure().savefig('/media/pauloricardo/basement/projeto/line_graphs/line_' + str(target) + edge_type + '.pdf')
