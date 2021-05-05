import pandas as pd

from ephin_utils import decode_html_text

path = "/media/pauloricardo/basement/projeto/"
targets = [389118]
df = pd.read_csv('/media/pauloricardo/basement/projeto/df11-03.csv').drop(columns='Unnamed: 0').reset_index(drop=True)

for target in targets:
    precursor = pd.read_csv(path + "/precursors/" + str(target) + ".csv").drop(columns='Unnamed: 0').reset_index(drop=True)
    precursor_named = precursor.join(df.set_index('GKGRECORDID'), on='event')
    precursor_named = precursor_named[['text', 'date']]
    precursor_named['text'] = precursor_named['text'].apply(decode_html_text)
    precursor_named.to_csv(path + "/precursors/" + str(target) + "_named.csv")
    print(precursor_named)
