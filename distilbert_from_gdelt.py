#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 13:48:42 2020

@author: pauloricardo
"""

#get GDELT data and make the embeddings
import pandas as pd
import glob
import os

from ephen_utils import decode_html_text
from ephen_utils import date_string

path = r'/media/pauloricardo/basement/projeto/datasetsGDELT/'
all_files = glob.glob(os.path.join(path, "*.csv"))

df_from_each_file = (pd.read_csv(f) for f in all_files)
df = pd.concat(df_from_each_file, ignore_index=True)
del df_from_each_file

df = df.drop(columns='Unnamed: 0').reset_index().drop(columns='index')
df['text'] = df['text'].astype('string')
df = df[df['text'] != ''].reset_index().drop(columns='index')
df['DATE'] = df['DATE'].apply(date_string)
df = df.sort_values('DATE').reset_index().drop(columns='index')
df['text'] = df['text'].apply(decode_html_text).astype('string')

from sentence_transformers import SentenceTransformer, LoggingHandler
import numpy as np
import logging

#### Just some code to print debug information to stdout
np.set_printoptions(threshold=100)

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

# Load Sentence model (based on BERT) from URL
model = SentenceTransformer('distiluse-base-multilingual-cased')

df['embedding'] = list(model.encode(df['text'].to_list()))
print(df.info)

# split df
middle = df.shape[0] // 2
df_1 = df[df.index < middle]
print(df_1.info)
df_2 = df[df.index >= middle]
print(df_2.info)
del df

# saving
print('df 1')
df_1.to_parquet('/media/pauloricardo/basement/projeto/df01-10_1.parquet', engine='pyarrow', compression='brotli')
del df_1
print('df 2')
df_2.to_parquet('/media/pauloricardo/basement/projeto/df01-10_2.parquet', engine='pyarrow', compression='brotli')