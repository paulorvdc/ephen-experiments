# ephen-experiments
event network completion experiments with ephen and other information network embedding methods
experiments run from graphs at `graph_embeddings.py`
distilbert can be applied on gdelt data with text in at `distilbert_from_gdelt.py`
graphs can be generated from distibert and gdelt data at `ephen_utils.py`

## how to cite
```
@INPROCEEDINGS{9671645,  
  author={do Carmo, Paulo and Marcacini, Ricardo},  
  booktitle={2021 IEEE International Conference on Big Data (Big Data)},   
  title={Embedding propagation over heterogeneous event networks for link prediction},   
  year={2021},  
  volume={}, 
  number={},  
  pages={4812-4821},  
  doi={10.1109/BigData52589.2021.9671645}
}
```

## GraphEmbeddings
GraphEmbeddings submodule based on https://github.com/shenweichen/GraphEmbedding but the used algorithms works with tf 2.x
### install
inside GraphEmbeddings directory from this repository run
```
python setup.py install
```

## gcn
GCN submodule based on https://github.com/dbusbridge/gcn_tutorial

## networks
all the events represented in the context sub-netwokrs were extracted from the GDELT project (https://www.gdeltproject.org/) and are represented as networkx (https://networkx.org) pickle5 graphs. all the networks have DistilBERT-multilingual (https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased) embedding for the events, and other objects available from the GDELT project database. you can download them at:
https://drive.google.com/drive/folders/1_Wz9O4Nzr8JgjzbMzMHI54M3LITG7PCZ?usp=sharing

### networks like these are required to the experiments, so download them (or create yours) to an accessible directory on your machine and update the path on the code. 

## other results
a compilation of other results from experiments with their respective targets MAP averages and standard deviation can be found at: https://docs.google.com/spreadsheets/d/1ub2Fya25PS0GDY0joM8HOe3vCk7SY7xMgXaaMzPdK6Q/edit?usp=sharing
