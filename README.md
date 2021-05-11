# ephin-experiments
event prediction experiments with ephin and other information network embedding methods

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
all the events represented in the context sub-netwokrs were extracted from the GDELT project (https://www.gdeltproject.org/) and are represented as networkx (https://networkx.org) pickle5 graphs. all the networks have DistilBERT-multilingual (https://huggingface.co/distilbert-base-multilingual-cased) embedding for the events, and other objects available from the GDELT project database. you can download them at:
https://drive.google.com/drive/folders/11OhFR3ycVxMAVIfQaiqmIY7ZI3hJFk4D?usp=sharing

### networks like these are required to the experiments, so download them (or create yours) to an accessible directory on your machine and update the path on the code. 

## wip
new experiments, methods, data and usability will be improved and updated with the research progress.
