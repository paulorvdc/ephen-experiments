# ephin-experiments
event prediction experiments with ephin and other information network embedding methods

## GraphEmbeddings
GraphEmbeddings submodule based on https://github.com/shenweichen/GraphEmbedding but the used algorithms work with tf 2.x
### install
inside GraphEmbeddings directory run
```
python setup.py install
```

## dataset
### graphs
all the events represented in the context subgraphs were extracted from GDELT project (https://www.gdeltproject.org/) and are represented as networkx (https://networkx.org) pickle5s with a DistilBERT-multilingual (https://huggingface.co/distilbert-base-multilingual-cased) embedding and they can be found at:
https://drive.google.com/drive/folders/11OhFR3ycVxMAVIfQaiqmIY7ZI3hJFk4D?usp=sharing

graphs are required to the experiments, so download them (or create yours) and leave them accessible on your machine and update the path on the python code. 

## wip
new experiments, methods, data and usability will be improved and updated with the research progress.
