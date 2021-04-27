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
networkx (https://networkx.org) graphs with a DistilBERT-multilingual (https://huggingface.co/distilbert-base-multilingual-cased) embedding can be found at:
https://drive.google.com/drive/folders/11OhFR3ycVxMAVIfQaiqmIY7ZI3hJFk4D?usp=sharing

graphs are required to the experiments, so download them (or create yours) and leave them accessible on your machine and update the path on the python code. 

### GDELT events
all the events extracted from GDELT project (https://www.gdeltproject.org/) can be found at:
https://drive.google.com/drive/folders/1BApN5s8hxhm1wOs-4FrEykZzFpkP7zW_?usp=sharing


## wip
new experiments, methods and data will be updated with the research progress.
