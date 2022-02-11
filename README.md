# GPA

This is a pytorch and pytorch-geometric based implementation of **Graph Contrastive Learning with Personalized Augmentation**. 

## Installation

The required packages can be installed by running `pip install -r requirements.txt`.

## GPA

## Datasets
The datasets used in our paper can be automatically downlowad. 

## Quick Start
Train on the TUDatasets (PROTEINS, NCI1, DD, COLLAB, MUTAG, IMDB-BINARY, REDDIT-BINARY, REDDIT-MULTI-5K, github_stargazers):
```
python main_gpa.py --dataset "MUTAG" --mode unsuper
python main_gpa.py --dataset "MUTAG" --mode semi
```