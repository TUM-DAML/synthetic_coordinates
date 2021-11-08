# Directional Message Passing on Molecular Graphs via Synthetic Coordinates

This repository is the official implementation of 

**[Directional Message Passing on Molecular Graphs via Synthetic Coordinates](https://openreview.net/forum?id=ZRu0_3azrCd)**  
by Johannes Klicpera, Chandan Yeshwanth, Stephan Günnemann  
Published at NeurIPS 2021.

The `deepergcn_smp` folder contains DeeperGCN and SMP implementations in torch, and 
`dimenetpp` contains the Tensorflow implementation of Dimenet++.

## Requirements

We use separate conda environments for DeeperGCN/SMP and dimenetpp. Use this
command to create the respective environments
```
conda env create -f environment.yml
```

### Datasets
We use the `ogbg-molhiv` and `ZINC` datasets from [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) which are automatically downloaded to the `data` folder.
The QM9 dataset is provided in the `data` folder. 

## Training
Reference training scripts with the best hyperparameters are included
### DeeperGCN and SMP
The model: `deepergcn` and `smp` can be configured in the script, as well as
the dataset: `ogbg-molhiv`, `QM9`, `ZINC`. 
We provide reference training scripts in the `scripts` folder for -

1. the baseline model: `python scripts/train_baseline.py`
1. baseline model with distance: bounds matrix (BM) or PPR: `python scripts/train_sc_basic.py`
1. and linegraph with distance and angle using both BM and PPR: `python scripts/train_sc_linegraph.py`

These can be modified to perform other ablations.

### DimeNet++
The model parameters and ablations can be configured in the training script. 
```
cd dimenetpp
python run.py
```

### Training with SEML
Alternately, use the config file to train with [SEML](https://github.com/TUM-DAML/seml) on a Slurm cluster.

```
seml <collection> add configs/graph_clsreg.yml
seml <collection> start
```

## Evaluation
### DeeperGCN and SMP
The model is evaluated on the validation set during training, and the final test
score is printed at the end of training. Logs with losses and metrics are written to Tensorboard,
the unique experiment ID is printed to console.

### DimeNet++
Checkpoints are saved to a uniquely named folder, this unique name is printed 
during training and can be used in the `predict.ipynb` notebook to run 
on the test set. The model configuration as used during training must be specified in `config_pp.yaml`. The same unique name can be used to view losses
and metrics in Tensorboard.

## Pre-trained Models

Pretrained models will be added soon.

## Results

Our models achieve the following results as reported in the paper

### ogbg-molhiv
| Model         |  Accuracy |
| ------------------ | -------------- |
| DeeperGCN   |          0.7674 +-0.0162       |

### ZINC
| Model         |  MAE |
| ------------------ | -------------- |
| DeeperGCN   |  0.1423 +- 0.0064 |
| SMP   | 0.1263 +- 0.0039 |

### QM9
| Model         |  Target | MAE |
| ------------------ | ----- | --------- |
| DimeNet++   |     U0         |      28.7       |
| DimeNet++   |     epsilon HOMO         |      61.7       |


## Contact
Please contact klicpera@in.tum.de if you have any questions.

## Cite
Please cite our paper if you use our method or code in your own work:

```
@inproceedings{klicpera_2021_dmp,
  title={Directional Message Passing on Molecular Graphs via Synthetic Coordinates},
  author={Klicpera, Johannes and Yeshwanth, Chandan and G{\"u}nnemann, Stephan},
  booktitle = {Thirty-fifth Conference on Neural Information Processing Systems (NeurIPS)},
  year={2021},
}
```

## License
Hippocratic License