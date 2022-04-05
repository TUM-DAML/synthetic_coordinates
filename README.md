# Directional Message Passing on Molecular Graphs via Synthetic Coordinates

![Synthetic coordinates and GNN conversion to directional message passing.](https://www.in.tum.de/fileadmin/_processed_/a/8/csm_sc_figure_d484aa3e94.png)

Reference implementation of synthetic coordinates and directional message passing for multiple GNNs, as proposed in

**[Directional Message Passing on Molecular Graphs via Synthetic Coordinates](https://www.in.tum.de/daml/synthetic-coordinates/)**  
by Johannes Gasteiger, Chandan Yeshwanth, Stephan Günnemann  
Published at NeurIPS 2021.

Note that the author's name has changed from Johannes Klicpera to Johannes Gasteiger.

The `deepergcn_smp` folder contains DeeperGCN and SMP implementations in PyTorch, and 
`dimenetpp` contains the TensorFlow implementation of DimeNet++ with synthetic coordinates.

## Requirements

We use separate Anaconda environments for DeeperGCN/SMP and DimeNet++. Use this
command to create the respective environments for each folder
```
conda env create -f environment.yml
```

### Datasets
We use the `ogbg-molhiv` and `ZINC` datasets from [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/),
which are automatically downloaded to the `data` folder. The QM9 dataset is provided in the `data` folder. 

## Training
Reference training scripts with the best hyperparameters are included.

### DeeperGCN and SMP
You can select the model (`deepergcn` or `smp`) and the dataset (`ogbg-molhiv`, `QM9`, `ZINC`) in the script. 
We provide reference training scripts in the `scripts` folder for:

1. the baseline model: `python scripts/train_baseline.py`
2. baseline model with distance: bounds matrix (BM) or PPR: `python scripts/train_sc_basic.py`
3. and linegraph with distance and angle using both BM and PPR: `python scripts/train_sc_linegraph.py`

These can be modified to perform other ablations, such as choosing any one of 
the distance methods, or using only the distance on the linegraph.

### DimeNet++
The model hyperparameters and ablations can be configured in the training script. 
```
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
the unique experiment ID is printed to console as well as written to the SEML 
database. You can also use the `results.ipynb` notebook to fetch results from the 
SEML Database. Set the collection name and batch IDs in the notebook and run
to fetch the required results.

### DimeNet++
Checkpoints are saved to a uniquely named folder. This unique name is printed 
during training and can be used in the `predict.ipynb` notebook to run 
on the test set. The model configuration used during training must be specified in `config_pp.yaml`.
The same unique name can be used to view losses and metrics in Tensorboard.

## Results

Our models achieve the following results (as reported in the paper)

### ZINC
| Model     | MAE           |
| --------- | ------------- |
| DeeperGCN | 0.142 +-0.006 |
| SMP       | 0.109 +-0.004 |

### Coordinate-free QM9
| Model     | Target | MAE (meV) |
| --------- | ------ | --------- |
| DimeNet++ | U0     | 28.7      |
| DimeNet++ | HOMO   | 61.7      |


## Contact
Please contact j.gasteiger@in.tum.de if you have any questions.

## Cite
Please cite our paper if you use our method or code in your own work:

```
@inproceedings{gasteiger_2021_dmp,
  title={Directional Message Passing on Molecular Graphs via Synthetic Coordinates},
  author={Gasteiger, Johannes and Yeshwanth, Chandan and G{\"u}nnemann, Stephan},
  booktitle = {Thirty-fifth Conference on Neural Information Processing Systems (NeurIPS)},
  year={2021},
}
```

## License
Hippocratic License v2.1
