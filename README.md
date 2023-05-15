# Structure-Graph-Pair-Hackathon

## Objective
To develop a code-base for graph deep learning models using protein structures as input, with a focus on residue-level prediction tasks and pair input (i.e protein-protein, protein-peptide, pocket-ligand, protein-ligand). 

## Dataset

The dataset consists of protein complexes with 2 sets of interacting chains, extracted from [the MaSIF paper](https://www.nature.com/articles/s41592-019-0666-6). Interface residues are defined as residues that have at least one heavy atom within a distance threshold (6A) from the other chain set. The train-test split is based on structure comparison of interfaces, so should be robust.

```
data/
    raw/
        PDB files of individual chain-sets (can be more than one chain in a chain-set)
    training.txt
        List of interacting chain-set pairs <pdbid>_<chainA>_<chainB>
    testing.txt
        List of interacting chain-set pairs <pdbid>_<chainA>_<chainB>
    interface_labels.txt
        Tab-separated list of <pdbid_chainA> <pdbid_chainB> <chainA interface chain and residue numbers> <chainB interface chain and residue numbers>
```

## Environment setup
Download [mamba](https://github.com/conda-forge/miniforge#mambaforge)

```sh
chmod +x Mambaforge.sh
./Mambaforge.sh
```

To use with pascal or rtx8000 GPU nodes:
```sh
srun --nodes=1 --cpus-per-task=8 --mem=16G --gres=gpu:1 --partition=rtx8000,pascal --pty bash
mamba create -n hackathon pytorch torchvision torchaudio pytorch-cuda=11.7 pyg -c pytorch -c nvidia -c pyg
```
Exit the interactive node

On the worker node:
```sh
conda activate hackathon
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
pip install pytorch-lightning 'graphein[extras]' egnn-pytorch tensorboard 'jsonargparse[signatures]'
```

## Libraries

### [Graphein](https://github.com/a-r-j/graphein)
Used to produce graphs of protein structures (and small molecules) with different methods for creating edges, and different node and edge features. The documentation website is sparse in some areas, look at the code instead. See `utils/load_protein_as_graph` for an example.


### [Pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html): 
Graph deep learning library

Provides 
- ready-to-use [models](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#models) - e.g the `GAT` in `models/single_models/GATModel` 
- [transforms](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#graph-transforms) to add edges, node features, and edge features to your graphs - see `dataloader/single_loader/ProteinDataModule`
- [data and dataset](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html) objects for individual graphs (see `dataloader/single_loader`) and pairs of graphs with HeteroData (see `dataloader/pair_loader`)
- [Heterogenous graph learning](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/heterogeneous.html) for paired data - see `models/pair_models/CrossGAT`

### [Pytorch-lightning](https://lightning.ai/docs/pytorch/latest/)
- [LightningDataModule](https://lightning.ai/docs/pytorch/latest/api/lightning.pytorch.core.LightningDataModule.html?highlight=lightningdatamodule#lightning.pytorch.core.LightningDataModule) - see `dataloader/single_loader/ProteinDataModule` and `dataloader/pair_loader/ProteinPairDataModule`
- [LightningModule](https://lightning.ai/docs/pytorch/latest/common/lightning_module.html) - see `models/single_models/GATModel`
- logging to Tensorboard, checkpoints, early stopping, and other callbacks - see `config.yaml` and `callbacks.py`
- [Trainer](https://lightning.ai/docs/pytorch/latest/common/trainer.html) - see `config.yaml` and `main.py`
- [Configuration](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html) - see `config.yaml` and `sbatch_train.sh`

### [egnn-pytorch](https://github.com/lucidrains/egnn-pytorch)
E(3)-Equivariant Graph Neural Networks

Usage:
```python
from egnn_pytorch import EGNN_Sparse
egnn_layer = EGNN_Sparse(feats_dim=in_channels)
# Assume Data object with data.pos and data.x attributes given as input to `forward` function
new_x = torch.cat([pos, x], dim=-1)
new_x = egnn_layer(new_x, edge_index)
new_x = x[:, 3:]
```

# Tasks

1. Data Preparation
   - [ ] Construct graphs from protein structures
   - [ ] Explore different featurisations
   - [ ] Explore different ways of making cross-edges for paired data
   - [ ] Load and batch data

2. Architecture
   - [ ] Explore off-the-shelf graph-based models
   - [ ] Incorporate pair architectures (e.g cross-attention)
   - [ ] Incorporate EGNN layers
   - [ ] Implement evaluation metrics

3. Training and Tracking
   - [ ] Train and validate the models
   - [ ] Save and load model checkpoints
   - [ ] Log, visualise and track model(s) performance
   - [ ] Optimise and tune model hyperparameters
