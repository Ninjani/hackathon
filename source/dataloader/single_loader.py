from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from pytorch_lightning import LightningDataModule
from torch_geometric import transforms as T
from tqdm import tqdm

from ..utils import graphein_to_pytorch_graph, load_protein_as_graph, read_interface_labels

class ProteinDataset(Dataset):
    """
    torch-geometric Dataset class for loading protein files as graphs.
    """
    def __init__(self, root, pdb_dir, node_attr_columns: list, edge_attr_columns: list, 
                 edge_kinds: set,
                 protein_names: list, label_mapping: dict, pre_transform=None, transform=None, num_workers=1):
        self.pdb_dir = Path(pdb_dir)
        self.node_attr_columns = node_attr_columns
        if edge_attr_columns is None:
            edge_attr_columns = []
        self.edge_attr_columns = edge_attr_columns
        self.edge_kinds = edge_kinds
        self.protein_names = protein_names
        self.label_mapping = label_mapping
        self.num_workers = num_workers
        super(ProteinDataset, self).__init__(root, pre_transform=pre_transform, transform=transform)
        
    def download(self):
        for protein_name in tqdm(self.protein_names):
            self._download_one(protein_name)

    def _download_one(self, protein_name):
        output = Path(self.raw_dir) / f'{protein_name}.pkl'
        if not output.exists():
            graph = load_protein_as_graph(self.pdb_dir / f"{protein_name}.pdb")
            with open(output, "wb") as f:
                pickle.dump(graph, f)

    @property
    def raw_file_names(self):
        return [Path(self.raw_dir) / f"{protein_name}.pkl" for protein_name in self.protein_names]
    
    @property
    def processed_file_names(self):
        return [Path(self.processed_dir) / f"{protein_name}.pt" for protein_name in self.protein_names]

    def process(self):
        for protein_name in self.protein_names:
            output = Path(self.processed_dir) / f'{protein_name}.pt'
            if output.exists():
                continue
            with open(Path(self.raw_dir) / f"{protein_name}.pkl", "rb") as f:
                data = pickle.load(f)
            data = graphein_to_pytorch_graph(data, self.node_attr_columns, self.edge_attr_columns, self.edge_kinds, self.label_mapping[protein_name])
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            torch.save(data, output)


    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        data = torch.load(self.processed_file_names[idx])
        return data
    

class ProteinDataModule(LightningDataModule):
    """
    Pytorch Lightning DataModule wrapping the ProteinDataset class.
    Here you set/choose
    - how training, validation and test data are split
    - the batch size and number of workers for the DataLoader
    - the transforms to apply to the graphs
    """
    def __init__(self, root, pdb_dir, node_attr_columns, edge_attr_columns, edge_kinds, batch_size: int = 32, num_workers: int = 1):
        super().__init__()
        self.pdb_dir = pdb_dir
        self.node_attr_columns = node_attr_columns
        if edge_attr_columns is None:
            edge_attr_columns = []
        self.edge_attr_columns = edge_attr_columns
        self.edge_kinds = edge_kinds
        self.train_file = Path(root) / "training.txt"
        self.test_file = Path(root) / "testing.txt"
        self.full_list_file = Path(root) / "full_list.txt"
        self.labels_file = Path(root) / "interface_labels.txt"
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pre_transform = None
        self.transform = None

    def prepare_data(self):
        protein_pair_names_to_labels = read_interface_labels(self.labels_file)
        protein_names_to_labels = {}
        for p1, p2 in protein_pair_names_to_labels:
            protein_names_to_labels[p1] = protein_pair_names_to_labels[(p1, p2)][0]
            protein_names_to_labels[p2] = protein_pair_names_to_labels[(p1, p2)][1]
        protein_names = set()
        with open(self.full_list_file, "r") as f:
            for line in f:
                pdb_id, chainA, _ = line.strip().split("_")
                name = f"{pdb_id}_{chainA}"
                if name in protein_names_to_labels:
                    protein_names.add(name)
        protein_names = list(protein_names)[:30]
        ProteinDataset(root=self.root, pdb_dir=self.pdb_dir, 
                       node_attr_columns=self.node_attr_columns, 
                       edge_attr_columns=self.edge_attr_columns,
                       edge_kinds=self.edge_kinds, 
                       protein_names=protein_names, 
                       label_mapping=protein_names_to_labels, 
                       pre_transform=self.pre_transform, transform=self.transform)

    def setup(self, stage: str):
        protein_pair_names_to_labels = read_interface_labels(self.labels_file)
        protein_names_to_labels = {}
        for p1, p2 in protein_pair_names_to_labels:
            protein_names_to_labels[p1] = protein_pair_names_to_labels[(p1, p2)][0]
            protein_names_to_labels[p2] = protein_pair_names_to_labels[(p1, p2)][1]
        if stage == "fit":
            protein_names = set()
            with open(self.train_file, "r") as f:
                for line in f:
                    pdb_id, chainA, _ = line.strip().split("_")
                    name = f"{pdb_id}_{chainA}"
                    if name in protein_names_to_labels:
                        protein_names.add(name)
            protein_names = list(protein_names)
            # split train and val TODO: improve this
            self.train_protein_names, self.val_protein_names = train_test_split(protein_names, test_size=0.2, random_state=42)
            self.train_dataset = ProteinDataset(root=self.root, 
                                                 pdb_dir=self.pdb_dir, 
                                                node_attr_columns=self.node_attr_columns, 
                                                edge_attr_columns=self.edge_attr_columns,
                                                edge_kinds=self.edge_kinds, protein_names=self.train_protein_names, label_mapping=protein_names_to_labels, 
                                                transform=self.transform, pre_transform=self.pre_transform)
            self.val_dataset = ProteinDataset(root=self.root,  pdb_dir=self.pdb_dir, 
                                                node_attr_columns=self.node_attr_columns, 
                                                edge_attr_columns=self.edge_attr_columns,
                                                edge_kinds=self.edge_kinds, protein_names=self.val_protein_names, label_mapping=protein_names_to_labels, 
                                              transform=self.transform, pre_transform=self.pre_transform)
        elif stage == "test":
            self.test_protein_names = set()
            with open(self.test_file, "r") as f:
                for line in f:
                    pdb_id, chainA, _ = line.strip().split("_")
                    name = f"{pdb_id}_{chainA}"
                    if name in protein_names_to_labels:
                        self.test_protein_names.add(name)
            self.test_protein_names = list(self.test_protein_names)
            self.test_dataset = ProteinDataset(root=self.root, 
                                               pdb_dir=self.pdb_dir, 
                                               node_attr_columns=self.node_attr_columns, 
                                               edge_attr_columns=self.edge_attr_columns,
                                               edge_kinds=self.edge_kinds, 
                                               protein_names=self.test_protein_names, label_mapping=protein_names_to_labels, 
                                               transform=self.transform, pre_transform=self.pre_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
