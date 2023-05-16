from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split
from torch_geometric.data import Dataset, Data, HeteroData
from torch_geometric.loader import DataLoader
from pytorch_lightning import LightningDataModule
from torch_geometric import transforms as T
import torch
from tqdm import tqdm

from ..utils import graphein_to_pytorch_graph, load_protein_as_graph, read_interface_labels


def make_hetero_graph(graph_1: Data, graph_2: Data, sasa_threshold=None):
    """
    Combine two pytorch-geometric Data objects into a single HeteroData object.
    i.e data_1.x and data_2.x become data["protein_1"].x and data["protein_2"].x
    Also add all vs. all edges between the two proteins.
    """    
    graph = HeteroData()
    graph["protein_1"].x = graph_1.x
    graph["protein_1"].y = graph_1.y
    graph["protein_1", "intra", "protein_1"].edge_index = graph_1.edge_index

    graph["protein_2"].x = graph_2.x
    graph["protein_2"].y = graph_2.y
    graph["protein_2", "intra", "protein_2"].edge_index = graph_2.edge_index

    # add edges between interface residues and their distances
    interacting_edge_index = []
    interacting_edge_distance = []
    for i in range(graph_1.x.shape[0]):
        for j in range(graph_2.x.shape[0]):
            if graph_1.y[i] == graph_2.y[j] == 1:
                interacting_edge_index.append([i, j])
                # Euclidean distance between the pos vectors of the two nodes
                interacting_edge_distance.append(torch.norm(graph_1.pos[i] - graph_2.pos[j]))
    graph["protein_1", "interacts", "protein_2"].edge_index = torch.tensor(interacting_edge_index).T
    graph["protein_1", "interacts", "protein_2"].edge_attr = torch.tensor(interacting_edge_distance).unsqueeze(1)

    # add all vs. all edges between the two proteins (or those above sasa_threshold if given)
    edge_index_inter = []
    for i in range(graph_1.x.shape[0]):
        for j in range(graph_2.x.shape[0]):
            if sasa_threshold is None:
                edge_index_inter.append([i, j])
            else:
                if graph_1.sasa[i] > sasa_threshold and graph_2.sasa[j] > sasa_threshold:
                    edge_index_inter.append([i, j])
    graph["protein_1", "inter", "protein_2"].edge_index = torch.tensor(edge_index_inter).T
    return graph


class ProteinPairDataset(Dataset):
    """
    torch-geometric Dataset class for loading pairs of protein files as HeteroData objects.
    """
    def __init__(self, root, pdb_dir, node_attr_columns: list, edge_attr_columns: list, 
                 edge_kinds: set, sasa_threshold: float, 
                 protein_pair_names: list, label_mapping: dict, pre_transform=None, transform=None, num_workers=1):
        self.pdb_dir = Path(pdb_dir)
        self.node_attr_columns = node_attr_columns
        self.edge_attr_columns = edge_attr_columns
        self.edge_kinds = edge_kinds
        self.sasa_threshold = sasa_threshold
        self.protein_pair_names = protein_pair_names
        self.label_mapping = label_mapping
        self.num_workers = num_workers
        self.protein_names = set()
        for name_1, name_2 in self.protein_pair_names:
            self.protein_names.add(name_1)
            self.protein_names.add(name_2)
        super(ProteinPairDataset, self).__init__(root, pre_transform=pre_transform, transform=transform)
        
    def download(self):
        # with torch.multiprocessing.Pool(self.num_workers) as pool:
        #     results = [pool.apply_async(self._download_one, 
        #                                 args=protein_name) for protein_name in self.protein_names]
        #     for result in tqdm(results, total=len(results)):
        #         result.wait()
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
        return [Path(self.processed_dir) / f"{p1}__{p2}.pt" for p1, p2 in self.protein_pair_names]

    def process(self):
        for p1, p2 in self.protein_pair_names:
            output = Path(self.processed_dir) / f'{p1}__{p2}.pt'
            if output.exists():
                continue
            with open(Path(self.raw_dir) / f"{p1}.pkl", "rb") as f:
                data_1 = pickle.load(f)
            with open(Path(self.raw_dir) / f"{p2}.pkl", "rb") as f:
                data_2 = pickle.load(f)
            data_1 = graphein_to_pytorch_graph(data_1, self.node_attr_columns, self.edge_attr_columns, self.edge_kinds, self.label_mapping[(p1, p2)][0])
            data_2 = graphein_to_pytorch_graph(data_2, self.node_attr_columns, self.edge_attr_columns, self.edge_kinds, self.label_mapping[(p1, p2)][1])
            if self.pre_transform is not None:
                data_1 = self.pre_transform(data_1)
                data_2 = self.pre_transform(data_2)
            data = make_hetero_graph(data_1, data_2, self.sasa_threshold)
            torch.save(data, output)

    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        data = torch.load(self.processed_file_names[idx])
        return data
    

class ProteinPairDataModule(LightningDataModule):
    """
    Pytorch Lightning DataModule wrapping the ProteinPairDataset class.
    Here you set/choose
    - how training, validation and test data are split
    - the batch size and number of workers for the DataLoader
    - the transforms to apply to the individual graphs (in pre-transform) and to the combined HeteroData object (in transform)
    """
    def __init__(self, root, pdb_dir, node_attr_columns, edge_attr_columns, edge_kinds, sasa_threshold=None, batch_size: int = 32, num_workers: int = 1):
        super().__init__()
        self.pdb_dir = pdb_dir
        self.node_attr_columns = node_attr_columns
        self.edge_attr_columns = edge_attr_columns
        self.edge_kinds = edge_kinds
        self.sasa_threshold = sasa_threshold
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
        protein_pair_names = []
        with open(self.full_list_file, "r") as f:
            for line in f:
                pdb_id, chain_1, chain_2 = line.strip().split("_")
                protein_pair_name = (f"{pdb_id}_{chain_1}", f"{pdb_id}_{chain_2}")
                if protein_pair_name in protein_pair_names_to_labels:
                    protein_pair_names.append(protein_pair_name)
        protein_pair_names = protein_pair_names[:30]
        ProteinPairDataset(root=self.root, pdb_dir=self.pdb_dir, node_attr_columns=self.node_attr_columns, edge_attr_columns=self.edge_attr_columns,
                            edge_kinds=self.edge_kinds, sasa_threshold=self.sasa_threshold, 
                            protein_pair_names=protein_pair_names, label_mapping=protein_pair_names_to_labels, 
                            pre_transform=self.pre_transform, transform=self.transform)

    def setup(self, stage: str):
        protein_pair_names_to_labels = read_interface_labels(self.labels_file)
        if stage == "fit":
            protein_pair_names = []
            with open(self.train_file, "r") as f:
                for line in f:
                    pdb_id, chain_1, chain_2 = line.strip().split("_")
                    protein_pair_name = (f"{pdb_id}_{chain_1}", f"{pdb_id}_{chain_2}")
                    if protein_pair_name in protein_pair_names_to_labels:
                        protein_pair_names.append(protein_pair_name)
            protein_pair_names = protein_pair_names[:30]
            # split train and val
            self.train_protein_pair_names, self.val_protein_pair_names = train_test_split(protein_pair_names, test_size=0.2, random_state=42)
            self.train_dataset = ProteinPairDataset(root=self.root, pdb_dir=self.pdb_dir, 
                                                    node_attr_columns=self.node_attr_columns,
                                                    edge_attr_columns=self.edge_attr_columns,
                                                    edge_kinds=self.edge_kinds,
                                                    sasa_threshold=self.sasa_threshold,
                                                    protein_pair_names=self.train_protein_pair_names, 
                                                    label_mapping=protein_pair_names_to_labels, 
                                                    pre_transform=self.pre_transform, transform=self.transform)
            self.val_dataset = ProteinPairDataset(root=self.root, pdb_dir=self.pdb_dir, 
                                                  node_attr_columns=self.node_attr_columns,
                                                  edge_attr_columns=self.edge_attr_columns,
                                                  edge_kinds=self.edge_kinds,
                                                  sasa_threshold=self.sasa_threshold,
                                                  protein_pair_names=self.val_protein_pair_names, 
                                                  label_mapping=protein_pair_names_to_labels, 
                                                  pre_transform=self.pre_transform, transform=self.transform)
        elif stage == "test":
            self.test_protein_pair_names = []
            with open(self.test_file, "r") as f:
                for line in f:
                    pdb_id, chain_1, chain_2 = line.strip().split("_")
                    protein_pair_name = (f"{pdb_id}_{chain_1}", f"{pdb_id}_{chain_2}")
                    if protein_pair_name in protein_pair_names_to_labels:
                        self.test_protein_pair_names.append(protein_pair_name)
            self.test_dataset = ProteinPairDataset(root=self.root, pdb_dir=self.pdb_dir,
                                                   node_attr_columns=self.node_attr_columns,
                                                   edge_attr_columns=self.edge_attr_columns,
                                                   edge_kinds=self.edge_kinds,
                                                   sasa_threshold=self.sasa_threshold,
                                                   protein_pair_names=self.test_protein_pair_names,
                                                   label_mapping=protein_pair_names_to_labels, 
                                                   pre_transform=self.pre_transform, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
