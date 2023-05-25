from pathlib import Path
import pickle
from torch_geometric.data import Dataset, Data, HeteroData
from torch_geometric.loader import DataLoader
from pytorch_lightning import LightningDataModule
import torch
from tqdm import tqdm
from ..utils import get_protein_pair_names, graphein_to_pytorch_graph, load_protein_as_graph, read_interface_labels

def make_hetero_graph(graph_1: Data, graph_2: Data, use_surface=False):
    """
    Combine two pytorch-geometric Data objects into a single HeteroData object.
    i.e data_1.x and data_2.x become data["protein_1"].x and data["protein_2"].x
    Also add all vs. all edges between the two proteins.
    """    
    graph = HeteroData()
    graph["protein_1"].x = graph_1.x
    graph["protein_1"].y = graph_1.y
    graph["protein_1"].pos = graph_1.pos
    graph["protein_1", "intra", "protein_1"].edge_index = graph_1.edge_index
    graph["protein_1", "intra", "protein_1"].edge_attr = graph_1.edge_attr

    graph["protein_2"].x = graph_2.x
    graph["protein_2"].y = graph_2.y
    graph["protein_2"].pos = graph_2.pos
    graph["protein_2", "intra", "protein_2"].edge_index = graph_2.edge_index
    graph["protein_2", "intra", "protein_2"].edge_attr = graph_2.edge_attr

    # add edges between interface residues and their distances
    interacting_edge_index = []
    interacting_edge_distance = []
    for i in range(graph_1.x.shape[0]):
        for j in range(graph_2.x.shape[0]):
            if graph_1.y[i] == graph_2.y[j] == 1:
                interacting_edge_index.append([i, j])
                # Euclidean distance between the pos vectors of the two nodes
                interacting_edge_distance.append(torch.norm(graph_1.pos[i] - graph_2.pos[j]))
    if len(interacting_edge_index) > 0:
        graph["protein_1", "interacts", "protein_2"].edge_index = torch.tensor(interacting_edge_index).T
        graph["protein_1", "interacts", "protein_2"].edge_attr = torch.tensor(interacting_edge_distance).unsqueeze(1)

    # add all vs. all edges between the two proteins (or those with rel_sasa > 0.20 if use_surface=True)
    edge_index_inter = []
    for i in range(graph_1.x.shape[0]):
        for j in range(graph_2.x.shape[0]):
            if not use_surface or (graph_1.rel_sasa[i] > 0.20 and graph_2.rel_sasa[j] > 0.20):
                edge_index_inter.append([i, j])
    assert len(edge_index_inter) > 0, "No residues with rel_sasa > 0.20 found in one or both proteins"
    graph["protein_1", "inter", "protein_2"].edge_index = torch.tensor(edge_index_inter).T
    graph["protein_2", "inter", "protein_1"].edge_index = torch.tensor(edge_index_inter).T.flip(0)
    return graph

class ProteinPairDataset(Dataset):
    """
    torch-geometric Dataset class for loading pairs of protein files as HeteroData objects.
    """
    def __init__(self, root, pdb_dir, pkl_dir, processed_dir_suffix: str,
                 node_attr_columns: list, edge_attr_columns: list, 
                 edge_kinds: set, use_surface: bool, 
                 protein_pair_names: list, label_mapping: dict, 
                 pre_transform=None, transform=None):
        self.pdb_dir = Path(pdb_dir)
        self.pkl_dir = Path(pkl_dir)
        self.processed_dir_suffix = processed_dir_suffix
        self.node_attr_columns = node_attr_columns
        self.edge_attr_columns = edge_attr_columns
        self.edge_kinds = edge_kinds
        self.use_surface = use_surface
        self.protein_pair_names = protein_pair_names
        self.label_mapping = label_mapping
        self.protein_names = set()
        for name_1, name_2 in self.protein_pair_names:
            self.protein_names.add(name_1)
            self.protein_names.add(name_2)
        super(ProteinPairDataset, self).__init__(root, pre_transform=pre_transform, transform=transform)
    
    @property
    def processed_dir(self) -> str:
        return str(Path(self.root) / f"processed_{self.processed_dir_suffix}")    

    @property
    def raw_dir(self) -> str:
        return str(self.pkl_dir)

    @property
    def raw_file_names(self):
        return [Path(self.raw_dir) / f"{protein_name}.pkl" for protein_name in self.protein_names]
    
    def download(self):
        print(f"Downloading {len(self.protein_names)} proteins...")
        for i, protein_name in enumerate(self.protein_names):
            
            try:
                if i > 0 and i % 100 == 0:
                    print(f"{(i/len(self.protein_names)*100):.1f}% complete")
                output_file = Path(self.raw_dir) / f"{protein_name}.pkl"
                if not output_file.exists():
                    load_protein_as_graph(self.pdb_dir / f"{protein_name}.pdb", output_file)
            except:
                print(f'some problem with {protein_name}')
            
            

    @property
    def processed_file_names(self):
        return [Path(self.processed_dir) / f"{p1}__{p2}.pt" for p1, p2 in self.protein_pair_names]

    def process(self):
        for p1, p2 in self.protein_pair_names:
            output = Path(self.processed_dir) / f'{p1}__{p2}.pt'
            if output.exists():
                continue
            
            try:
                with open(Path(self.raw_dir) / f"{p1}.pkl", "rb") as f:
                    data_1 = pickle.load(f)     
                with open(Path(self.raw_dir) / f"{p2}.pkl", "rb") as f:
                    data_2 = pickle.load(f)
                data_1 = graphein_to_pytorch_graph(data_1, self.node_attr_columns, self.edge_attr_columns, self.edge_kinds, self.label_mapping[(p1, p2)][0])
                data_2 = graphein_to_pytorch_graph(data_2, self.node_attr_columns, self.edge_attr_columns, self.edge_kinds, self.label_mapping[(p1, p2)][1])
                if self.pre_transform is not None:
                    data_1 = self.pre_transform(data_1)
                    data_2 = self.pre_transform(data_2)
                data = make_hetero_graph(data_1, data_2, self.use_surface)
                torch.save(data, output)
            except Exception as e:
                print(f'Some problem with protein pairs {p1} and {p2}. Skipping... ')
                print(e)
                
    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        data = torch.load(self.processed_file_names[idx])
        if not hasattr(data["protein_1", "interacts", "protein_2"], 'edge_index') or len(data["protein_1", "interacts", "protein_2"].edge_index) == 0:
            del data["protein_1", "interacts", "protein_2"]
        return data
    

class ProteinPairDataModule(LightningDataModule):
    """
    Pytorch Lightning DataModule wrapping the ProteinPairDataset class.
    Here you set/choose
    - how training, validation and test data are split
    - the batch size and number of workers for the DataLoader
    - the transforms to apply to the individual graphs (in pre-transform) and to the combined HeteroData object (in transform)
    """
    def __init__(self, root, pdb_dir, pkl_dir, processed_dir_suffix, node_attr_columns, edge_attr_columns, edge_kinds, use_surface=False, batch_size: int = 32, num_workers: int = 1):
        super().__init__()
        self.pdb_dir = pdb_dir
        self.pkl_dir = pkl_dir
        self.processed_dir_suffix = processed_dir_suffix
        self.node_attr_columns = node_attr_columns
        self.edge_attr_columns = edge_attr_columns
        self.edge_kinds = edge_kinds
        self.use_surface = use_surface
        self.train_file = Path(root) / "training.txt"
        self.val_file = Path(root) / "testing.txt"
        self.test_file = Path(root) / "testing.txt"
        self.full_list_file = Path(root) / "full_list.txt"
        self.labels_file = Path(root) / "interface_labels.txt"
        self.non_labels_file = Path(root) / "non-interface_labels.txt"
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pre_transform = None
        self.transform = None

    def _create_dataset(self, names_file, use_non_interacting=False):
        protein_pair_names_to_labels = read_interface_labels(self.labels_file)
        if use_non_interacting:
            protein_pair_names_to_non_labels = read_interface_labels(self.non_labels_file)
            protein_pair_names_to_labels_combi = {**protein_pair_names_to_labels, **protein_pair_names_to_non_labels}
        else:
            protein_pair_names_to_non_labels = None
            protein_pair_names_to_labels_combi = protein_pair_names_to_labels
        protein_pair_names = get_protein_pair_names(names_file, 
                                                    protein_pair_names_to_labels=protein_pair_names_to_labels, 
                                                    protein_pair_names_to_non_labels=protein_pair_names_to_non_labels)
        protein_pair_names = [p for p in protein_pair_names if (Path(self.pkl_dir) / f"{p[0]}.pkl").exists() and (Path(self.pkl_dir) / f"{p[1]}.pkl").exists()]
        return ProteinPairDataset(root=self.root, pdb_dir=self.pdb_dir, pkl_dir=self.pkl_dir,
                                  processed_dir_suffix=self.processed_dir_suffix,
                                  node_attr_columns=self.node_attr_columns, edge_attr_columns=self.edge_attr_columns, edge_kinds=self.edge_kinds,
                                  protein_pair_names=protein_pair_names, label_mapping=protein_pair_names_to_labels_combi, use_surface=self.use_surface,
                                  pre_transform=self.pre_transform, transform=self.transform)
    
    def prepare_data(self):
        self._create_dataset(self.full_list_file, use_non_interacting=True)

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = self._create_dataset(self.train_file, use_non_interacting=True)
            self.val_dataset = self._create_dataset(self.val_file)
            print(f"Number of training pairs: {len(self.train_dataset)}")
            print(f"Number of validation pairs: {len(self.val_dataset)}")
        elif stage == "test":
            self.test_dataset = self._create_dataset(self.test_file)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
