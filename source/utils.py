from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.protein.features.nodes.amino_acid import amino_acid_one_hot
from graphein.protein.edges.distance import add_peptide_bonds
from graphein.ml import GraphFormatConvertor
import graphein
import torch
graphein.verbose(enabled=False)


def load_protein_as_graph(pdb_file, labels):
    """
    Loads a protein chain PDB file as a pytorch-geometric Data object using Graphein.
    In this case, 
        uses a CA-level graph (default in config)
        with peptide bonds as edges (data.edge_index) (default in config)
        the amino acid one-hot encoding and the B-factor as node features (data.x)
        interface residues are labelled as 1, non-interface residues are labelled as 0 (data.y)
        coordinates of the CA atoms are stored as node positions (data.pos)

    :param pdb_file: Path to PDB file
    :type pdb_file: str
    :param labels: Residue numbers of interface residues
    :type labels: set

    :return: Pytorch-geometric Data object
    """
    config = ProteinGraphConfig(**{"node_metadata_functions": [amino_acid_one_hot],
                                   "edge_construction_functions": [add_peptide_bonds]}) # TODO: explore other featurisations
    g = construct_graph(config=config, path=pdb_file, verbose=False)
    columns = [
                    "b_factor",
                    "chain_id",
                    "coords",
                    "edge_index",
                    "kind",
                    "name",
                    "node_id",
                    "residue_name",
                    "residue_number",
                    "amino_acid_one_hot",
                ]
    convertor = GraphFormatConvertor(src_format="nx", dst_format="pyg", columns=columns, verbose = None)
    data = convertor(g)
    data.x = torch.hstack([data.amino_acid_one_hot, 
                                data.b_factor.unsqueeze(1)]).float()
    data.pos = data.coords.float()
    data.y = torch.zeros(data.num_nodes)
    for i, res_num in enumerate(data.residue_number):
        if f"{data.chain_id[i]}_{res_num.item()}" in labels:
            data.y[i] = 1
    return data


def read_interface_labels(filename):
    """
    Read the interface residue numbers for each chain in a protein complex from the `interface_labels.txt` file.
    """
    protein_pair_names_to_labels = {}
    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 4:
                name_1, name_2, labels_1, labels_2 = parts
                protein_pair_names_to_labels[(name_1, name_2)] = (set(labels_1.split(",")), set(labels_2.split(",")))
    return protein_pair_names_to_labels