from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.protein.features.nodes.amino_acid import amino_acid_one_hot
from graphein.protein.edges.distance import add_peptide_bonds
from graphein.ml import GraphFormatConvertor
import graphein
import torch
graphein.verbose(enabled=False)


def load_protein_as_graph(pdb_file):
    """
    Loads a protein chain PDB file as a Graphein networkx graph.

    :param pdb_file: Path to PDB file
    :type pdb_file: str

    :return: Networkx graph
    """
    config = ProteinGraphConfig(**{"node_metadata_functions": [amino_acid_one_hot],
                                   "edge_construction_functions": [add_peptide_bonds]}) # TODO: explore other featurisations
    graph = construct_graph(config=config, path=pdb_file, verbose=False)
    return graph


def graphein_to_pytorch_graph(graphein_graph, node_attr_columns: list, edge_attr_columns: list, edge_kinds: set, labels):
    """
    Converts a Graphein graph to a pytorch-geometric Data object.
    """
    if edge_attr_columns is None:
        edge_attr_columns = []
    columns = node_attr_columns + edge_attr_columns + ["edge_index", "kind", "coords", "chain_id", "node_id", "residue_number"]
    convertor = GraphFormatConvertor(src_format="nx", dst_format="pyg", columns=columns, verbose = None)
    data = convertor(graphein_graph)
    data_dict= data.to_dict()
    x_data = []
    for x in node_attr_columns:
        if data_dict[x].ndim == 1:
            x_data.append(torch.atleast_2d(data_dict[x]).T)
        else:
            x_data.append(torch.atleast_2d(data_dict[x]))
    data.x = torch.hstack(x_data).float()
    if len(edge_attr_columns) > 0:
        edge_attr_data = []
        for x in edge_attr_columns:
            if data_dict[x].ndim == 1:
                edge_attr_data.append(torch.atleast_2d(data_dict[x]).T)
            else:
                edge_attr_data.append(torch.atleast_2d(data_dict[x]))
        data.edge_attr = torch.hstack(edge_attr_data).float()
    # prune edge_index and edge_attr based on edge_kinds
    edge_index = []
    edge_attr = []
    for i, kind in enumerate(data_dict["kind"]):
        if kind in edge_kinds:
            edge_index.append(data.edge_index[:, i])
            if len(edge_attr_columns) > 0:
                edge_attr.append(data.edge_attr[i])
    data.edge_index = torch.tensor(edge_index).T.long()
    data.edge_attr = torch.tensor(edge_attr).float()
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