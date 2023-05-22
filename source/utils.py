from functools import partial
from pathlib import Path
import pickle
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.protein.features.sequence import compute_esm_embedding
from graphein.protein.features.sequence.utils import (
    subset_by_node_feature_value,
)
from graphein.protein.features.nodes.amino_acid import (
    amino_acid_one_hot,
    meiler_embedding,
    expasy_protein_scale,
    hydrogen_bond_acceptor,
    hydrogen_bond_donor,
)
from graphein.protein.edges.distance import (
    add_distance_threshold,
    add_peptide_bonds,
    add_hydrophobic_interactions,
    add_hydrogen_bond_interactions,
    add_disulfide_interactions,
    add_ionic_interactions,
    add_aromatic_interactions,
    add_aromatic_sulphur_interactions,
    add_cation_pi_interactions,
)
from graphein.ml import GraphFormatConvertor
import graphein
import networkx as nx
import numpy as np
import torch
import freesasa
graphein.verbose(enabled=False)

def esm_residue_embedding(
    G: nx.Graph,
    model_name: str = "esm1b_t33_650M_UR50S",
    output_layer: int = 33,
) -> nx.Graph:
    """
    Computes ESM residue embeddings from a protein sequence and adds the to the
    graph.

        *Biological Structure and Function Emerge from Scaling Unsupervised*
        *Learning to 250 Million Protein Sequences* (2019)
        Rives, Alexander and Meier, Joshua and Sercu, Tom and Goyal, Siddharth
        and Lin, Zeming and Liu, Jason and Guo,
        Demi and Ott, Myle and Zitnick, C. Lawrence and Ma, Jerry and Fergus,
        Rob


        *Transformer protein language models are unsupervised structure learners*
        (2020) Rao, Roshan M and Meier, Joshua and Sercu, Tom and Ovchinnikov,
        Sergey and Rives, Alexander

    **Pre-trained models**

    =========                     ====== ====== ======= ============= =========
    Full Name                     layers params Dataset Embedding Dim Model URL
    =========                     ====== ====== ======= ============= =========
    ESM-1b esm1b_t33_650M_UR50S   33    650M   UR50/S     1280        https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt
    ESM1-main esm1_t34_670M_UR50S 34    670M   UR50/S     1280        https://dl.fbaipublicfiles.com/fair-esm/models/esm1_t34_670M_UR50S.pt
    esm1_t34_670M_UR50D           34    670M   UR50/D     1280        https://dl.fbaipublicfiles.com/fair-esm/models/esm1_t34_670M_UR50D.pt
    esm1_t34_670M_UR100           34    670M   UR100      1280        https://dl.fbaipublicfiles.com/fair-esm/models/esm1_t34_670M_UR100.pt
    esm1_t12_85M_UR50S            12    85M    UR50/S     768         https://dl.fbaipublicfiles.com/fair-esm/models/esm1_t12_85M_UR50S.pt
    esm1_t6_43M_UR50S             6     43M    UR50/S     768         https://dl.fbaipublicfiles.com/fair-esm/models/esm1_t6_43M_UR50S.pt
    =========                     ====== ====== ======= ============= =========

    :param G: ``nx.Graph`` to add esm embedding to.
    :type G: nx.Graph
    :param model_name: Name of pre-trained model to use.
    :type model_name: str
    :param output_layer: index of output layer in pre-trained model.
    :type output_layer: int
    :return: ``nx.Graph`` with esm embedding feature added to nodes.
    :rtype: nx.Graph
    """

    for chain in G.graph["chain_ids"]:
        sequence = G.graph[f"sequence_{chain}"]
        embeddings = []
        for i in range(0, len(sequence), 1024):
            partial_embedding = compute_esm_embedding(
                sequence[i : i + 1022],
                representation="residue",
                model_name=model_name,
                output_layer=output_layer,
            )
            embeddings.append(partial_embedding[0, 1:-1])
        embedding = np.concatenate(embeddings, axis=0)
        assert len(embedding) == len(sequence)
        subgraph = subset_by_node_feature_value(G, "chain_id", chain)

        for i, (n, d) in enumerate(subgraph.nodes(data=True)):
            G.nodes[n]["esm_embedding"] = embedding[i]

    return G


def load_protein_as_graph(pdb_file, output_file=None):
    """
    Loads a protein chain PDB file as a Graphein networkx graph.

    :param pdb_file: Path to PDB file
    :type pdb_file: str

    :return: Networkx graph
    """
    config = ProteinGraphConfig(
        node_metadata_functions= [
            amino_acid_one_hot,
            meiler_embedding,
            expasy_protein_scale,
            hydrogen_bond_acceptor,
            hydrogen_bond_donor,
        ],
        edge_construction_functions=[       # List of functions to call to construct edges.
            add_peptide_bonds,
            add_hydrophobic_interactions,
            add_aromatic_interactions,
            add_disulfide_interactions,
            add_ionic_interactions,
            add_hydrogen_bond_interactions,
            add_aromatic_sulphur_interactions,
            add_cation_pi_interactions,
            partial(add_distance_threshold, long_interaction_threshold=5, threshold=10.),
        ],
        graph_metadata_functions=[
                             esm_residue_embedding,
                             ]
    )

    graph = construct_graph(config=config, path=pdb_file, verbose=False)
    
    #Add SASA as a node feature using freesasa
    structure = freesasa.Structure(str(pdb_file))
    result = freesasa.calc(structure)

    for node, data in graph.nodes(data=True):
        chain_id = data['chain_id']
        residue_number = str(data['residue_number'])
        
        # Calculate the SASA for each node/residue
        try:
            sasa = result.residueAreas()[chain_id][residue_number].total
            rel_sasa = result.residueAreas()[chain_id][residue_number].relativeTotal
        except KeyError:
            sasa = 0
            rel_sasa = 0
        
        # Add SASA as a node feature
        data['sasa'] = sasa
        data['rel_sasa'] = rel_sasa
    if output_file is not None:
        with open(output_file, "wb") as f:
            pickle.dump(graph, f)
    return graph

def graphein_to_pytorch_graph(graphein_graph, node_attr_columns: list, edge_attr_columns: list, edge_kinds: set, labels):
    """
    Converts a Graphein graph to a pytorch-geometric Data object.
    """
    if edge_attr_columns is None:
        edge_attr_columns = []
    columns = [
            "chain_id",
            "coords",
            "edge_index",
            "kind",
            "node_id",
            "residue_number",
        ] + node_attr_columns + edge_attr_columns
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
        if len(kind.intersection(edge_kinds)) > 0:
            edge_index.append([data.edge_index[0, i], data.edge_index[1, i]])
            if len(edge_attr_columns) > 0:
                edge_attr.append(data.edge_attr[i])
    data.edge_index = torch.tensor(edge_index).T.long()
    if len(edge_attr_columns) > 0:
        data.edge_attr = torch.tensor(edge_attr).float()
    data.pos = data.coords.float()
    data.y = torch.zeros(data.num_nodes)
    for i, res_num in enumerate(data.residue_number):
        if f"{data.chain_id[i]}_{res_num.item()}" in labels:
            data.y[i] = 1
    return data


def read_interface_labels(filename):
    """
    Read the interface residue numbers for each chain in a protein complex from the `interface_labels.txt` or `non_interface_labels.txt` file.
    """
    protein_pair_names_to_labels = {}
    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 4:
                name_1, name_2, labels_1, labels_2 = parts
                protein_pair_names_to_labels[(name_1, name_2)] = (set(labels_1.split(",")), set(labels_2.split(",")))

            elif len(parts) == 2:
                name_1, name_2 = parts
                protein_pair_names_to_labels[(name_1, name_2)] = (set([]), set([]))
    return protein_pair_names_to_labels


def get_protein_pair_names(names_file, protein_pair_names_to_labels, protein_pair_names_to_non_labels=None):  
    protein_pair_names = []
    # only for labeled data
    with open(names_file, "r") as f:
        for line in f:
            pdb_id, chain_1, chain_2 = line.strip().split("_")
            protein_pair_name = (f"{pdb_id}_{chain_1}", f"{pdb_id}_{chain_2}")
            if protein_pair_name in protein_pair_names_to_labels:
                protein_pair_names.append(protein_pair_name)
    if protein_pair_names_to_non_labels is None:
        return protein_pair_names
    non_label_dict = {key[0]:key[1] for key in list(protein_pair_names_to_non_labels.keys())}
    # I want to use the same proteins as in my labeled set, but with another protein that i receive from the dict above
    protein_pair_names_non_label = [(prot[0],non_label_dict[prot[0]]) for prot in protein_pair_names]
    length_per_set = min(len(protein_pair_names_non_label), len(protein_pair_names))
    protein_pair_names_non_label = protein_pair_names_non_label[:length_per_set]
    protein_pair_names = protein_pair_names[:length_per_set]
    return protein_pair_names + protein_pair_names_non_label


def get_protein_names(names_file, protein_pair_names_to_labels):
    protein_names_to_labels = {}
    for p1, p2 in protein_pair_names_to_labels:
        protein_names_to_labels[p1] = protein_pair_names_to_labels[(p1, p2)][0]
        protein_names_to_labels[p2] = protein_pair_names_to_labels[(p1, p2)][1]
    protein_names = set()
    with open(names_file, "r") as f:
        for line in f:
            pdb_id, chainA, _ = line.strip().split("_")
            name = f"{pdb_id}_{chainA}"
            if name in protein_names_to_labels:
                protein_names.add(name)
    protein_names = list(protein_names)
    return protein_names, protein_names_to_labels