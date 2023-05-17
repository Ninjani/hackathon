from collections import defaultdict
from pytorch_lightning import LightningModule
import torch
from torch_geometric.nn import GATConv, HeteroConv
from torch import nn
import torch.nn.functional as F
from egnn_pytorch import EGNN_Sparse
from torch_geometric.nn.conv.hetero_conv import group


class HeteroEGNN(HeteroConv):
    def __init__(self, convs, aggr='add', node_dim: int = 12):
        super().__init__(convs, aggr)
        self.node_dim = node_dim

    def forward(
        self,
        x_dict,
        pos_dict,
        edge_index_dict,
        *args_dict,
        **kwargs_dict,
    ):
        out_dict = defaultdict(list)
        for edge_type, edge_index in edge_index_dict.items():
            src, rel, dst = edge_type

            str_edge_type = '__'.join(edge_type)
            if str_edge_type not in self.convs:
                continue

            args = []
            for value_dict in args_dict:
                if edge_type in value_dict:
                    args.append(value_dict[edge_type])
                elif src == dst and src in value_dict:
                    args.append(value_dict[src])
                elif src in value_dict or dst in value_dict:
                    args.append(
                        (value_dict.get(src, None), value_dict.get(dst, None)))

            kwargs = {}
            for arg, value_dict in kwargs_dict.items():
                arg = arg[:-5]  # `{*}_dict`
                if edge_type in value_dict:
                    kwargs[arg] = value_dict[edge_type]
                elif src == dst and src in value_dict:
                    kwargs[arg] = value_dict[src]
                elif src in value_dict or dst in value_dict:
                    kwargs[arg] = (value_dict.get(src, None),
                                   value_dict.get(dst, None))

            conv = self.convs[str_edge_type]

            if self.node_dim == x_dict[src].shape[1]:
                out = conv(torch.hstack((pos_dict[src], x_dict[src])), edge_index, *args, **kwargs)
            else:
                out = conv(x_dict[src] , edge_index, *args, **kwargs)

            out_dict[dst].append(out)

        for key, value in out_dict.items():
            out_dict[key] = group(value, self.aggr)

        return out_dict


class CrossGAT(nn.Module):
    def __init__(self, hidden_channels, out_channels, num_gat_layers, num_inter_layers, num_egnn_layers, dropout):
        """
        The heterograph contains a protein_1 graph and a protein_2 graph.
        in_channels: int
            The number of input channels.
        out_channels: int
            The number of output channels.
        num_gat_layers: int
            The number of GAT layers in the same protein
        num_inter_layers: int
            The number of GAT layers between two proteins
        num_egnn_layers: int
            The number of EGNN layers in the same protein
        """
        super().__init__()

        self.gat_convs = nn.ModuleList()
        self.num_gat_layers = num_gat_layers
        self.num_egnn_layers = num_egnn_layers
        self.num_inter_layers = num_inter_layers
        assert self.num_gat_layers > 0 or self.num_egnn_layers == 0, "Must have at least one GAT layer if EGNN is used so that the input dimension is correct"
        for _ in range(num_gat_layers):
            het_conv = HeteroConv({
                # ('node_type', 'edge_type', 'node_type'): MessagePassing
                ('protein_1', 'intra', 'protein_1'): GATConv(-1, hidden_channels, dropout=dropout),
                ('protein_2', 'intra', 'protein_2'): GATConv(-1, hidden_channels, dropout=dropout),
            }, aggr='sum')
            self.gat_convs.append(het_conv)
        for _ in range(num_inter_layers):
            het_conv = HeteroConv({
                # ('node_type', 'edge_type', 'node_type'): MessagePassing
                ('protein_1', 'inter', 'protein_2'): GATConv((-1, -1), hidden_channels, add_self_loops=False, dropout=dropout),
                ('protein_2', 'inter', 'protein_1'): GATConv((-1, -1), hidden_channels, add_self_loops=False, dropout=dropout),
            }, aggr='sum')
            self.gat_convs.append(het_conv)
        self.egnn_convs = nn.ModuleList()
        for _ in range(num_egnn_layers):
            het_conv = HeteroEGNN({
                # ('node_type', 'edge_type', 'node_type'): MessagePassing
                ('protein_1', 'intra', 'protein_1'): EGNN_Sparse(hidden_channels, dropout=dropout),
                ('protein_2', 'intra', 'protein_2'): EGNN_Sparse(hidden_channels, dropout=dropout),
            }, aggr='sum', node_dim=hidden_channels)
            self.egnn_convs.append(het_conv)

        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, pos_dict, edge_index_dict):
        for conv in self.gat_convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        if self.num_egnn_layers > 0:
            for conv in self.egnn_convs:
                x_dict = conv(x_dict, pos_dict, edge_index_dict)
            x_dict = {key: x[:, 3:] for key, x in x_dict.items()}

        return (
            self.linear(x_dict['protein_1']), 
            self.linear(x_dict['protein_2'])
        )

    
class CrossGATModel(LightningModule):
    def __init__(self, in_channels, num_layers, out_channels):
        super(CrossGATModel, self).__init__()
        self.save_hyperparameters()
        self.model = CrossGAT(in_channels=in_channels,
                         out_channels=out_channels,
                         num_layers=num_layers)
        
    def forward(self, x_dict, edge_index_dict):
        return self.model(x_dict, edge_index_dict)
    
    def training_step(self, batch, batch_idx):
        out_1, out_2 = self(batch.x_dict, batch.pos_dict, batch.edge_index_dict)
        loss = F.binary_cross_entropy_with_logits(out_1, 
                                                  batch.y_dict["protein_1"].view(-1, 1)) + F.binary_cross_entropy_with_logits(out_2, 
                                                                                                                              batch.y_dict["protein_2"].view(-1, 1))
        batch_size = max(batch.x_dict['protein_1'].shape[0], batch.x_dict['protein_2'].shape[0])
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        out_1, out_2 = self(batch.x_dict, batch.pos_dict, batch.edge_index_dict)
        loss = F.binary_cross_entropy_with_logits(out_1, 
                                                  batch.y_dict["protein_1"].view(-1, 1)) + F.binary_cross_entropy_with_logits(out_2, 
                                                                                                                              batch.y_dict["protein_2"].view(-1, 1))        
        batch_size = max(batch.x_dict['protein_1'].shape[0], batch.x_dict['protein_2'].shape[0])
        self.log('hp_metric', loss, batch_size=batch_size)
        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        return loss