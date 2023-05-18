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
    def __init__(self, hidden_channels, out_channels, num_gat_layers, num_inter_layers, num_egnn_layers, 
                 auxiliary_loss_weight: float, dropout):
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
        auxiliary_loss_weight: float
            The weight of auxiliary loss (no auxiliary loss if set to 0)
        dropout: float
            Dropout rate
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
        self.use_auxiliary_head = auxiliary_loss_weight > 0
        if self.use_auxiliary_head:
            self.auxiliary_loss_weight = auxiliary_loss_weight
            ### linear layer to predic residue distances
            ### the input is the concatenation of the residue embeddingsSS 
            self.linear_aux = nn.Sequential(nn.Linear(2*hidden_channels, hidden_channels),
                                            nn.ReLU(),
                                            nn.Linear(hidden_channels, out_channels),
                                            nn.ReLU())
            self.auxiliary_loss = nn.MSELoss()
        self.validation_step_outputs = []
        self.loss = nn.BCEWithLogitsLoss()


    def forward(self, x_dict, pos_dict, edge_index_dict):
        for conv in self.gat_convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        if self.num_egnn_layers > 0:
            for conv in self.egnn_convs:
                x_dict = conv(x_dict, pos_dict, edge_index_dict)
            x_dict = {key: x[:, 3:] for key, x in x_dict.items()}
        if self.auxiliary_loss:
            indices = edge_index_dict[('protein_1', 'interacts', 'protein_2')]
            x_dict_interacts = torch.hstack([x_dict['protein_1'][indices[0]], x_dict['protein_2'][indices[1]]]) 
            distance_predictions = self.linear_aux(x_dict_interacts)
        else:
            distance_predictions = None
        return (
            self.linear(x_dict['protein_1']), 
            self.linear(x_dict['protein_2']),
            distance_predictions
        )
    
    def calculate_loss(self, predicted_interface_1, predicted_interface_2, predicted_distances, true_interface_1, true_interface_2, true_distances):
        loss = self.loss(predicted_interface_1, true_interface_1) + self.loss(predicted_interface_2, true_interface_2)
        if self.use_auxiliary_head:
            loss = (1 - self.auxiliary_loss_weight) * loss + self.auxiliary_loss_weight * self.auxiliary_loss(predicted_distances.squeeze(), true_distances.squeeze())
        return loss


    
class CrossGATModel(LightningModule):
    def __init__(self, hidden_channels, num_gat_layers, num_inter_layers, num_egnn_layers, out_channels, auxiliary_loss_weight, dropout):
        super(CrossGATModel, self).__init__()
        self.save_hyperparameters()
        self.model = CrossGAT(hidden_channels=hidden_channels,
                            num_gat_layers=num_gat_layers,
                            num_inter_layers=num_inter_layers,
                            num_egnn_layers=num_egnn_layers,
                            auxiliary_loss_weight=auxiliary_loss_weight,
                            dropout=dropout,
                            out_channels=out_channels)
        self.validation_step_outputs = []
        
    def forward(self, x_dict, pos_dict, edge_index_dict):
        return self.model(x_dict, pos_dict, edge_index_dict)
    
    def get_loss(self, batch, out_1, out_2, distance_predictions):
        loss = self.model.calculate_loss(out_1, out_2, distance_predictions, 
                                         batch.y_dict["protein_1"].view(-1, 1), 
                                         batch.y_dict["protein_2"].view(-1, 1), 
                                         batch.edge_attr_dict["protein_1", "interacts", "protein_2"])
        return loss
    
    def training_step(self, batch, batch_idx):
        out_1, out_2, distance_predictions = self(batch.x_dict, batch.pos_dict, batch.edge_index_dict)
        loss = self.get_loss(batch, out_1, out_2, distance_predictions)
        batch_size = max(batch.x_dict['protein_1'].shape[0], batch.x_dict['protein_2'].shape[0])
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        out_1, out_2, distance_predictions = self(batch.x_dict, batch.pos_dict, batch.edge_index_dict)
        loss = self.get_loss(batch, out_1, out_2, distance_predictions)
        batch_size = max(batch.x_dict['protein_1'].shape[0], batch.x_dict['protein_2'].shape[0])
        self.log('hp_metric', loss, batch_size=batch_size)
        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        validation_step_output = dict(loss=loss,
                                        y=torch.cat([batch.y_dict["protein_1"].view(-1, 1), batch.y_dict["protein_2"].view(-1, 1)], dim=0),
                                        out= torch.cat([out_1, out_2], dim=0))
        if distance_predictions is not None:
            validation_step_output["predicted_distances"] = distance_predictions.squeeze()
            validation_step_output["distances"] = batch.edge_attr_dict["protein_1", "interacts", "protein_2"].squeeze()
        self.validation_step_outputs.append(validation_step_output)
        return loss
