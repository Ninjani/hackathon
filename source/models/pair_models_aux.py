import torch
from pytorch_lightning import LightningModule
from torch_geometric.nn import GATConv, HeteroConv
from torch import nn
import torch.nn.functional as F

# def pearson_loss(output, target):
#         vx = output - torch.mean(output)
#         vy = target - torch.mean(target)
#         return torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

class CrossGAT(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        """
        The heterograph contains a protein_1 graph and a protein_2 graph.
        in_channels: int
            The number of input channels.
        out_channels: int
            The number of output channels.
        num_layers: int
            The number of layers.
        """
        super().__init__()

        self.gat_convs = nn.ModuleList()
        for _ in range(num_layers):
            het_conv = HeteroConv({
                # ('node_type', 'edge_type', 'node_type'): MessagePassing
                ('protein_1', 'intra', 'protein_1'): GATConv(in_channels, in_channels, add_self_loops=False), # TODO: other kinds of convolutions
                ('protein_2', 'intra', 'protein_2'): GATConv(in_channels, in_channels, add_self_loops=False),
                ('protein_1', 'inter', 'protein_2'): GATConv((in_channels, in_channels), in_channels, add_self_loops=False),
            }, aggr='sum')
            self.gat_convs.append(het_conv)

        self.linear = nn.Linear(in_channels, out_channels)
       
        ### linear layer to predic residue distances
        ### the input is the concatenation of the residue embeddingsSS 
        self.linear_aux = nn.Sequential(nn.Linear(2*in_channels, in_channels),nn.Linear(in_channels, out_channels))

    def forward(self, x_dict, edge_index_dict):
        for conv in self.gat_convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        
        indices = edge_index_dict[('protein_1', 'interacts', 'protein_2')]
        x_dict_interacts = torch.hstack([x_dict['protein_1'][indices[0]], x_dict['protein_2'][indices[1]]]) 

        return (
            self.linear_aux(x_dict_interacts),
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
    

    def distance_loss(self, edge_attr_dict, distance_predictions):
        ### compute the loss for all pairs
        loss_function = torch.nn.MSELoss()
        loss =  loss_function(distance_predictions.squeeze(), edge_attr_dict["protein_1", "interacts", "protein_2"].squeeze())
        return loss

    def training_step(self, batch, batch_idx):
        distance_predictions, out_1, out_2 = self(batch.x_dict, batch.edge_index_dict)
        print(distance_predictions)
        print(batch.edge_attr_dict["protein_1", "interacts", "protein_2"].squeeze())
        aux_loss = self.distance_loss(batch.edge_attr_dict, distance_predictions)
        loss = F.binary_cross_entropy_with_logits(out_1, 
                                                  batch.y_dict["protein_1"].view(-1, 1)) + F.binary_cross_entropy_with_logits(out_2, 
                                                                                                                              batch.y_dict["protein_2"].view(-1, 1))
        batch_size = max(batch.x_dict['protein_1'].shape[0], batch.x_dict['protein_2'].shape[0])
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log('aux_loss', aux_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        return loss + aux_loss
    
    def validation_step(self, batch, batch_idx):
        distance_predictions, out_1, out_2 = self(batch.x_dict, batch.edge_index_dict)
        aux_loss = self.distance_loss(batch.edge_attr_dict, distance_predictions)
        loss = F.binary_cross_entropy_with_logits(out_1, 
                                                  batch.y_dict["protein_1"].view(-1, 1)) + F.binary_cross_entropy_with_logits(out_2, 
                                                                                                                              batch.y_dict["protein_2"].view(-1, 1))        
        batch_size = max(batch.x_dict['protein_1'].shape[0], batch.x_dict['protein_2'].shape[0])
        self.log('hp_metric', loss, batch_size=batch_size)
        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log('val_aux_loss', aux_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        
        return loss + aux_loss
        

        