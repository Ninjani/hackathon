from pytorch_lightning import LightningModule
from torch_geometric.nn import GATConv, HeteroConv
from torch import nn
import torch.nn.functional as F

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

    def forward(self, x_dict, edge_index_dict):
        for conv in self.gat_convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}

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
        out_1, out_2 = self(batch.x_dict, batch.edge_index_dict)
        loss = F.binary_cross_entropy_with_logits(out_1, 
                                                  batch.y_dict["protein_1"].view(-1, 1)) + F.binary_cross_entropy_with_logits(out_2, 
                                                                                                                              batch.y_dict["protein_2"].view(-1, 1))
        batch_size = max(batch.x_dict['protein_1'].shape[0], batch.x_dict['protein_2'].shape[0])
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        out_1, out_2 = self(batch.x_dict, batch.edge_index_dict)
        loss = F.binary_cross_entropy_with_logits(out_1, 
                                                  batch.y_dict["protein_1"].view(-1, 1)) + F.binary_cross_entropy_with_logits(out_2, 
                                                                                                                              batch.y_dict["protein_2"].view(-1, 1))        
        batch_size = max(batch.x_dict['protein_1'].shape[0], batch.x_dict['protein_2'].shape[0])
        self.log('hp_metric', loss, batch_size=batch_size)
        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        return loss
        