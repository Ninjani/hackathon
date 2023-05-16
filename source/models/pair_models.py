from pytorch_lightning import LightningModule
import torch
from torch_geometric.nn import GATConv, HeteroConv
from torch import nn
import torch.nn.functional as F

class CrossGAT(LightningModule):
    def __init__(self, in_channels, out_channels, num_layers=1):
        """
        The heterograph contains a protein_1 graph and a protein_2 graph.
        in_channels: int
            The number of input channels.
        out_channels: int
            The number of output channels.
        num_layers: int
            The number of layers.
        """
        super(CrossGAT, self).__init__()
        self.save_hyperparameters()
        self.example_input_array = (torch.empty(32, in_channels), torch.zeros(2, 100, dtype=torch.long))

        self.gat_convs = nn.ModuleList()
        for _ in range(num_layers):
            het_conv = HeteroConv({
                # ('node_type', 'edge_type', 'node_type'): MessagePassing
                ('protein_1', 'intra', 'protein_1'): GATConv(in_channels, in_channels, add_self_loops=False), # TODO: other kinds of convolutions
                ('protein_2', 'intra', 'protein_2'): GATConv(in_channels, in_channels, add_self_loops=False),
                ('protein_1', 'inter', 'protein_2'): GATConv((in_channels, in_channels), in_channels, add_self_loops=False),
            }, aggr='sum')
            self.gat_convs.append(het_conv)

        print(in_channels)
        print(out_channels)

        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, node_attributes, edge_index):
        for conv in self.gat_convs:
            node_attributes = conv(node_attributes, edge_index)
            node_attributes = {key: x.relu() for key, x in node_attributes.items()}

        return (
            self.linear(node_attributes['protein_1']), 
            self.linear(node_attributes['protein_2'])
        )
    
    def training_step(self, batch, batch_idx):
        out1, out2 = self(batch.x, batch.edge_index)
        loss1 = F.binary_cross_entropy_with_logits(out1, batch.y.view(-1, 1))
        loss2 = F.binary_cross_entropy_with_logits(out2, batch.y.view(-1, 1))
        loss = loss1 + loss2
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True,
                 batch_size=batch.x.shape[0])

        return loss
    
    def validation_step(self, batch, batch_idx):
        out1, out2 = self(batch.x, batch.edge_index)
        loss1 = F.binary_cross_entropy_with_logits(out1, batch.y.view(-1, 1))
        loss2 = F.binary_cross_entropy_with_logits(out2, batch.y.view(-1, 1))
        loss = loss1 + loss2
        self.log('hp_metric', loss, on_step=True, on_epoch=True, sync_dist=True,
                 batch_size=batch.x.shape[0])

        return loss
