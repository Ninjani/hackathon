from pytorch_lightning import LightningModule
import torch
from torch_geometric.nn import GAT
import torch.nn.functional as F


class GATModel(LightningModule):
    """
    LightningModule wrapping a GAT model.
    """
    def __init__(self, in_channels, hidden_channels, num_layers, num_heads, out_channels, dropout, jk):
        super(GATModel, self).__init__()
        self.save_hyperparameters()
        self.example_input_array = (torch.empty(32, in_channels), torch.zeros(2, 100, dtype=torch.long))
        self.model = GAT(in_channels=in_channels,
                         hidden_channels=hidden_channels,
                         num_layers=num_layers,
                         heads=num_heads,
                         out_channels=out_channels,
                         dropout=dropout,
                         jk=jk, v2=True) # TODO: try other models

    def forward(self, node_attributes, edge_index):
        return self.model(node_attributes, edge_index)
    
    def training_step(self, batch, batch_idx):
        out = self(batch.x, batch.edge_index)
        loss = F.binary_cross_entropy_with_logits(out, batch.y.view(-1, 1))
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True,
                 batch_size=batch.x.shape[0])
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        out = self(batch.x, batch.edge_index)
        loss = F.binary_cross_entropy_with_logits(out, batch.y.view(-1, 1))
        self.log('hp_metric', loss, on_step=True, on_epoch=True, sync_dist=True,
                 batch_size=batch.x.shape[0])
        return loss