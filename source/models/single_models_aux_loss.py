from pytorch_lightning import LightningModule
import torch
from torch_geometric.nn import GAT
import torch.nn.functional as F
from torch import nn


class GATModel(LightningModule):
    """
    LightningModule wrapping a GAT model.
    """
    def __init__(self, in_channels, hidden_channels, num_layers, num_heads, out_channels, dropout, jk):
        super(GATModel, self).__init__()
        self.save_hyperparameters()
        #self.example_input_array = (torch.empty(32, in_channels), torch.zeros(2, 100, dtype=torch.long))
        self.model = GAT(in_channels=in_channels,
                         hidden_channels=hidden_channels,
                         num_layers=num_layers,
                         heads=num_heads,
                         out_channels=None,
                         dropout=dropout,
                         jk=jk, v2=True) # TODO: try other models

        ### add a layer for the auxiliary classification and 1 for the main output
        out_channels_aux = out_channels
        self.linear_aux = nn.Linear(hidden_channels, out_channels_aux)
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, node_attributes, edge_index):
        node_attributes = self.model(node_attributes, edge_index)
        return (
            self.linear_aux(node_attributes), 
            self.linear(node_attributes)
            )
    

    def compute_loss(self, batch, aux_out, out, weights = [0.5,0.5]):
        loss_aux = F.binary_cross_entropy_with_logits(aux_out, batch.y.view(-1, 1))
        loss = F.binary_cross_entropy_with_logits(out, batch.y.view(-1, 1))

        return weights[0]*loss_aux, weights[1]*loss


    ### do we need an y_aux for the data features
    def training_step(self, batch, batch_idx):
        aux_out, out = self(batch.x, batch.edge_index)
        loss_aux, loss = self.compute_loss(batch, aux_out, out)

        self.log('train_aux_loss', loss_aux, on_step=True, on_epoch=True, sync_dist=True,
                 batch_size=batch.x.shape[0])
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True,
                 batch_size=batch.x.shape[0])
        
        return loss_aux + loss
    
    def validation_step(self, batch, batch_idx):
        aux_out, out = self(batch.x, batch.edge_index)
        #loss = F.binary_cross_entropy_with_logits(out, batch.y.view(-1, 1))
        loss_aux, loss = self.compute_loss(batch, aux_out, out)
        
        self.log('hp_aux_metric', loss_aux, on_step=True, on_epoch=True, sync_dist=True,
                 batch_size=batch.x.shape[0])
        self.log('hp_metric', loss, on_step=True, on_epoch=True, sync_dist=True,
                 batch_size=batch.x.shape[0])
        
        return loss + loss_aux