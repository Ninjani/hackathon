from pytorch_lightning import LightningModule
import torch
from torch import nn
from torch_geometric.nn import GAT
import torch.nn.functional as F
from egnn_pytorch import EGNN_Sparse


class EGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()
        self.egnn_convs = nn.ModuleList()
        for _ in range(num_layers):
            self.egnn_convs.append(EGNN_Sparse(feats_dim=in_channels, 
                                               m_dim=hidden_channels, 
                                               dropout=dropout))
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, pos, edge_index):
        for conv in self.egnn_convs:
            x = torch.cat([pos, x], dim=-1)
            x = conv(x, edge_index)
            pos, x = x[:, :3], x[:, 3:]
        x = x.relu()
        return self.linear(x)
    

class EGNNModel(LightningModule):
    """
    LightningModule wrapping a EGNN model.
    """
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, dropout):
        super(EGNNModel, self).__init__()
        self.save_hyperparameters()
        self.example_input_array = (torch.empty(32, in_channels), torch.empty(32, 3), torch.zeros(2, 100, dtype=torch.long))
        self.model = EGNN(in_channels=in_channels,
                            hidden_channels=hidden_channels,
                            out_channels=out_channels,
                            num_layers=num_layers,
                            dropout=dropout)

    def forward(self, x, pos, edge_index):
        return self.model(x, pos.float(), edge_index)
    
    def training_step(self, batch, batch_idx):
        out = self(batch.x, batch.pos, batch.edge_index)
        loss = F.binary_cross_entropy_with_logits(out, batch.y.view(-1, 1))
        self.log('train_loss', 
                 loss, on_step=True, 
                 on_epoch=True,
                 batch_size=batch.x.shape[0], 
                 prog_bar=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        out = self(batch.x, batch.pos, batch.edge_index)
        loss = F.binary_cross_entropy_with_logits(out, batch.y.view(-1, 1))
        self.log('hp_metric', loss, batch_size=batch.x.shape[0])
        self.log('val_loss', loss, 
                 on_step=True, on_epoch=True, 
                 batch_size=batch.x.shape[0], prog_bar=True, 
                 sync_dist=True)
        return loss

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
        self.log('train_loss', 
                 loss, on_step=True, 
                 on_epoch=True,
                 batch_size=batch.x.shape[0], 
                 prog_bar=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        out = self(batch.x, batch.edge_index)
        loss = F.binary_cross_entropy_with_logits(out, batch.y.view(-1, 1))
        self.log('hp_metric', loss, batch_size=batch.x.shape[0])
        self.log('val_loss', loss, 
                 on_step=True, on_epoch=True, 
                 batch_size=batch.x.shape[0], prog_bar=True, 
                 sync_dist=True)
        return loss