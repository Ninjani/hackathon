from torch_geometric.nn import GATConv, HeteroConv
from torch import nn

class CrossGAT(nn.Module): # TODO: wrap with LightningModule
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