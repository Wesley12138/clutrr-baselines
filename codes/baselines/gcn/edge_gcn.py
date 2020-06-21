# GCN with Edge features
import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from codes.baselines.gcn.inits import *
from codes.net.base_net import Net


class EdgeGcnConv(MessagePassing):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 edge_dim: int,
                 # heads: int = 1,
                 # negative_slope: float = 0.2,
                 dropout: float = 0.0,
                 bias: bool = True
                 # **kwargs
                 ):
        super(EdgeGcnConv, self).__init__(aggr='max')
        self.mlp = Seq(Linear(2 * in_channels, out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        # self.concat = concat
        # self.negative_slope = negative_slope
        self.dropout = dropout
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.edge_update = Parameter(torch.Tensor(out_channels + edge_dim, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    # def reset_parameters(self):
    #     stdv = 1. / math.sqrt(self.weight.size(1))
    #     self.weight.data.uniform_(-stdv, stdv)
    #     stdv = 1. / math.sqrt(self.edge_update.size(1))
    #     self.weight.data.uniform_(-stdv, stdv)
    #     if self.bias is not None:
    #         self.bias.data.uniform_(-stdv, stdv)
    def reset_parameters(self):
        glorot(self.weight)
        # glorot(self.att)
        glorot(self.edge_update)
        zeros(self.bias)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_i, x_j):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]

        tmp = torch.cat([x_i, x_j - x_i], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.mlp(tmp)

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]
        bia = self.bias
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_channels) + ' -> ' \
               + str(self.out_channels) + ')'


class GcnEncoder(Net):
    """
    Encoder which uses EdgeGcnConv
    """

    def __init__(self,model_config, shared_embeddings=None):
        super(GcnEncoder, self).__init__(model_config)

        # flag to enable one-hot embedding if needed
        self.graph_mode = True
        self.one_hot = self.model_config.embedding.emb_type == 'one-hot'
        if self.one_hot:
            self.embedding = torch.nn.Embedding(num_embeddings=self.model_config.unique_nodes,
                                                embedding_dim=self.model_config.unique_nodes)
            self.embedding.weight = Parameter(torch.eye(self.model_config.unique_nodes))
            self.model_config.embedding.dim = self.model_config.unique_nodes
            self.model_config.graph.node_dim = self.model_config.unique_nodes
        else:
            self.embedding = torch.nn.Embedding(num_embeddings=self.model_config.unique_nodes,
                                                embedding_dim=self.model_config.embedding.dim,
                                                max_norm=1)
            torch.nn.init.xavier_uniform_(self.embedding.weight)

        # learnable embeddings
        if self.model_config.graph.edge_dim_type == 'one-hot':
            self.edge_embedding = torch.nn.Embedding(model_config.edge_types, model_config.edge_types)
            self.edge_embedding.weight = Parameter(torch.eye(self.model_config.edge_types))
            self.model_config.graph.edge_dim = self.model_config.edge_types
        else:
            self.edge_embedding = torch.nn.Embedding(model_config.edge_types, model_config.graph.edge_dim)
            torch.nn.init.xavier_uniform_(self.edge_embedding.weight)

        self.att1 = EdgeGcnConv(self.model_config.embedding.dim, self.model_config.embedding.dim,
                                self.model_config.graph.edge_dim, dropout = self.model_config.graph.dropout)
        self.att2 = EdgeGcnConv(self.model_config.embedding.dim, self.model_config.embedding.dim,
                                self.model_config.graph.edge_dim)
        # self.dropout = self.model_config.graph.dropout

    def forward(self, batch):
        data = batch.geo_batch
        x = self.embedding(data.x).squeeze(1) # N x node_dim
        # edge_attr = self.edge_embedding(data.edge_attr).squeeze(1) # E x edge_dim
        for nr in range(self.model_config.graph.num_message_rounds):
            x = F.relu(self.att1(x, data.edge_index))
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.att2(x, data.edge_index)
        # restore x into B x num_node x dim
        chunks = torch.split(x, batch.geo_slices, dim=0)
        chunks = [p.unsqueeze(0) for p in chunks]
        x = torch.cat(chunks, dim=0)
        return x, None


class GcnDecoder(Net):
    """
    Compute the graph state with the query
    """
    def __init__(self, model_config):
        super(GcnDecoder, self).__init__(model_config)
        input_dim = model_config.embedding.dim * 3
        if model_config.embedding.emb_type == 'one-hot':
            input_dim = self.model_config.unique_nodes * 3
        self.decoder2vocab = self.get_mlp(
            input_dim,
            model_config.target_size
        )

    def calculate_query(self, batch):
        """
        Extract the node embeddings using batch.query_edge
        :param batch:
        :return:
        """
        nodes = batch.encoder_outputs # B x node x dim
        query = batch.query_edge.squeeze(1).unsqueeze(2).repeat(1,1,nodes.size(2)) # B x num_q x dim
        query_emb = torch.gather(nodes, 1, query)
        return query_emb.view(nodes.size(0), -1) # B x (num_q x dim)

    def forward(self, batch, step_batch):
        query = step_batch.query_rep
        # pool the nodes
        # mean pooling
        node_avg = torch.mean(batch.encoder_outputs,1) # B x dim
        # concat the query
        node_cat = torch.cat((node_avg, query), -1) # B x (dim + dim x num_q)

        return self.decoder2vocab(node_cat), None, None # B x num_vocab
