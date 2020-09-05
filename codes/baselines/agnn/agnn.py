import torch
from torch.nn import Parameter, Linear
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from codes.baselines.gcn.inits import *
from codes.net.base_net import Net

class EdgeAGNNConv(MessagePassing):
    def __init__(self,
                 in_channels: int,
                 edge_dim: int,
                 requires_grad: bool = True):
        super(EdgeAGNNConv, self).__init__(aggr='add')

        self.in_channels = in_channels
        self.edge_dim = edge_dim
        self.requires_grad = requires_grad

        self.weight = Parameter(torch.Tensor(in_channels + edge_dim, in_channels))
        self.edge_update = Parameter(torch.Tensor(in_channels + edge_dim, in_channels))

        if requires_grad:
            self.beta = Parameter(torch.Tensor(1))
        else:
            self.register_buffer('beta', torch.ones(1))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.edge_update)
        if self.requires_grad:
            self.beta.data.fill_(1)

    def forward(self, x, edge_index, edge_attr, size=None):
        """"""
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(self.node_dim))

        self_loop_edges = torch.zeros(x.size(0), edge_attr.size(1)).to(edge_index.device)
        edge_attr = torch.cat([edge_attr, self_loop_edges], dim=0)

        x_norm = F.normalize(x, p=2., dim=-1)

        return self.propagate(edge_index, x=x, edge_attr=edge_attr, x_norm=x_norm, size=None)

    def message(self, x_j, edge_attr, x_norm_i, index, size_i):
        x_j = torch.cat([x_j, edge_attr], dim=-1)   # 500 x (100+20)
        x_norm_j = F.normalize(x_j, p=2., dim=-1)
        x_norm_j = torch.mm(x_norm_j, self.weight)

        alpha = self.beta * (x_norm_i * x_norm_j).sum(dim=-1)
        alpha = softmax(alpha, index, size_i)
        return x_j * alpha.view(-1, 1)

    def update(self, aggr_out):
        aggr_out = torch.mm(aggr_out, self.edge_update)
        return aggr_out

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class AgnnEncoder(Net):
    """
    Encoder which uses AGNNConv
    """

    def __init__(self, model_config, shared_embeddings=None):
        super(AgnnEncoder, self).__init__(model_config)

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

        self.lin = Linear(self.model_config.embedding.dim, self.model_config.embedding.dim)
        self.prop1 = EdgeAGNNConv(self.model_config.embedding.dim, self.model_config.graph.edge_dim, requires_grad=False)
        self.prop2 = EdgeAGNNConv(self.model_config.embedding.dim, self.model_config.graph.edge_dim, requires_grad=True)

    def forward(self, batch):
        data = batch.geo_batch
        x = self.embedding(data.x).squeeze(1) # N x node_dim   torch.Size([300, 100])
        edge_attr = self.edge_embedding(data.edge_attr).squeeze(1)  # E x edge_dim
        for nr in range(self.model_config.graph.num_message_rounds):
            x = F.dropout(x, p=0.6, training=self.training)
            x = F.elu(self.lin(x))
            x = self.prop1(x, data.edge_index, edge_attr)
            x = self.prop2(x, data.edge_index, edge_attr)
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.lin(x)
        # restore x into B x num_node x dim
        chunks = torch.split(x, batch.geo_slices, dim=0)
        chunks = [p.unsqueeze(0) for p in chunks]
        x = torch.cat(chunks, dim=0)
        return x, None


class AgnnDecoder(Net):
    """
    Compute the graph state with the query
    """
    def __init__(self, model_config):
        super(AgnnDecoder, self).__init__(model_config)
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