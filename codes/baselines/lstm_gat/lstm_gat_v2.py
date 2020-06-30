from torch import Tensor
from torch.nn import RNN, LSTM, GRU

from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.modules.seq2vec_encoders import CnnEncoder
from allennlp.modules.seq2vec_encoders.cnn_highway_encoder import CnnHighwayEncoder
from allennlp.modules.seq2seq_encoders import IntraSentenceAttentionEncoder
from allennlp.modules.similarity_functions import MultiHeadedSimilarity

from typing import Tuple, List, Callable, Optional

import torch
from torch import nn, Tensor

from torch_geometric.data import Batch

from codes.utils.util import pad_sequences

from typing import List, Callable, Any

# GAT with Edge features
import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from codes.baselines.gat.inits import *
from codes.net.base_net import Net

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from codes.net.base_net import Net
from codes.utils.util import check_id_emb
from codes.net.attention import Attn
from torch.nn import functional as F
import numpy as np
import pdb

class EdgeGatConv(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 edge_dim,
                 heads=1,
                 concat=False,
                 negative_slope=0.2,
                 dropout=0.,
                 bias=True):
        super(EdgeGatConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = Parameter(torch.Tensor(in_channels, heads * out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels + edge_dim))
        self.edge_update = Parameter(torch.Tensor(out_channels + edge_dim, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        glorot(self.edge_update)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr, size=None):
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_edges = torch.zeros(x.size(0), edge_attr.size(1)).to(edge_index.device)
        edge_attr = torch.cat([edge_attr, self_loop_edges], dim=0)
        x = torch.mm(x, self.weight).view(-1, self.heads, self.out_channels)
        return self.propagate(edge_index, size=size, x=x, edge_attr=edge_attr)

    def message(self, edge_index_i, x_i, x_j, size_i, edge_attr):
        edge_attr = edge_attr.unsqueeze(1).repeat(1, self.heads, 1)
        x_j = torch.cat([x_j, edge_attr], dim=-1)

        x_i = x_i.view(-1, self.heads, self.out_channels)
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)

        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out = torch.mm(aggr_out, self.edge_update)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

class GLEncoder(Net):
    """
    Encoder which uses EdgeGatConv
    """

    def __init__(self,model_config, shared_embeddings=None):
        super(GLEncoder, self).__init__(model_config)

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

        self.att1 = EdgeGatConv(self.model_config.embedding.dim, self.model_config.embedding.dim,
                                self.model_config.graph.edge_dim, heads=self.model_config.graph.num_reads,
                                dropout=self.model_config.graph.dropout)
        self.att2 = EdgeGatConv(self.model_config.embedding.dim, self.model_config.embedding.dim,
                                self.model_config.graph.edge_dim)

        self.lstm = PytorchSeq2VecWrapper(LSTM(input_size=model_config.embedding.dim,
                                               hidden_size=model_config.embedding.dim,
                                               bidirectional=model_config.encoder.bidirectional,
                                               batch_first=True, dropout=model_config.encoder.dropout))

    def forward(self, batch):
        data = batch.geo_batch
        x = self.embedding(data.x).squeeze(1) # N x node_dim
        edge_attr = self.edge_embedding(data.edge_attr).squeeze(1) # E x edge_dim
        for nr in range(self.model_config.graph.num_message_rounds):
            x = F.dropout(x, p=0.6, training=self.training)
            x = F.elu(self.att1(x, data.edge_index, edge_attr))
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.att2(x, data.edge_index, edge_attr)
        # restore x into B x num_node x dim
        chunks = torch.split(x, batch.geo_slices, dim=0)
        chunks = [p.unsqueeze(0) for p in chunks]
        x = torch.cat(chunks, dim=0)

        UNK = 'UNK'
        targets = batch.query_edge
        relation_lst = list(self.model_config.classes)  # {list:22} ['aunt', 'brother', ...]
        entity_lst = list(self.model_config.entity_lst)
        symbol_lst = sorted({s for s in entity_lst + relation_lst} | {'UNK'})
        symbol_to_idx = {s: i for i, s in enumerate(symbol_lst)}
        nb_symbols = len(symbol_lst)  # 101
        symbol_embeddings = nn.Embedding(nb_symbols, self.embedding.embedding_dim, sparse=False)  # Embedding(101, 100)
        linear_story_lst = []
        target_lst = []
        for tri_state in batch.tri_states:
            linear_story = [symbol_to_idx[t] for t in tri_state]
            linear_story_lst += [linear_story[3:]]
            target = [linear_story[0], linear_story[2]]
            target_lst += [target]
        story_padded = pad_sequences(linear_story_lst, value=symbol_to_idx['UNK'])
        # {ndarray:(100,6)}  i.e. 0=[ 0 84 12 12 95  1]
        batch_linear_story = torch.tensor(story_padded, dtype=torch.long, device=targets.device)  # torch.Size([100, 6])
        batch_target = torch.tensor(target_lst, dtype=torch.long, device=targets.device)  # torch.Size([100, 2])
        batch_linear_story_emb = symbol_embeddings(batch_linear_story)  # torch.Size([100, 6, 100])
        batch_target_emb = symbol_embeddings(batch_target)  # torch.Size([100, 2, 100])
        story_code = self.lstm(batch_linear_story_emb, None)  # torch.Size([100, 200])  (batch, num_directions * hidden_size)    None:default to zero
        target_code = self.lstm(batch_target_emb, None)  # torch.Size([100, 200])
        story_target_code = torch.cat([story_code, target_code], dim=-1)  # torch.Size([100, 400])

        return x, story_target_code  #None


class GatDecoder(Net):
    """
    Compute the graph state with the query
    """

    def __init__(self, model_config):
        super(GatDecoder, self).__init__(model_config)
        input_dim = model_config.embedding.dim * 7  #3
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
        nodes = batch.encoder_outputs  # B x node x dim
        query = batch.query_edge.squeeze(1).unsqueeze(2).repeat(1, 1, nodes.size(2))  # B x num_q x dim
        query_emb = torch.gather(nodes, 1, query)
        return query_emb.view(nodes.size(0), -1)  # B x (num_q x dim)

    def forward(self, batch, step_batch):
        query = step_batch.query_rep        # B x (num_q x dim)     #100x200
        # pool the nodes
        # mean pooling
        node_avg = torch.mean(batch.encoder_outputs, 1)  # B x dim      #100x100
        # concat the query
        node_cat = torch.cat((node_avg, query), -1)  # B x (dim + dim x num_q)      #100x300

        node_cat = torch.cat((node_cat, batch.encoder_hidden), -1)  #100x700

        return self.decoder2vocab(node_cat), None, None  # B x num_vocab        #100x18
