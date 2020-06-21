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
from torch_geometric.utils import add_self_loops, softmax, scatter_
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

class LstmEncoder(Net):

    def __init__(self, model_config, shared_embeddings=None, use_embedding=True):
        super().__init__(model_config)

        if not shared_embeddings:
            self.init_embeddings()  # Embedding(3071, 100, padding_idx=0, max_norm=1)
        else:
            self.embedding = shared_embeddings

        self.lstm  = PytorchSeq2VecWrapper(LSTM(input_size=model_config.embedding.dim,
                    hidden_size=model_config.embedding.dim,
                    bidirectional= model_config.encoder.bidirectional,
                    batch_first = True, dropout = model_config.encoder.dropout))

        # LSTM(100, 100, num_layers=2, batch_first=True, bidirectional=True)
        # self.use_embedding = use_embedding

    def forward(self, batch):
        UNK = 'UNK'
        targets = batch.query_edge
        relation_lst = list(self.model_config.classes)  # {list:22} ['aunt', 'brother', ...]
        entity_lst = list(self.model_config.entity_lst)
        symbol_lst = sorted({s for s in entity_lst + relation_lst} | {'UNK'})
        symbol_to_idx = {s: i for i, s in enumerate(symbol_lst)}
        # {dict:101} {'ENTITY_0': 0, 'ENTITY_1': 1, 'ENTITY_10': 2,..., 'UNK':78,...,'uncle':99,...}

        nb_symbols = len(symbol_lst)  # 101
        symbol_embeddings = nn.Embedding(nb_symbols, self.embedding.embedding_dim, sparse=False)  # Embedding(101, 100)

        linear_story_lst = []
        target_lst = []

        for tri_state in batch.tri_states:  # 对每一个[triple(story)]\t(target)
            linear_story = [symbol_to_idx[t] for t in tri_state]
            linear_story_lst += [linear_story[3:]]
            target = [linear_story[0],linear_story[2]]
            target_lst += [target]

        story_padded = pad_sequences(linear_story_lst, value=symbol_to_idx['UNK'])
        # {ndarray:(100,6)}  i.e. 0=[ 0 84 12 12 95  1]
        batch_linear_story = torch.tensor(story_padded, dtype=torch.long, device=targets.device)  # torch.Size([100, 6])
        batch_target = torch.tensor(target_lst, dtype=torch.long, device=targets.device)  # torch.Size([100, 2])

        batch_linear_story_emb = symbol_embeddings(batch_linear_story)  # torch.Size([100, 6, 100])
        batch_target_emb = symbol_embeddings(batch_target)  # torch.Size([100, 2, 100])

        story_code = self.lstm(batch_linear_story_emb,None)  # torch.Size([100, 200])  (batch, num_directions * hidden_size)    None:default to zero
        #target_code = self.lstm(batch_target_emb, None)  # torch.Size([100, 200])

        # story_target_code = torch.cat([story_code, target_code], dim=-1)  # torch.Size([100, 400])
        # logits = self.projection(story_target_code)  # torch.Size([100, 18])      100x400  %*% 400x18 + bias:18
        # return logits  # torch.Size([100, 18])
        return story_code, None

class GatDecoder(Net):
    """
    Compute the graph state with the query
    """

    def __init__(self, model_config):
        super(GatDecoder, self).__init__(model_config)
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

        return self.decoder2vocab(node_cat), None, None  # B x num_vocab        #100x18
