from typing import Tuple, List, Callable, Optional

import torch
from allennlp.modules.seq2seq_encoders import IntraSentenceAttentionEncoder
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.modules.seq2vec_encoders import CnnEncoder
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.seq2vec_encoders.cnn_highway_encoder import CnnHighwayEncoder
from allennlp.modules.similarity_functions import MultiHeadedSimilarity
from allennlp.modules.augmented_lstm import AugmentedLstm
from allennlp.modules.stacked_alternating_lstm import StackedAlternatingLstm
from allennlp.modules.stacked_bidirectional_lstm import StackedBidirectionalLstm
from torch import Tensor
from torch.nn import Parameter, RNN, LSTM, GRU

from codes.net.base_net import Net


class Encoder(Net):
    """
    Encoder with boe/cnn/cnnh/rnn/lstm/gru/intra/multihead, selected from Seq2VecEncoderFactory
    """

    def __init__(self, model_config, shared_embeddings=None):
        super(Encoder, self).__init__(model_config)

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
            self.embedding = torch.nn.Embedding(num_embeddings=self.model_config.unique_nodes + 1,  # add 1 for padding
                                                embedding_dim=self.model_config.embedding.dim,
                                                max_norm=1)
            torch.nn.init.xavier_uniform_(self.embedding.weight)

        # learnable embeddings
        if self.model_config.graph.edge_dim_type == 'one-hot':
            self.edge_embedding = torch.nn.Embedding(model_config.edge_types, model_config.edge_types)
            self.edge_embedding.weight = Parameter(torch.eye(self.model_config.edge_types))
            self.model_config.graph.edge_dim = self.model_config.edge_types
        else:
            self.edge_embedding = torch.nn.Embedding(model_config.edge_types + 1, model_config.graph.edge_dim)
            torch.nn.init.xavier_uniform_(self.edge_embedding.weight)

        encoder_factory = Seq2VecEncoderFactory()
        self.encoder = encoder_factory.build(model_config)

    def forward(self, batch):
        data = batch.geo_batch
        edge_index = data.edge_index.t()  # s1,o1;s2,o2
        chunk_index = torch.split(edge_index, batch.story_edge_no, dim=0)  # s1,o1;s2,o2
        chunks_index = [i - torch.full([i.size(0), i.size(1)], int(i.min())) for i in chunk_index]  # back to 0,1,2
        edge_attr = data.edge_attr  # r1;r2  200x1
        r = torch.split(edge_attr, batch.story_edge_no, dim=0)  # r1;r2
        r = [i.float() for i in r]
        s = [i[:, 0].unsqueeze(-1) for i in chunks_index]  # s1;s2
        o = [i[:, 1].unsqueeze(-1) for i in chunks_index]  # o1;o2
        triples = [torch.cat((i, j, k), 1).view(1, -1) for i, j, k in zip(s, r, o)]  # s1,r1,o1,s2,r2,o2

        # if max(batch.story_edge_no) != min(batch.story_edge_no):
        max_no = max(batch.story_edge_no)
        need_add = [max_no - i for i in batch.story_edge_no]  # list:100
        add_triple = torch.tensor([[self.model_config.unique_nodes, self.model_config.edge_types,
                                    self.model_config.unique_nodes]], dtype=torch.float, device=edge_index.device)  # [11., 14., 11.]]
        story_padded = [torch.cat((i, add_triple.repeat(1, j)), 1) for i, j in zip(triples, need_add)]  # padding
        batch_story = torch.stack(story_padded, dim=0)
        chunk = torch.split(batch_story.squeeze(1).view(-1, 3).long(), 1, dim=1)  # list:3  200x1
        s_emb = self.embedding(chunk[0]).squeeze(1)
        r_emb = self.edge_embedding(chunk[1]).squeeze(1)
        o_emb = self.embedding(chunk[2]).squeeze(1)
        story_emb = torch.cat((s_emb, r_emb, o_emb), 1)
        story_emb = story_emb.view(batch.batch_size, -1, self.model_config.embedding.dim)
        story_code = self.encoder(story_emb, None)
        query = self.embedding(batch.query_edge.squeeze(1))
        query_emb = self.encoder(query, None)
        return story_code, query_emb


class Decoder(Net):
    """
    Compute the graph state with the query
    """

    def __init__(self, model_config):
        super(Decoder, self).__init__(model_config)

        model_name = model_config.encoder.model_name.split('_')[-1]
        if model_name == 'boe':
            input_dim = model_config.embedding.dim * 2
        elif model_name == 'cnn':
            input_dim = model_config.encoder.output_dim * 2
        elif model_name == 'cnnh':
            input_dim = model_config.encoder.projection_dim * 2
        elif model_name == 'birnn' or 'bigru' or 'bilstm' or 'intra' or 'multihead':
            input_dim = model_config.encoder.hidden_dim * 4
        else: # rnn, lstm, gru
            input_dim = model_config.encoder.hidden_dim * 2

        if model_config.embedding.emb_type == 'one-hot':
            input_dim = self.model_config.unique_nodes * 3

        self.decoder2vocab = self.get_mlp(
            input_dim,
            model_config.target_size
        )

        # self.projection = None

    def calculate_query(self, batch):
        """
        Extract the node embeddings using batch.query_edge
        :param batch:
        :return:
        """

        return batch.encoder_hidden

    def forward(self, batch, step_batch):
        query = step_batch.query_rep
        node_cat = torch.cat([batch.encoder_outputs, query], dim=-1)

        return self.decoder2vocab(node_cat), None, None

        # if self.projection is None:
        #     self.projection = nn.Linear(node_cat[-1], 18)
        # return self.projection, None, None

class Seq2VecEncoderFactory:
    def __init__(self):
        super().__init__()

    @staticmethod
    def build(model_config,
              ngram_filter_sizes=(1, 2),
              filters=[[1, 4], [2, 8]]
              ) -> Callable[[Tensor, Optional[Tensor]], Tensor]:

        name = model_config.encoder.model_name.split('_')[1]
        embedding_dim = model_config.embedding.dim
        hidden_size = model_config.encoder.hidden_dim
        num_filters = model_config.encoder.num_filters
        num_heads = model_config.graph.num_reads
        output_dim = model_config.encoder.output_dim
        num_highway = model_config.encoder.num_highway
        projection_dim = model_config.encoder.projection_dim

        encoder = None
        if name == 'boe':
            encoder = BagOfEmbeddingsEncoder(embedding_dim=embedding_dim, averaged=True)
        elif name == 'cnn':
            encoder = CnnEncoder(embedding_dim=embedding_dim, num_filters=num_filters,
                                 ngram_filter_sizes=ngram_filter_sizes, output_dim=output_dim)
        elif name == 'cnnh':
            encoder = CnnHighwayEncoder(embedding_dim=embedding_dim, filters=filters, num_highway=num_highway,
                                        projection_dim=projection_dim, projection_location="after_cnn")
        elif name == 'rnn':
            rnn = RNN(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)
            encoder = PytorchSeq2VecWrapper(rnn)
        elif name == 'lstm':
            lstm = LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)
            encoder = PytorchSeq2VecWrapper(lstm)
        elif name == 'gru':
            gru = GRU(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)
            encoder = PytorchSeq2VecWrapper(gru)
        elif name == 'birnn':
            birnn = RNN(input_size=embedding_dim, bidirectional=True, hidden_size=hidden_size, batch_first=True)
            encoder = PytorchSeq2VecWrapper(birnn)
        elif name == 'bilstm':
            bilstm = LSTM(input_size=embedding_dim, bidirectional=True, hidden_size=hidden_size, batch_first=True)
            encoder = PytorchSeq2VecWrapper(bilstm)
        elif name == 'bigru':
            bigru = GRU(input_size=embedding_dim, bidirectional=True, hidden_size=hidden_size, batch_first=True)
            encoder = PytorchSeq2VecWrapper(bigru)
        elif name == 'intra':
            intra = IntraSentenceAttentionEncoder(input_dim=embedding_dim, projection_dim=output_dim, combination="1,2")
            aggr = PytorchSeq2VecWrapper(LSTM(input_size=embedding_dim + output_dim, bidirectional=True,
                                              hidden_size=hidden_size, batch_first=True))
            encoder = lambda x, y: aggr(intra(x, y), y)
        elif name == 'multihead':
            sim = MultiHeadedSimilarity(num_heads, embedding_dim)
            multi = IntraSentenceAttentionEncoder(input_dim=embedding_dim, projection_dim=embedding_dim,
                                                  similarity_function=sim, num_attention_heads=num_heads,
                                                  combination="1+2")
            aggr = PytorchSeq2VecWrapper(LSTM(input_size=embedding_dim, bidirectional=True,
                                              hidden_size=hidden_size, batch_first=True))
            encoder = lambda x, y: aggr(multi(x, y), y)
        elif name == 'auglstm':  # allennlp.common.checks.ConfigurationError: "inputs must be PackedSequence but got <class 'torch.Tensor'>"
            encoder = AugmentedLstm(input_size=embedding_dim, hidden_size=hidden_size)
        elif name == 'stalstm':  # allennlp.common.checks.ConfigurationError: "inputs must be PackedSequence but got <class 'torch.Tensor'>"
            encoder = StackedAlternatingLstm(input_size=embedding_dim, hidden_size=hidden_size, num_layers=2)
        elif name == 'stabilstm':  # allennlp.common.checks.ConfigurationError: "inputs must be PackedSequence but got <class 'torch.Tensor'>"
            encoder = StackedBidirectionalLstm(input_size=embedding_dim, hidden_size=hidden_size, num_layers=2)
        assert encoder is not None
        return encoder
