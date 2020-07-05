import torch
from torch.nn import Parameter
from torch.nn import LSTM
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from codes.net.base_net import Net

class GraphLstmEncoder(Net):
    """
    Encoder which uses EdgeGatConv
    """

    def __init__(self,model_config, shared_embeddings=None):
        super(GraphLstmEncoder, self).__init__(model_config)

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
            self.embedding = torch.nn.Embedding(num_embeddings=self.model_config.unique_nodes+1,  # add 1 for padding
                                                embedding_dim=self.model_config.embedding.dim,
                                                max_norm=1)
            torch.nn.init.xavier_uniform_(self.embedding.weight)

        # learnable embeddings
        if self.model_config.graph.edge_dim_type == 'one-hot':
            self.edge_embedding = torch.nn.Embedding(model_config.edge_types, model_config.edge_types)
            self.edge_embedding.weight = Parameter(torch.eye(self.model_config.edge_types))
            self.model_config.graph.edge_dim = self.model_config.edge_types
        else:
            self.edge_embedding = torch.nn.Embedding(model_config.edge_types+1, model_config.graph.edge_dim)
            torch.nn.init.xavier_uniform_(self.edge_embedding.weight)

        self.lstm = PytorchSeq2VecWrapper(LSTM(input_size=model_config.embedding.dim,
                                           hidden_size=model_config.encoder.hidden_dim,  #   embedding.dim,
                                           bidirectional=model_config.encoder.bidirectional,
                                           batch_first=True, dropout=model_config.encoder.dropout))


    def forward(self, batch):
        data = batch.geo_batch
        edge_index = data.edge_index.t()  # s1,o1;s2,o2  200x2
        chunk_index = torch.split(edge_index, batch.story_edge_no, dim=0)  # s1,o1;s2,o2   list:100  2 x 2
        chunks_index = [i - torch.full([i.size(0),i.size(1)],int(i.min())) for i in chunk_index]  # back to 0,1,2  list:100  2 x 2
        edge_attr = data.edge_attr   # r1;r2  200x1
        r = torch.split(edge_attr, batch.story_edge_no, dim=0)  # r1;r2   list:100  2 x 1
        r = [i.float() for i in r]
        s = [i[:,0].unsqueeze(-1) for i in chunks_index] # s1;s2  list:100  2x1
        o = [i[:,1].unsqueeze(-1) for i in chunks_index]   # o1;o2  list:100  2x1
        triples = [torch.cat((i,j,k),1).view(1,-1) for i,j,k in zip(s,r,o)]   # s1,r1,o1,s2,r2,o2  list:100 1x6

        # if max(batch.story_edge_no) != min(batch.story_edge_no):
        max_no = max(batch.story_edge_no)
        need_add = [max_no-i for i in batch.story_edge_no]   # list:100
        add_triple = torch.tensor([[11.,14.,11.]], dtype=torch.float, device=edge_index.device)
        story_padded = [torch.cat((i,add_triple.repeat(1,j)),1) for i,j in zip(triples,need_add)]   # padding  list:100 1x6(max)
        batch_story = torch.stack(story_padded, dim=0)  # tensor  100x1x6(max)
        chunk = torch.split(batch_story.squeeze(1).view(-1,3).long(), 1, dim=1)  # list:3  200x1
        s_emb = self.embedding(chunk[0]).squeeze(1)  # 200x100
        r_emb = self.edge_embedding(chunk[1]).squeeze(1)  # 200x100
        o_emb = self.embedding(chunk[2]).squeeze(1)  # 200x100
        story_emb = torch.cat((s_emb, r_emb, o_emb), 1)  # 200x300
        story_emb = story_emb.view(batch.batch_size, -1, self.model_config.embedding.dim)  # 100x6x100
        story_code = self.lstm(story_emb, None)  # 100 x 200
        query = self.embedding(batch.query_edge.squeeze(1))  # 100 x 2 x 100
        query_emb = self.lstm(query, None)  # 100 x 200
        return story_code, query_emb

        # emb_dim = self.model_config.embedding.dim
        # data = batch.geo_batch
        # edge_attr = self.edge_embedding(data.edge_attr).squeeze(1)  # r1;r2   200x100
        # x = self.embedding(data.x).squeeze(1) # 0;1;2;0;1;2   300x100
        # edge_index = data.edge_index.t()  # 0,1;1,2   200x2
        # edge_emb = [x[i, :] for e in edge_index for i in e]  # list:400  1x100
        # edge_emb = torch.cat(edge_emb, dim=0).view(-1, 2*emb_dim)    # s1,o1;s2,o2   200x200
        # triple_emb = torch.cat((edge_emb[:, :emb_dim], edge_attr, edge_emb[:, emb_dim:]),1) # s1,r1,o1 ; s2,r2,o2   200 x 300
        # story_dim = [i*3 for i in batch.story_edge_no]
        # chunks = torch.split(triple_emb.view(-1,emb_dim), story_dim, dim=0) # s1;r1;o1;s2;r2;o2   list:100  6 x 100
        # chunks = torch.stack(chunks, dim=0)  # 100 x 6 x 100
        # story_code = self.lstm(chunks, None)      # 100 x 200
        # return story_code, x


class Decoder(Net):
    """
    Compute the graph state with the query
    """
    def __init__(self, model_config):
        super(Decoder, self).__init__(model_config)
        input_dim = model_config.encoder.hidden_dim * 4
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

        return batch.encoder_hidden   # B x (2 x hidden_dim)

    def forward(self, batch, step_batch):
        query = step_batch.query_rep
        node_cat = torch.cat([batch.encoder_outputs, query], dim=-1)  # 100 x 400

        return self.decoder2vocab(node_cat), None, None # B x num_vocab
