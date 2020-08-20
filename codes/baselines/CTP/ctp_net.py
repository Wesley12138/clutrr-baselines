from typing import Tuple, List, Callable, Optional

import torch
from torch import nn, Tensor
from torch.nn import Parameter
from codes.net.base_net import Net
import torch.nn.functional as F

from codes.baselines.CTP.kb import NeuralKB
from codes.reformulators import BaseReformulator
from codes.reformulators import StaticReformulator
from codes.reformulators import LinearReformulator
from codes.reformulators import AttentiveReformulator
from codes.reformulators import MemoryReformulator
from codes.reformulators import NTPReformulator
from codes.kernels import BaseKernel, GaussianKernel

import numpy as np


class Hoppy(nn.Module):
    def __init__(self,
                 model: NeuralKB,
                 hops_lst: List[Tuple[BaseReformulator, bool]],
                 k: int = 10,   # 3
                 depth: int = 0,   # 2
                 tnorm_name: str = 'min',
                 R: Optional[int] = None):
        super().__init__()

        self.model: NeuralKB = model
        # NeuralKB(
        #     (kernel): GaussianKernel()
        # )
        self.k = k   # 3
        self.depth = depth   # 2
        assert self.depth >= 0
        self.tnorm_name = tnorm_name   # 'min
        assert self.tnorm_name in {'min', 'prod'}
        self.R = R   # None
        self._hops_lst = nn.ModuleList([hops for hops, _ in hops_lst])   # similar to nn.Sequential()，but not follow the order and forward editable
        self.hops_lst = hops_lst

        print(f'Hoppy(k={k}, depth={depth}, hops_lst={[h.__class__.__name__ for h in self._hops_lst]})')
        # INFO:kbcr.clutrr.models.model:Hoppy(k=3, depth=2, hops_lst=['MemoryReformulator', ...'MemoryReformulator'])  10 times

    def _tnorm(self, x: Tensor, y: Tensor):
        return x * y if self.tnorm_name == 'prod' else torch.min(x, y)

    def r_hop(self,
              rel: Tensor,   # torch.Size([22, 50])   all target r'
              arg1: Optional[Tensor],   # torch.Size([22, 50])   target s'
              arg2: Optional[Tensor],   # None
              facts: List[Tensor],   # [2x50 2x50 2x50]  story[r, s, o] emb
              entity_embeddings: Tensor,   # 3x50  story nodes(names) ascending
              depth: int) -> Tuple[Tensor, Tensor]:   # 0=1-1
        assert (arg1 is None) ^ (arg2 is None)
        assert depth >= 0

        batch_size, embedding_size = rel.shape[0], rel.shape[1]   # 22, 50

        # [B, N]
        scores_sp, scores_po = self.r_forward(rel, arg1, arg2, facts, entity_embeddings, depth=depth)   # torch.Size([22, 3]), None
        scores = scores_sp if arg2 is None else scores_po   # torch.Size([22, 3])   arg2=None

        k = min(self.k, scores.shape[1])   # 3  k=3, scores.shape[1]=3

        # [B, K], [B, K]
        z_scores, z_indices = torch.topk(scores, k=k, dim=1)   # torch.Size([22, 3]), torch.Size([22, 3])
        # [B, K, E]
        # z_emb = entity_embeddings(z_indices)
        z_emb = F.embedding(z_indices, entity_embeddings)   # torch.Size([22, 3, 50])      entity_embeddings:3x50

        assert z_emb.shape[0] == batch_size
        assert z_emb.shape[2] == embedding_size

        return z_scores, z_emb

    def score(self,
              rel: Tensor,    # 22x50   all target r  (22 possible relations)
              arg1: Tensor,    # 22x50   target s (repeat 22 times)
              arg2: Tensor,    # 22x50   target o (repeat 22 times)
              facts: List[Tensor],   # [2x50 2x50 2x50]  story[r, s, o] emb
              entity_embeddings: Tensor) -> Tensor:   # 3x50    story nodes(names) ascending
        res = self.r_score(rel, arg1, arg2, facts, entity_embeddings, depth=self.depth)   #22  depth=2
        return res

    def r_score(self,
                rel: Tensor,   # 22x50   all target r  (22 possible relations)
                arg1: Tensor,   # 22x50   target s (repeat 22 times)
                arg2: Tensor,   # 22x50   target o (repeat 22 times)
                facts: List[Tensor],   # [2x50 2x50 2x50]  story[r, s, o] emb
                entity_embeddings: Tensor,   # 3x50    story nodes(names) ascending
                depth: int) -> Tensor:   # 2
        res = None
        for d in range(depth + 1):   # 0~2
            scores = self.depth_r_score(rel, arg1, arg2, facts, entity_embeddings, depth=d)   # depth=0~2  i.e. obtain 22 scores
            res = scores if res is None else torch.max(res, scores)
        return res

    def depth_r_score(self,
                      rel: Tensor,   # 22x50   all target r  (22 possible relations)
                      arg1: Tensor,   # 22x50   target s (repeat 22 times)
                      arg2: Tensor,   # 22x50   target o (repeat 22 times)
                      facts: List[Tensor],   # [2x50 2x50 2x50]  story[r, s, o] emb
                      entity_embeddings: Tensor,   # 3x50    story nodes(names) ascending
                      depth: int) -> Tensor:   # 0~2
        assert depth >= 0

        if depth == 0:
            return self.model.score(rel, arg1, arg2, facts)   # Gaussian kernel

        batch_size, embedding_size = rel.shape[0], rel.shape[1]   # 22, 50
        global_res = None

        mask = None

        new_hops_lst = self.hops_lst   # list:10

        if self.R is not None:
            batch_rules_scores = torch.cat([h.prior(rel).view(-1, 1) for h, _ in self.hops_lst], 1)
            topk, indices = torch.topk(batch_rules_scores, self.R)

            # [R x E]
            rule_heads = torch.cat([h.head for h, _ in self.hops_lst], dim=0)
            rule_body1s = torch.cat([h.memory_lst[0] for h, _ in self.hops_lst], dim=0)
            rule_body2s = torch.cat([h.memory_lst[1] for h, _ in self.hops_lst], dim=0)

            kernel = self.hops_lst[0][0].kernel
            new_rule_heads = F.embedding(indices, rule_heads)
            new_rule_body1s = F.embedding(indices, rule_body1s)
            new_rule_body2s = F.embedding(indices, rule_body2s)

            # print(new_rule_heads.shape[1], self.R)
            assert new_rule_heads.shape[1] == self.R

            new_hops_lst = []
            for i in range(new_rule_heads.shape[1]):
                r = GNTPReformulator(kernel=kernel, head=new_rule_heads[:, i, :],
                                     body=[new_rule_body1s[:, i, :], new_rule_body2s[:, i, :]])
                new_hops_lst += [(r, False)]

            # import sys
            # sys.exit(0)

            # mask = torch.zeros_like(batch_rules_scores).scatter_(1, indices, torch.ones_like(topk))

        # for hops_generator, is_reversed in self.hops_lst:
        # for rule_idx, (hops_generator, is_reversed) in enumerate(self.hops_lst):
        for rule_idx, (hops_generator, is_reversed) in enumerate(new_hops_lst):   # 0~9，512x50+50x512 ，False
            sources, scores = arg1, None   # torch.Size([22, 50]) target s, None

            # XXX
            prior = hops_generator.prior(rel)   # None   rel:22x50 all target r
            if prior is not None:

                if mask is not None:
                    prior = prior * mask[:, rule_idx]
                    if (prior != 0.0).sum() == 0:
                        continue

                scores = prior

            # scores = hops_generator.prior(rel)

            hop_rel_lst = hops_generator(rel)   #list:2 [torch.Size([22, 50]), torch.Size([22, 50])]    rel:22x50 all target r, with linear(50,512), @memory(512,50)
            nb_hops = len(hop_rel_lst)   # 2

            for hop_idx, hop_rel in enumerate(hop_rel_lst, start=1):   # 1~2   torch.Size([22, 50])
                # [B * S, E]
                sources_2d = sources.view(-1, embedding_size)   # torch.Size([22, 50])
                nb_sources = sources_2d.shape[0]   # 22

                nb_branches = nb_sources // batch_size   # 1=22/22

                hop_rel_3d = hop_rel.view(-1, 1, embedding_size).repeat(1, nb_branches, 1)   # torch.Size([22, 1, 50])
                hop_rel_2d = hop_rel_3d.view(-1, embedding_size)   # torch.Size([22, 50])

                if hop_idx < nb_hops:   # 1<2
                    # [B * S, K], [B * S, K, E]
                    if is_reversed:
                        z_scores, z_emb = self.r_hop(hop_rel_2d, None, sources_2d,
                                                     facts, entity_embeddings, depth=depth - 1)
                    else:
                        z_scores, z_emb = self.r_hop(hop_rel_2d, sources_2d, None,
                                                     facts, entity_embeddings, depth=depth - 1)
                        # all target r’: torch.Size([22, 50]), target s‘: torch.Size([22, 50]), None, [2x50 2x50 2x50] story[r, s, o] emb, 3x50 story nodes(names) ascending, 0=1-1
                        # z_scores:torch.Size([22, 3]), z_emb:torch.Size([22, 3, 50])
                    k = z_emb.shape[1]   # 3

                    # [B * S * K]
                    z_scores_1d = z_scores.view(-1)   # 66
                    # [B * S * K, E]
                    z_emb_2d = z_emb.view(-1, embedding_size)   # torch.Size([66, 50])

                    # [B * S * K, E]
                    sources = z_emb_2d   # torch.Size([66, 50])
                    # [B * S * K]
                    scores = z_scores_1d if scores is None \
                        else self._tnorm(z_scores_1d, scores.view(-1, 1).repeat(1, k).view(-1))   # 66
                else:
                    # [B, S, E]
                    arg2_3d = arg2.view(-1, 1, embedding_size).repeat(1, nb_branches, 1)   # torch.Size([22, 3, 50])
                    # [B * S, E]
                    arg2_2d = arg2_3d.view(-1, embedding_size)   # torch.Size([66, 50])

                    # [B * S]
                    if is_reversed:
                        z_scores_1d = self.r_score(hop_rel_2d, arg2_2d, sources_2d,
                                                   facts, entity_embeddings, depth=depth - 1)
                    else:
                        z_scores_1d = self.r_score(hop_rel_2d, sources_2d, arg2_2d,
                                                   facts, entity_embeddings, depth=depth - 1)   # 66

                    scores = z_scores_1d if scores is None else self._tnorm(z_scores_1d, scores)   # self._tnorm=min(z_scores_1d, scores)

            if scores is not None:
                scores_2d = scores.view(batch_size, -1)   # torch.Size([22, 3])
                res, _ = torch.max(scores_2d, dim=1)   # 22
            else:
                res = self.model.score(rel, arg1, arg2, facts)

            global_res = res if global_res is None else torch.max(global_res, res)   # 22

        return global_res

    def forward(self,
                rel: Tensor, arg1: Optional[Tensor], arg2: Optional[Tensor],
                facts: List[Tensor],
                entity_embeddings: Tensor) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        res_sp, res_po = self.r_forward(rel, arg1, arg2, facts, entity_embeddings, depth=self.depth)
        return res_sp, res_po

    def r_forward(self,
                  rel: Tensor, arg1: Optional[Tensor], arg2: Optional[Tensor],   # 22x50(target,r'), 22x50(target,s'), None
                  facts: List[Tensor],   # [2x50 2x50 2x50] story[r, s, o] emb
                  entity_embeddings: Tensor,   # 3x50 story nodes(names) ascending
                  depth: int) -> Tuple[Optional[Tensor], Optional[Tensor]]:   # 0
        res_sp, res_po = None, None
        for d in range(depth + 1):   # 0   depth=0
            scores_sp, scores_po = self.depth_r_forward(rel, arg1, arg2, facts, entity_embeddings, depth=d)   # torch.Size([22, 3]), None
            res_sp = scores_sp if res_sp is None else torch.max(res_sp, scores_sp)   # torch.Size([22, 3])
            res_po = scores_po if res_po is None else torch.max(res_po, scores_po)   # None
        return res_sp, res_po

    def depth_r_forward(self,
                        rel: Tensor, arg1: Optional[Tensor], arg2: Optional[Tensor],   # 22x50(target,r'), 22x50(target,s'), None
                        facts: List[Tensor],   # [2x50 2x50 2x50] story[r, s, o] emb
                        entity_embeddings: Tensor,   # 3x50 story nodes(names) ascending
                        depth: int) -> Tuple[Optional[Tensor], Optional[Tensor]]:   # 0
        batch_size, embedding_size = rel.shape[0], rel.shape[1]   # 22, 50

        if depth == 0:
            return self.model.forward(rel, arg1, arg2, facts, entity_embeddings)

        global_scores_sp = global_scores_po = None

        mask = None
        new_hops_lst = self.hops_lst

        if self.R is not None:
            batch_rules_scores = torch.cat([h.prior(rel).view(-1, 1) for h, _ in self.hops_lst], 1)
            topk, indices = torch.topk(batch_rules_scores, self.R)

            # [R x E]
            rule_heads = torch.cat([h.head for h, _ in self.hops_lst], dim=0)
            rule_body1s = torch.cat([h.memory_lst[0] for h, _ in self.hops_lst], dim=0)
            rule_body2s = torch.cat([h.memory_lst[1] for h, _ in self.hops_lst], dim=0)

            kernel = self.hops_lst[0][0].kernel
            new_rule_heads = F.embedding(indices, rule_heads)
            new_rule_body1s = F.embedding(indices, rule_body1s)
            new_rule_body2s = F.embedding(indices, rule_body2s)

            # print(new_rule_heads.shape[1], self.R)
            assert new_rule_heads.shape[1] == self.R

            new_hops_lst = []
            for i in range(new_rule_heads.shape[1]):
                r = GNTPReformulator(kernel=kernel, head=new_rule_heads[:, i, :],
                                     body=[new_rule_body1s[:, i, :], new_rule_body2s[:, i, :]])
                new_hops_lst += [(r, False)]

        # for rule_idx, (hop_generators, is_reversed) in enumerate(self.hops_lst):
        for rule_idx, (hop_generators, is_reversed) in enumerate(new_hops_lst):
            scores_sp = scores_po = None
            hop_rel_lst = hop_generators(rel)
            nb_hops = len(hop_rel_lst)

            if arg1 is not None:
                sources, scores = arg1, None

                # XXX
                prior = hop_generators.prior(rel)
                if prior is not None:

                    if mask is not None:
                        prior = prior * mask[:, rule_idx]
                        if (prior != 0.0).sum() == 0:
                            continue

                    scores = prior

                for hop_idx, hop_rel in enumerate(hop_rel_lst, start=1):
                    # [B * S, E]
                    sources_2d = sources.view(-1, embedding_size)
                    nb_sources = sources_2d.shape[0]

                    nb_branches = nb_sources // batch_size

                    hop_rel_3d = hop_rel.view(-1, 1, embedding_size).repeat(1, nb_branches, 1)
                    hop_rel_2d = hop_rel_3d.view(-1, embedding_size)

                    if hop_idx < nb_hops:
                        # [B * S, K], [B * S, K, E]
                        if is_reversed:
                            z_scores, z_emb = self.r_hop(hop_rel_2d, None, sources_2d,
                                                         facts, entity_embeddings, depth=depth - 1)
                        else:
                            z_scores, z_emb = self.r_hop(hop_rel_2d, sources_2d, None,
                                                         facts, entity_embeddings, depth=depth - 1)
                        k = z_emb.shape[1]

                        # [B * S * K]
                        z_scores_1d = z_scores.view(-1)
                        # [B * S * K, E]
                        z_emb_2d = z_emb.view(-1, embedding_size)

                        # [B * S * K, E]
                        sources = z_emb_2d
                        # [B * S * K]
                        scores = z_scores_1d if scores is None \
                            else self._tnorm(z_scores_1d, scores.view(-1, 1).repeat(1, k).view(-1))
                    else:
                        # [B * S, N]
                        if is_reversed:
                            _, scores_sp = self.r_forward(hop_rel_2d, None, sources_2d,
                                                          facts, entity_embeddings, depth=depth - 1)
                        else:
                            scores_sp, _ = self.r_forward(hop_rel_2d, sources_2d, None,
                                                          facts, entity_embeddings, depth=depth - 1)

                        nb_entities = scores_sp.shape[1]

                        if scores is not None:
                            scores = scores.view(-1, 1).repeat(1, nb_entities)
                            scores_sp = self._tnorm(scores, scores_sp)

                            # [B, S, N]
                            scores_sp = scores_sp.view(batch_size, -1, nb_entities)
                            # [B, N]
                            scores_sp, _ = torch.max(scores_sp, dim=1)

            if arg2 is not None:
                sources, scores = arg2, None

                # XXX
                prior = hop_generators.prior(rel)
                if prior is not None:
                    scores = prior
                # scores = hop_generators.prior(rel)

                for hop_idx, hop_rel in enumerate(reversed([h for h in hop_rel_lst]), start=1):
                    # [B * S, E]
                    sources_2d = sources.view(-1, embedding_size)
                    nb_sources = sources_2d.shape[0]

                    nb_branches = nb_sources // batch_size

                    hop_rel_3d = hop_rel.view(-1, 1, embedding_size).repeat(1, nb_branches, 1)
                    hop_rel_2d = hop_rel_3d.view(-1, embedding_size)

                    if hop_idx < nb_hops:
                        # [B * S, K], [B * S, K, E]
                        if is_reversed:
                            z_scores, z_emb = self.r_hop(hop_rel_2d, sources_2d, None,
                                                         facts, entity_embeddings, depth=depth - 1)
                        else:
                            z_scores, z_emb = self.r_hop(hop_rel_2d, None, sources_2d,
                                                         facts, entity_embeddings, depth=depth - 1)
                        k = z_emb.shape[1]

                        # [B * S * K]
                        z_scores_1d = z_scores.view(-1)
                        # [B * S * K, E]
                        z_emb_2d = z_emb.view(-1, embedding_size)

                        # [B * S * K, E]
                        sources = z_emb_2d
                        # [B * S * K]
                        scores = z_scores_1d if scores is None \
                            else self._tnorm(z_scores_1d, scores.view(-1, 1).repeat(1, k).view(-1))
                    else:
                        # [B * S, N]
                        if is_reversed:
                            scores_po, _ = self.r_forward(hop_rel_2d, sources_2d, None,
                                                          facts, entity_embeddings, depth=depth - 1)
                        else:
                            _, scores_po = self.r_forward(hop_rel_2d, None, sources_2d,
                                                          facts, entity_embeddings, depth=depth - 1)

                        nb_entities = scores_po.shape[1]

                        if scores is not None:
                            scores = scores.view(-1, 1).repeat(1, nb_entities)
                            scores_po = self._tnorm(scores, scores_po)

                            # [B, S, N]
                            scores_po = scores_po.view(batch_size, -1, nb_entities)
                            # [B, N]
                            scores_po, _ = torch.max(scores_po, dim=1)

            if scores_sp is None and scores_po is None:
                scores_sp, scores_po = self.model.forward(rel, arg1, arg2, facts, entity_embeddings)

            global_scores_sp = scores_sp if global_scores_sp is None else torch.max(global_scores_sp, scores_sp)
            global_scores_po = scores_po if global_scores_po is None else torch.max(global_scores_po, scores_po)

        if global_scores_sp is None and global_scores_po is None:
            global_scores_sp, global_scores_po = self.model.forward(rel, arg1, arg2, facts, entity_embeddings)

        return global_scores_sp, global_scores_po

    def factor(self,
               embedding_vector: Tensor) -> Tensor:
        return self.model.factor(embedding_vector)

    def extra_factors(self,
                      rel: Tensor,
                      arg1: Optional[Tensor],
                      arg2: Optional[Tensor]) -> List[Tensor]:
        return [hop_generator(rel) for hop_generators in self.hops_lst for hop_generator in hop_generators]


class CtpEncoder(Net):
    def __init__(self, model_config):
        super(CtpEncoder, self).__init__(model_config)

        slope = model_config.encoder.slope
        init_size = model_config.encoder.init_size
        init_type = model_config.encoder.init_type
        scoring_type = model_config.encoder.scoring_type
        hops_str = model_config.encoder.hops_str   # ['2', '2', '2', '2']
        k_max = model_config.encoder.k_max
        max_depth = model_config.encoder.max_depth
        tnorm_name = model_config.encoder.tnorm_name
        gntp_R = None

        kernel = GaussianKernel(slope=slope)  # slope=1.0
        self.embedding = torch.nn.Embedding(num_embeddings=self.model_config.unique_nodes,
                                            embedding_dim=self.model_config.embedding.dim)  # num_embeddings=len(self.model_config.entity_lst)    max_norm=1
        # torch.nn.init.uniform_(self.embedding.weight, -1.0, 1.0)
        # self.embedding.requires_grad = False  # prevent changing of weight
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        self.embedding.weight.data *= init_size  # init_size=1.0

        self.edge_embedding = torch.nn.Embedding(model_config.target_size, model_config.graph.edge_dim)
        # if init_type in {'uniform'}:
        #     torch.nn.init.uniform_(self.edge_embedding.weight, -1.0, 1.0)
        # self.edge_embedding.weight.data *= init_size  # init_size=1.0
        torch.nn.init.xavier_uniform_(self.edge_embedding.weight)
        model = NeuralKB(kernel=kernel, scoring_type=scoring_type)  # kernel=GaussianKernel()  scoring_type='concat'
        self.memory = None

        encoder_factory = Encoder_hop()
        hops_lst = [encoder_factory.make_hop(s, model_config, self.edge_embedding, kernel) for s in hops_str]
        # hops_str=['2', '2', '2', '2', '2', '2', '2', '2', '2', '2']
        self.hoppy = Hoppy(model=model, k=k_max, depth=max_depth, tnorm_name=tnorm_name, hops_lst=hops_lst, R=gntp_R)


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
        scores = torch.zeros(len(r), self.model_config.target_size)   # Bx22
        # print(len(r))
        for i in range(len(r)):
            # print(r[i])
            s_emb = self.embedding(s[i].long()).squeeze(1)
            r_emb = self.edge_embedding(r[i].long()).squeeze(1)
            o_emb = self.embedding(o[i].long()).squeeze(1)
            facts = [r_emb, s_emb, o_emb]   # list:3   [2x50 2x50 2x50]   即[r, s, o]的emb
            node_lst = [node for node in range(batch.geo_slices[i])]  # [0, 1, 2]   Todo: the name/idx of the nodes
            node_lst = torch.from_numpy(np.array(node_lst, dtype=np.int64))
            embeddings = self.embedding(node_lst)  # node_name(ascending) 3x50

            query = self.embedding(batch.query_edge.squeeze(1)[i,:].squeeze(0))  # 2x50
            arg1_emb = query[0,:].repeat([self.model_config.target_size, 1])  # 22x50
            arg2_emb = query[1,:].repeat([self.model_config.target_size, 1])  # 22x50
            p_relation_lst = [rela for rela in range(self.model_config.target_size)]  # [0, 1, ..., 21]
            p_relation_lst = torch.from_numpy(np.array(p_relation_lst, dtype=np.int64))
            rel_emb = self.edge_embedding(p_relation_lst)   # 22x50
            scores[i, :] = self.hoppy.score(rel_emb, arg1_emb, arg2_emb, facts, embeddings)   # 22

        return scores, None

class Decoder(Net):
    """
    Compute the graph state with the query
    """
    def __init__(self, model_config):
        super(Decoder, self).__init__(model_config)
        pass

    def calculate_query(self, batch):
        return batch.encoder_hidden

    def forward(self, batch, step_batch):
        return batch.encoder_outputs, None, None

class Encoder_hop:
    def __init__(self):
        super().__init__()

    @staticmethod
    def make_hop(s: str, model_config, relation_embeddings, kernel, memory=None) -> Tuple[BaseReformulator, bool]:   # i.e. '2'
        if s.isdigit():
            nb_hops, is_reversed = int(s), False   # nb_hops=2, is_reversed=False
        else:
            nb_hops, is_reversed = int(s[:-1]), True

        model_name = model_config.encoder.model_name   # ctp_s, ctp_l, ctp_a, ctp_m, ntp
        if model_name == 'ntp':
            reformulator_name = model_name
        else:
            reformulator_name = model_name.split('_')[1]

        nb_rules = model_config.encoder.hidden_dim
        embedding_size= model_config.embedding.dim
        ref_init_type = 'random'
        res = None
        if reformulator_name in {'s'}:  # static
            res = StaticReformulator(nb_hops, embedding_size, init_name=ref_init_type)
        elif reformulator_name in {'l'}:   # linear
            res = LinearReformulator(nb_hops, embedding_size, init_name=ref_init_type)
        elif reformulator_name in {'a'}:   # attentive
            res = AttentiveReformulator(nb_hops, relation_embeddings, init_name=ref_init_type)
        elif reformulator_name in {'m'}:   # memory
            if memory is None:
                memory = MemoryReformulator.Memory(nb_hops, nb_rules, embedding_size, init_name=ref_init_type)
                # nb_hops=2, nb_rules=512, embedding_size=50, init_name='random'
                # memory = ParameterList(
                #     (0): Parameter containing: [torch.FloatTensor of size 512x50]
                #     (1): Parameter containing: [torch.FloatTensor of size 512x50]
                # )
            res = MemoryReformulator(memory)
            # MemoryReformulator(
            #     (memory): Memory(
            #         (memory): ParameterList(
            #             (0): Parameter containing: [torch.FloatTensor of size 512x50]
            #             (1): Parameter containing: [torch.FloatTensor of size 512x50]
            #         )
            #     )
            #     (projection): Linear(in_features=50, out_features=512, bias=True)
            # )
        elif reformulator_name in {'ntp'}:
            res = NTPReformulator(nb_hops=nb_hops, embedding_size=embedding_size,
                                  kernel=kernel, init_name=ref_init_type)
        assert res is not None
        return res, is_reversed   # is_reversed=False
