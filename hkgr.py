import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from hypercomplex import ComponentDropout, HypercomplexLinear, cayley_dickson

class HKGRO(nn.Module):
    components = 8

    def __init__(self, args, n_user, num_entity, num_relation, cf_graph, kg_graph):
        super().__init__()
        self.n_user = n_user
        self.n_item = cf_graph.shape[0] - n_user
        self.total_dim = getattr(args, 'hkgr_dim', 64)
        if self.total_dim <= 0 or self.total_dim % self.components:
            raise ValueError()
        self.component_dim = self.total_dim // self.components
        self.n_layers = getattr(args, 'hkgr_layers', 1)
        self.hin_layers = getattr(args, 'hin_layers', 2)
        self.edge_batch_size = getattr(args, 'graph_batch_size', 2048)
        self.graph_checkpoint = getattr(args, 'graph_checkpoint', True)
        self.cl_chunk_size = getattr(args, 'cl_chunk_size', 512)
        self.cl_weight = getattr(args, 'cl_weight', 1e-4)
        self.temperature = getattr(args, 'cl_temperature', 0.1)
        self.reg_weight = getattr(args, 'weight_decay', 1e-6)
        if min(self.n_layers, self.hin_layers, self.edge_batch_size, self.cl_chunk_size) < 1:
            raise ValueError()
        if self.temperature <= 0 or self.cl_weight < 0 or self.reg_weight < 0:
            raise ValueError()

        self.user_embedding = nn.Parameter(torch.empty(n_user, 8, self.component_dim))
        self.item_embedding = nn.Parameter(torch.empty(self.n_item, 8, self.component_dim))
        self.entity_embedding = nn.Parameter(torch.empty(num_entity, 8, self.component_dim))
        self.relation_embedding = nn.Parameter(torch.empty(num_relation, 8, self.component_dim))
        for table in (self.user_embedding, self.item_embedding, self.entity_embedding, self.relation_embedding):
            for component in table.unbind(dim=1):
                nn.init.xavier_uniform_(component)
        self.register_buffer('cf_graph', cf_graph.coalesce())
        self.register_buffer('kg_edges', kg_graph[0])
        self.register_buffer('kg_relations', kg_graph[1])
        attention_input = self.total_dim + self.component_dim
        self.queries = nn.ModuleList(nn.Linear(attention_input, self.total_dim, bias=False)
                                     for _ in range(self.n_layers))
        self.keys = nn.ModuleList(nn.Linear(attention_input, self.total_dim, bias=False)
                                  for _ in range(self.n_layers))
        for projection in (*self.queries, *self.keys):
            nn.init.xavier_uniform_(projection.weight)
        self.alpha_weights = nn.ModuleList(HypercomplexLinear(self.component_dim) for _ in range(self.hin_layers))
        self.beta_weights = nn.ModuleList(HypercomplexLinear(self.component_dim) for _ in range(self.hin_layers))
        self.component_dropout = ComponentDropout(getattr(args, 'component_dropout', 0.1))
        self._graph_cache = None
        self._cache_versions = None

    def train(self, mode=True):
        if mode or mode != self.training:
            self._graph_cache = None
            self._cache_versions = None
        return super().train(mode)

    def _apply(self, fn, recurse=True):
        self._graph_cache = None
        self._cache_versions = None
        return super()._apply(fn, recurse)

    def _edge_scores(self, embeddings, relations, start, stop, layer):
        source, neighbor = self.kg_edges[:, start:stop]
        values = embeddings[neighbor]
        centers = embeddings[source].flatten(1).unsqueeze(1).expand(-1, 8, -1)
        relation = relations[self.kg_relations[start:stop]].flatten(1).unsqueeze(1).expand(-1, 8, -1)
        q = self.queries[layer](torch.cat((values, centers), dim=-1))
        k = self.keys[layer](torch.cat((values, relation), dim=-1))
        return (q * k).sum(dim=-1) / math.sqrt(self.total_dim)

    def knowledge_step(self, embeddings, layer):

        if not self.kg_edges.shape[1]:
            return embeddings * 0
        scores = []
        for start in range(0, self.kg_edges.shape[1], self.edge_batch_size):
            stop = min(start + self.edge_batch_size, self.kg_edges.shape[1])

            def score_chunk(x, r, start=start, stop=stop, layer=layer):
                return self._edge_scores(x, r, start, stop, layer)

            if self.training and self.graph_checkpoint and torch.is_grad_enabled():
                scores.append(checkpoint(score_chunk, embeddings, self.relation_embedding, use_reentrant=False))
            else:
                scores.append(score_chunk(embeddings, self.relation_embedding))
        scores = torch.cat(scores, dim=0)
        indices = self.kg_edges[0, :, None].expand(-1, 8)
        maxima = scores.new_full((embeddings.shape[0], 8), -torch.inf)
        maxima.scatter_reduce_(0, indices, scores.detach(), reduce='amax', include_self=True)
        weights = (scores - maxima[self.kg_edges[0]]).exp()
        denominator = weights.new_zeros((embeddings.shape[0], 8)).scatter_add(0, indices, weights)
        weights = weights / denominator[self.kg_edges[0]].clamp_min(torch.finfo(weights.dtype).tiny)
        output = []
        for component in range(8):
            adjacency = torch.sparse_coo_tensor(self.kg_edges, weights[:, component],
                                                (len(embeddings), len(embeddings))).coalesce()
            output.append(torch.sparse.mm(adjacency, embeddings[:, component]))
        return torch.stack(output, dim=1)

    def encode_graphs(self):
        interaction = torch.cat((self.user_embedding, self.item_embedding), dim=0)
        knowledge = self.entity_embedding
        interaction_sum, knowledge_sum = interaction, knowledge
        for layer in range(self.n_layers):
            interaction = torch.sparse.mm(self.cf_graph, self.component_dropout(interaction).flatten(1))
            interaction = interaction.reshape(-1, 8, self.component_dim)
            knowledge = self.knowledge_step(self.component_dropout(knowledge), layer)
            interaction_sum = interaction_sum + interaction
            knowledge_sum = knowledge_sum + knowledge
        return interaction_sum / (self.n_layers + 1), knowledge_sum / (self.n_layers + 1)

    def graph_embeddings(self):
        if self.training:
            return self.encode_graphs()
        versions = tuple((id(t), t._version) for t in (*self.parameters(), *self.buffers()))
        if self._graph_cache is None or versions != self._cache_versions:
            with torch.no_grad():
                self._graph_cache = self.encode_graphs()
            self._cache_versions = versions
        return self._graph_cache

    def interaction_logits(self, user, item):

        alpha, beta = user, item
        for alpha_weight, beta_weight in zip(self.alpha_weights, self.beta_weights):
            alpha, beta = (alpha_weight(cayley_dickson(alpha, beta)),
                           beta_weight(cayley_dickson(beta, alpha)))
        component_logits = torch.cat((alpha.sum(dim=-1), beta.sum(dim=-1)), dim=-1)

        return (F.logsigmoid(component_logits).logsumexp(dim=-1)
                - F.logsigmoid(-component_logits).logsumexp(dim=-1))

    def _predict(self, u_id, i_id, interaction, knowledge, return_logits):
        device = self.user_embedding.device
        u_id = u_id.to(device=device, dtype=torch.long)
        i_id = i_id.to(device=device, dtype=torch.long)
        user = interaction[u_id]
        item = interaction[i_id] + knowledge[i_id - self.n_user]
        logits = self.interaction_logits(user, item)
        return (logits if return_logits else logits.sigmoid()), user.flatten(1), item.flatten(1)

    def forward(self, u_id, i_id, return_logits=False):
        return self._predict(u_id, i_id, *self.graph_embeddings(), return_logits)

    def contrastive_loss(self, interaction_items, knowledge_items, item_ids=None):

        if item_ids is None:
            item_ids = torch.arange(self.n_item, device=interaction_items.device)
        item_ids = item_ids.unique()
        cf = F.normalize(interaction_items, dim=-1)
        kg = F.normalize(knowledge_items, dim=-1)
        loss = cf.new_zeros(())
        for anchor_start in range(0, len(item_ids), self.cl_chunk_size):
            ids = item_ids[anchor_start:anchor_start + self.cl_chunk_size]
            anchors = cf[ids]
            positive = ((anchors * kg[ids]).sum(dim=-1) / self.temperature).logsumexp(dim=-1)
            denominator = anchors.new_full((len(ids),), -torch.inf)
            for start in range(0, self.n_item, self.cl_chunk_size):
                keys = kg[start:start + self.cl_chunk_size]

                def partition(q, k):
                    scores = torch.einsum('bcd,ncd->bnc', q, k) / self.temperature
                    return scores.flatten(1).logsumexp(dim=-1)

                if self.training and self.graph_checkpoint and torch.is_grad_enabled():
                    part = checkpoint(partition, anchors, keys, use_reentrant=False)
                else:
                    part = partition(anchors, keys)
                denominator = torch.logaddexp(denominator, part)
            loss = loss + (denominator - positive).sum()
        return loss

    def training_loss(self, user, item, labels):
        interaction, knowledge = self.encode_graphs()
        logits, _, _ = self._predict(user, item, interaction, knowledge, True)
        recommendation = F.binary_cross_entropy_with_logits(logits, labels, reduction='sum')
        contrastive = recommendation.new_zeros(())
        if self.cl_weight:
            ids = item.to(device=interaction.device, dtype=torch.long) - self.n_user
            contrastive = self.contrastive_loss(interaction[self.n_user:], knowledge[:self.n_item], ids)
        regularization = recommendation.new_zeros(())
        if self.reg_weight:
            regularization = sum(parameter.square().sum() for parameter in self.parameters())
        total = recommendation + self.cl_weight * contrastive + self.reg_weight * regularization
        return total, {'recommendation': recommendation.detach(), 'contrastive': contrastive.detach(),
                       'regularization': regularization.detach()}
