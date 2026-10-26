import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from mykan import KANLayer, KANNetwork, parse_bases

class InformationExchange(nn.Module):
    def __init__(self, dim, mode, bases, basis_size, spline_order):
        super().__init__()
        self.mode = mode
        if mode == 'attention':
            self.scores = nn.ModuleList(KANLayer(dim, 1, bases, basis_size, spline_order) for _ in range(4))
        else:
            self.gates = nn.ModuleList(KANLayer(2 * dim, dim, bases, basis_size, spline_order) for _ in range(2))

    def forward(self, v, e):
        if self.mode == 'attention':
            v_weights = torch.cat((self.scores[0](v), self.scores[1](e)), dim=-1).softmax(dim=-1)
            e_weights = torch.cat((self.scores[2](e), self.scores[3](v)), dim=-1).softmax(dim=-1)
            return v_weights[:, :1] * v + v_weights[:, 1:] * e, e_weights[:, :1] * e + e_weights[:, 1:] * v
        inputs = torch.cat((v, e), dim=-1)
        g1 = self.gates[0](inputs).sigmoid()
        g2 = self.gates[1](inputs).sigmoid()
        return g1 * v + (1.0 - g1) * e, g2 * e + (1.0 - g2) * v

class KnowledgeKan(nn.Module):
    def __init__(self, args, n_user, num_entity, num_relation, cf_graph, kg_graph):
        super().__init__()
        self.n_user = n_user
        self.n_item = cf_graph.shape[0] - n_user
        self.n_entity = num_entity
        self.n_relation = num_relation
        self.emb_size = args.dim
        self.n_layers = getattr(args, 'n_layers', 2)
        self.n_ki_layers = getattr(args, 'n_ki_layers', 2)
        self.epsilon = getattr(args, 'epsilon', 0.5)
        self.temperature = getattr(args, 'attention_temperature', 1.0)
        self.graph_batch_size = getattr(args, 'graph_batch_size', 2048)
        self.graph_checkpoint = getattr(args, 'graph_checkpoint', True)
        self.ieu_mode = getattr(args, 'ieu', 'none')
        bases = parse_bases(getattr(args, 'bases', None) or args.KAN_model)
        basis_size = getattr(args, 'basis_size', 5)
        spline_order = getattr(args, 'spline_order', 3)
        self.register_buffer('cf_graph', cf_graph.long())
        self.register_buffer('kg_neighbors', torch.where(kg_graph[0] >= 0, kg_graph[0] - n_user, -1).long())
        self.register_buffer('kg_relations', kg_graph[1].long())
        self.register_buffer('context_graph', kg_graph[2])
        self.emb = nn.Parameter(torch.empty(n_user + num_entity, self.emb_size))
        self.rel = nn.Parameter(torch.empty(num_relation, self.emb_size))
        nn.init.xavier_uniform_(self.emb)
        nn.init.xavier_uniform_(self.rel)
        self.kan_CF = nn.ModuleList(KANLayer(self.emb_size, self.emb_size, bases, basis_size, spline_order) for _ in range(self.n_layers))
        self.kan_KG = nn.ModuleList(KANLayer(self.emb_size, self.emb_size, bases, basis_size, spline_order) for _ in range(self.n_layers))
        self.kan_att = nn.ModuleList(KANNetwork([4 * self.emb_size, 2 * self.emb_size, 1], bases, basis_size, spline_order) for _ in range(self.n_layers))
        self.kan_preF = KANLayer(2 * self.emb_size, 2 * self.emb_size, bases, basis_size, spline_order)
        self.ki_weights = nn.ParameterList(nn.Parameter(torch.empty(2 * self.emb_size, 2 * self.emb_size)) for _ in range(self.n_ki_layers))
        for weight in self.ki_weights:
            nn.init.xavier_uniform_(weight)
        prediction_dim = 2 * self.emb_size * (self.n_ki_layers + 1)
        self.kan_pre = KANNetwork([prediction_dim, prediction_dim // 2, 1], bases, basis_size, spline_order)
        self.exchange = nn.ModuleList()
        if self.ieu_mode != 'none':
            self.exchange.extend(InformationExchange(self.emb_size, self.ieu_mode, bases, basis_size, spline_order) for _ in range(self.n_layers - 1))
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

    def _interaction_step(self, embeddings, start, stop, layer):
        neighbors = self.cf_graph[start:stop]
        mask = neighbors >= 0
        values = embeddings[neighbors.clamp_min(0)] * mask.unsqueeze(-1)
        aggregate = values.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        return self.kan_CF[layer]((1.0 + self.epsilon) * embeddings[start:stop] + aggregate)

    def _knowledge_step(self, embeddings, context, start, stop, layer):
        neighbors = self.kg_neighbors[start:stop]
        mask = neighbors >= 0
        values = embeddings[neighbors.clamp_min(0)]
        relation_table = torch.cat((self.rel, self.rel.new_zeros(1, self.emb_size)), dim=0)
        relation_ids = self.kg_relations[start:stop]
        relations = relation_table[torch.where(relation_ids >= 0, relation_ids, self.n_relation)]
        centers = embeddings[start:stop].unsqueeze(1).expand_as(values)
        contexts = context[start:stop].unsqueeze(1).expand_as(values)
        inputs = torch.cat((centers, relations, values, contexts), dim=-1)
        scores = self.kan_att[layer](inputs).squeeze(-1) / self.temperature
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=1) * mask
        aggregate = (weights.unsqueeze(-1) * values).sum(dim=1)
        return self.kan_KG[layer]((1.0 + self.epsilon) * embeddings[start:stop] + aggregate)

    def _propagate(self, function, embeddings, layer, context=None):
        outputs = []
        for start in range(0, embeddings.shape[0], self.graph_batch_size):
            stop = min(start + self.graph_batch_size, embeddings.shape[0])
            if context is None:
                def step(x, start=start, stop=stop, layer=layer):
                    return function(x, start, stop, layer)
                inputs = (embeddings,)
            else:
                def step(x, z, start=start, stop=stop, layer=layer):
                    return function(x, z, start, stop, layer)
                inputs = (embeddings, context)
            if self.training and self.graph_checkpoint and torch.is_grad_enabled():
                outputs.append(checkpoint(step, *inputs, use_reentrant=False))
            else:
                outputs.append(step(*inputs))
        return torch.cat(outputs, dim=0)

    def encode_graphs(self):
        interaction = self.emb[:self.n_user + self.n_item]
        knowledge = self.emb[self.n_user:]
        interaction_layers = [interaction]
        knowledge_layers = [knowledge]
        for layer in range(self.n_layers):
            if layer > 0 and self.ieu_mode != 'none':
                v, e = self.exchange[layer - 1](interaction[self.n_user:], knowledge[:self.n_item])
                interaction = torch.cat((interaction[:self.n_user], v), dim=0)
                knowledge = torch.cat((e, knowledge[self.n_item:]), dim=0)
            item_context = interaction[self.n_user:]
            nonitem_context = torch.sparse.mm(self.context_graph, item_context)
            context = torch.cat((item_context, nonitem_context[self.n_item:]), dim=0)
            next_interaction = self._propagate(self._interaction_step, interaction, layer)
            next_knowledge = self._propagate(self._knowledge_step, knowledge, layer, context)
            interaction_layers.append(next_interaction)
            knowledge_layers.append(next_knowledge)
            interaction = next_interaction
            knowledge = next_knowledge
        return torch.stack(interaction_layers).mean(dim=0), torch.stack(knowledge_layers).mean(dim=0)

    def graph_embeddings(self):
        if self.training:
            return self.encode_graphs()
        versions = tuple((id(tensor), tensor._version) for tensor in (*self.parameters(), *self.buffers()))
        if self._graph_cache is None or versions != self._cache_versions:
            with torch.no_grad():
                self._graph_cache = self.encode_graphs()
            self._cache_versions = versions
        return self._graph_cache

    def get_rating(self, user_emb, item_emb, return_logits=False):
        x = self.kan_preF(torch.cat((user_emb, item_emb), dim=-1))
        features = [x]
        for weight in self.ki_weights:
            x = x * F.linear(x, weight)
            features.append(x)
        logits = self.kan_pre(torch.cat(features, dim=-1))
        return logits if return_logits else logits.sigmoid()

    def forward(self, u_id, i_id, return_logits=False):
        device = self.emb.device
        u_id = u_id.to(device=device, dtype=torch.long)
        i_id = i_id.to(device=device, dtype=torch.long)
        interaction, knowledge = self.graph_embeddings()
        user_emb = interaction[u_id]
        item_emb = interaction[i_id] + knowledge[i_id - self.n_user]
        return self.get_rating(user_emb, item_emb, return_logits), user_emb, item_emb
