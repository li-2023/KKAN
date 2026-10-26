import torch
from torch import nn

class DeepFM(nn.Module):
    def __init__(self, args, n_user, num_entity, num_relation, cf_graph, kg_graph):
        super().__init__()
        self.n_user = n_user
        self.n_item = cf_graph.shape[0] - n_user
        self.dim = args.dim
        self.user_embedding = nn.Embedding(n_user, self.dim)
        self.item_embedding = nn.Embedding(self.n_item, self.dim)
        self.entity_embedding = nn.Embedding(num_entity, self.dim)
        self.user_linear = nn.Embedding(n_user, 1)
        self.item_linear = nn.Embedding(self.n_item, 1)
        self.entity_linear = nn.Embedding(num_entity, 1)
        self.bias = nn.Parameter(torch.zeros(()))

        edges = kg_graph[0]
        pairs = torch.unique(edges[:, edges[0] < self.n_item].T, dim=0).T
        degree = torch.bincount(pairs[0], minlength=self.n_item)
        weights = degree[pairs[0]].clamp_min(1).float().reciprocal()
        self.register_buffer('neighbor_mean', torch.sparse_coo_tensor(
            pairs, weights, (self.n_item, num_entity)).coalesce())
        self._context_cache = None
        self._cache_versions = None

        hidden = getattr(args, 'deepfm_hidden', [128, 64, 32])
        dropout = getattr(args, 'deepfm_dropout', 0.5)
        layers = []
        size = 3 * self.dim
        for width in hidden:
            layers.extend((nn.Linear(size, width), nn.ReLU(), nn.Dropout(dropout)))
            size = width
        layers.append(nn.Linear(size, 1))
        self.deep = nn.Sequential(*layers)
        for table in (self.user_embedding, self.item_embedding, self.entity_embedding):
            nn.init.xavier_uniform_(table.weight)
        for table in (self.user_linear, self.item_linear, self.entity_linear):
            nn.init.zeros_(table.weight)

    def train(self, mode=True):
        if mode or mode != self.training:
            self._context_cache = None
            self._cache_versions = None
        return super().train(mode)

    def _apply(self, fn, recurse=True):
        self._context_cache = None
        self._cache_versions = None
        return super()._apply(fn, recurse)

    def kg_context(self):
        def compute():
            return (torch.sparse.mm(self.neighbor_mean, self.entity_embedding.weight),
                    torch.sparse.mm(self.neighbor_mean, self.entity_linear.weight))

        if self.training:
            return compute()
        versions = tuple((id(t), t._version) for t in (*self.parameters(), *self.buffers()))
        if self._context_cache is None or versions != self._cache_versions:
            with torch.no_grad():
                self._context_cache = compute()
            self._cache_versions = versions
        return self._context_cache

    @staticmethod
    def fm_interaction(fields):

        return 0.5 * (fields.sum(dim=1).square() - fields.square().sum(dim=1)).sum(dim=-1)

    def forward(self, u_id, i_id, return_logits=False):
        device = self.user_embedding.weight.device
        u_id = u_id.to(device=device, dtype=torch.long)
        item_id = i_id.to(device=device, dtype=torch.long) - self.n_user
        user = self.user_embedding(u_id)
        item = self.item_embedding(item_id)
        context, context_linear = self.kg_context()
        fields = torch.stack((user, item, context[item_id]), dim=1)
        linear = (self.user_linear(u_id) + self.item_linear(item_id) + context_linear[item_id]).squeeze(-1)
        logits = self.bias + linear + self.fm_interaction(fields) + self.deep(fields.flatten(1)).squeeze(-1)
        return (logits if return_logits else logits.sigmoid()), user, item
