from __future__ import annotations

import torch
from torch import nn


class GoBERTExpressionFusion(nn.Module):
    """Learned GO-term pooling, expression-gated gene pooling, and drug fusion."""

    def __init__(self, gene_term_embeddings, gene_term_to_gene, n_genes: int,
                 n_drugs: int, n_datasets: int,
                 projection_dim: int = 128, drug_dim: int = 64,
                 dataset_dim: int = 8, dropout: float = 0.2,
                 go_pooling: str = "attention") -> None:
        super().__init__()
        if go_pooling not in {"attention", "mean"}:
            raise ValueError("go_pooling must be 'attention' or 'mean'")
        self.go_pooling = go_pooling
        self.n_genes = int(n_genes)
        self.register_buffer("gene_term_embeddings", torch.as_tensor(gene_term_embeddings).float())
        self.register_buffer("gene_term_to_gene", torch.as_tensor(gene_term_to_gene).long())
        embedding_dim = self.gene_term_embeddings.shape[1]
        self.go_term_norm = nn.LayerNorm(embedding_dim)
        self.go_term_attention = nn.Sequential(
            nn.Linear(embedding_dim, max(32, embedding_dim // 4)),
            nn.Tanh(),
            nn.Linear(max(32, embedding_dim // 4), 1, bias=False),
        )
        self.gene_projection = nn.Linear(embedding_dim, projection_dim, bias=False)
        self.embedding_gate = nn.Linear(projection_dim, 1, bias=False)
        self.expression_gate = nn.Sequential(nn.Linear(1, 16), nn.GELU(), nn.Linear(16, 1))
        self.drug_embedding = nn.Embedding(n_drugs, drug_dim)
        self.drug_projection = nn.Linear(drug_dim, projection_dim, bias=False)
        self.dataset_embedding = nn.Embedding(n_datasets, dataset_dim)
        fused_dim = projection_dim + drug_dim + projection_dim + dataset_dim
        self.response_head = nn.Sequential(
            nn.LayerNorm(fused_dim), nn.Linear(fused_dim, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 2),
        )

    def pool_gene_terms(self):
        """Pool packed, variable-length GO-term vectors into one vector per gene."""
        terms = self.gene_term_embeddings
        term_gene = self.gene_term_to_gene
        pooled = terms.new_zeros((self.n_genes, terms.shape[1]))
        if self.go_pooling == "attention":
            scores = self.go_term_attention(self.go_term_norm(terms)).squeeze(-1)
            max_score = scores.new_full((self.n_genes,), -torch.inf)
            max_score.scatter_reduce_(0, term_gene, scores, reduce="amax", include_self=True)
            weights = torch.exp(scores - max_score[term_gene])
            normalizer = scores.new_zeros((self.n_genes,))
            normalizer.scatter_add_(0, term_gene, weights)
            weights = weights / normalizer[term_gene].clamp_min(torch.finfo(weights.dtype).tiny)
            pooled.index_add_(0, term_gene, terms * weights.unsqueeze(-1))
        else:
            pooled.index_add_(0, term_gene, terms)
            counts = terms.new_zeros((self.n_genes,))
            counts.scatter_add_(0, term_gene, terms.new_ones((len(term_gene),)))
            pooled = pooled / counts.clamp_min(1).unsqueeze(-1)
        return pooled

    def forward(self, expression, drug_idx, dataset_idx):
        pooled_gene_terms = self.pool_gene_terms()
        gene_vectors = self.gene_projection(pooled_gene_terms)
        expression_score = self.expression_gate(expression.unsqueeze(-1)).squeeze(-1)
        embedding_score = self.embedding_gate(gene_vectors).squeeze(-1).unsqueeze(0)
        gates = torch.sigmoid(expression_score + embedding_score)
        cell_vector = (gates @ gene_vectors) / max(gene_vectors.shape[0], 1)
        drug_vector = self.drug_embedding(drug_idx)
        interaction = cell_vector * self.drug_projection(drug_vector)
        dataset_vector = self.dataset_embedding(dataset_idx)
        return self.response_head(torch.cat([cell_vector, drug_vector, interaction, dataset_vector], dim=-1))


class ExpressionDrugFusion(nn.Module):
    """No-GoBERT comparator using fold-fitted expression PCA plus drug ID."""

    def __init__(self, n_components: int, n_drugs: int, n_datasets: int,
                 hidden_dim: int = 128, drug_dim: int = 64,
                 dataset_dim: int = 8, dropout: float = 0.2) -> None:
        super().__init__()
        self.expression_encoder = nn.Sequential(
            nn.Linear(n_components, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.drug_embedding = nn.Embedding(n_drugs, drug_dim)
        self.drug_projection = nn.Linear(drug_dim, hidden_dim, bias=False)
        self.dataset_embedding = nn.Embedding(n_datasets, dataset_dim)
        fused_dim = hidden_dim + drug_dim + hidden_dim + dataset_dim
        self.response_head = nn.Sequential(
            nn.LayerNorm(fused_dim), nn.Linear(fused_dim, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 2),
        )

    def forward(self, expression_pcs, drug_idx, dataset_idx):
        cell_vector = self.expression_encoder(expression_pcs)
        drug_vector = self.drug_embedding(drug_idx)
        interaction = cell_vector * self.drug_projection(drug_vector)
        dataset_vector = self.dataset_embedding(dataset_idx)
        return self.response_head(torch.cat([cell_vector, drug_vector, interaction, dataset_vector], dim=-1))
