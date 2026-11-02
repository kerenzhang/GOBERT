from __future__ import annotations

import torch
from torch import nn


class GoBERTExpressionFusion(nn.Module):
    """Frozen gene GoBERT vectors, expression gate, mean pooling, and drug fusion."""

    def __init__(self, gene_embeddings, n_drugs: int, n_datasets: int,
                 projection_dim: int = 128, drug_dim: int = 64,
                 dataset_dim: int = 8, dropout: float = 0.2) -> None:
        super().__init__()
        self.register_buffer("gene_embeddings", torch.as_tensor(gene_embeddings).float())
        self.gene_projection = nn.Linear(self.gene_embeddings.shape[1], projection_dim, bias=False)
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

    def forward(self, expression, drug_idx, dataset_idx):
        gene_vectors = self.gene_projection(self.gene_embeddings)
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
