from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd


@dataclass
class DataBundle:
    responses: pd.DataFrame
    expression_by_model: np.ndarray
    model_to_row: dict[str, int]
    gene_ids: list[str]
    gene_symbols: list[str]
    gene_term_embeddings: np.ndarray
    gene_term_to_gene: np.ndarray
    drug_to_idx: dict[str, int]
    dataset_to_idx: dict[str, int]


def load_bundle(expression_path: Path, response_path: Path, embedding_path: Path) -> DataBundle:
    expression_df = pd.read_csv(expression_path)
    if "ModelID" not in expression_df.columns:
        raise ValueError("Expression matrix must have a ModelID column")
    parsed = []
    for column in expression_df.columns[1:]:
        match = re.fullmatch(r"(.+)\s+\((\d+)\)", str(column))
        if match is None:
            raise ValueError(f"Cannot parse gene expression header: {column}")
        parsed.append((match.group(1), match.group(2)))
    expr_gene_ids = [item[1] for item in parsed]
    if len(expr_gene_ids) != len(set(expr_gene_ids)):
        raise ValueError("Duplicate Entrez Gene IDs in expression columns")

    raw = np.load(embedding_path, allow_pickle=False)
    emb_ids = [str(value) for value in raw["entrez_gene_ids"].tolist()]
    if "term_embeddings" in raw.files:
        term_matrix = raw["term_embeddings"]
        offsets = np.asarray(raw["offsets"], dtype=np.int64)
        if term_matrix.ndim != 2 or offsets.shape != (len(emb_ids) + 1,):
            raise ValueError("Packed embedding NPZ must contain term_embeddings [terms, dim] and offsets [genes+1]")
        if offsets[0] != 0 or offsets[-1] != len(term_matrix) or np.any(np.diff(offsets) < 0):
            raise ValueError("Invalid offsets in packed embedding NPZ")
        packed_terms = True
    else:
        legacy_matrix = np.asarray(raw["embeddings"], dtype=np.float32)
        if legacy_matrix.ndim != 2 or len(emb_ids) != legacy_matrix.shape[0]:
            raise ValueError("Embedding NPZ must have matching entrez_gene_ids and embeddings")
        packed_terms = False
    if len(emb_ids) != len(set(emb_ids)):
        raise ValueError("Duplicate Entrez Gene IDs in GoBERT embedding NPZ")
    emb_index = {gene_id: i for i, gene_id in enumerate(emb_ids)}

    keep = [i for i, gene_id in enumerate(expr_gene_ids) if gene_id in emb_index]
    if not keep:
        raise ValueError("No Entrez Gene IDs overlap between expression and GoBERT embeddings")
    gene_ids = [expr_gene_ids[i] for i in keep]
    gene_symbols = [parsed[i][0] for i in keep]
    expression = expression_df.iloc[:, [i + 1 for i in keep]].to_numpy(dtype=np.float32)
    if packed_terms:
        term_chunks = []
        term_gene_chunks = []
        for gene_idx, gene_id in enumerate(gene_ids):
            source_idx = emb_index[gene_id]
            left, right = offsets[source_idx:source_idx + 2]
            if right <= left:
                raise ValueError(f"Gene {gene_id} has no stored GO-term vectors")
            term_chunks.append(np.asarray(term_matrix[left:right], dtype=np.float32))
            term_gene_chunks.append(np.full(right - left, gene_idx, dtype=np.int64))
        gene_term_embeddings = np.concatenate(term_chunks, axis=0)
        gene_term_to_gene = np.concatenate(term_gene_chunks)
    else:
        # Legacy archives contain one pre-pooled vector per gene. Treat each
        # vector as a one-term sequence so the same model remains usable.
        gene_term_embeddings = legacy_matrix[[emb_index[gene_id] for gene_id in gene_ids]]
        gene_term_to_gene = np.arange(len(gene_ids), dtype=np.int64)

    model_ids = expression_df["ModelID"].astype(str).tolist()
    if len(model_ids) != len(set(model_ids)):
        raise ValueError("Duplicate ModelID rows in expression matrix")
    model_to_row = {model_id: i for i, model_id in enumerate(model_ids)}
    responses = pd.read_csv(response_path)
    required = {"ModelID", "DRUG_ID", "DRUG_NAME", "DATASET", "AUC_PUBLISHED", "IC50_PUBLISHED"}
    missing = required - set(responses.columns)
    if missing:
        raise ValueError(f"Response table is missing columns: {sorted(missing)}")
    responses["ModelID"] = responses["ModelID"].astype(str)
    responses["DRUG_ID"] = responses["DRUG_ID"].astype(str)
    responses["DATASET"] = responses["DATASET"].astype(str)
    responses["AUC_PUBLISHED"] = pd.to_numeric(responses["AUC_PUBLISHED"], errors="coerce")
    responses["IC50_PUBLISHED"] = pd.to_numeric(responses["IC50_PUBLISHED"], errors="coerce")
    responses = responses.dropna(subset=["AUC_PUBLISHED", "IC50_PUBLISHED"])
    responses = responses[responses["IC50_PUBLISHED"] > 0]
    responses = responses[responses["ModelID"].isin(model_to_row)].copy().reset_index(drop=True)
    if responses.empty:
        raise ValueError("No response samples remain after matching to expression rows")
    responses["log2_IC50_PUBLISHED"] = np.log2(responses["IC50_PUBLISHED"].to_numpy(dtype=float))
    responses.insert(0, "sample_id", [f"S{i:06d}" for i in range(len(responses))])
    drugs = sorted(responses["DRUG_ID"].unique())
    datasets = sorted(responses["DATASET"].unique())
    return DataBundle(
        responses, expression, model_to_row, gene_ids, gene_symbols,
        gene_term_embeddings, gene_term_to_gene,
        {value: i for i, value in enumerate(drugs)},
        {value: i for i, value in enumerate(datasets)},
    )
