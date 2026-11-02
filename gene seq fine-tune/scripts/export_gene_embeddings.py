from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, BertForPreTraining


def resolve_annotations(requested: Path | None, omics_root: Path) -> Path:
    if requested is not None:
        if not requested.is_file():
            raise FileNotFoundError(f"Annotation file not found: {requested}")
        return requested

    candidates = [
        omics_root / "gene_go_annotations" / "human_gene_go.tsv",
        Path("/content/GoBERT/gene_go_annotations/human_gene_go.tsv"),
        Path("/content/GoBERT/data/human_gene_go.tsv"),
    ]
    for path in candidates:
        if path.is_file():
            return path

    annotation_dir = omics_root / "gene_go_annotations"
    tsv_files = sorted(annotation_dir.glob("*.tsv")) if annotation_dir.is_dir() else []
    if len(tsv_files) == 1:
        return tsv_files[0]
    if len(tsv_files) > 1:
        choices = "\n  ".join(str(path) for path in tsv_files)
        raise FileNotFoundError(
            "Found multiple TSV files under the Drive annotation folder. "
            f"Pass the intended one with --annotations:\n  {choices}"
        )
    raise FileNotFoundError(
        "Could not find human_gene_go.tsv. In Colab, first run "
        "`from google.colab import drive; drive.mount('/content/drive')`. "
        f"Expected it at: {omics_root / 'gene_go_annotations' / 'human_gene_go.tsv'}. "
        "You can also pass its full path with --annotations."
    )


def read_gene_go(annotations: Path):
    gene_terms: dict[str, set[str]] = {}
    gene_symbols: dict[str, str] = {}
    with annotations.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            qualifiers = set(filter(None, row["qualifier"].split("|")))
            if "NOT" in qualifiers:
                continue
            gene_id = str(row["entrez_gene_id"])
            gene_terms.setdefault(gene_id, set()).add(row["go_id"])
            gene_symbols.setdefault(gene_id, row["gene_symbol"])
    return gene_terms, gene_symbols


def main():
    parser = argparse.ArgumentParser(description="Export per-GO-term GoBERT vectors for each human gene.")
    parser.add_argument("--annotations", type=Path, default=None,
                        help="Annotation TSV; defaults to MyDrive/OMICS/gene_go_annotations/human_gene_go.tsv")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output .npz; defaults to MyDrive/OMICS/gene seq/human_gene_gobert_embeddings.npz")
    parser.add_argument("--omics-root", type=Path, default=Path("/content/drive/MyDrive/OMICS"),
                        help="OMICS folder in mounted Google Drive (Colab default)")
    parser.add_argument("--model-name", default="MM-YY-WW/GoBERT")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--embedding-dtype", choices=["float16", "float32"], default="float16",
                        help="float16 halves archive size; values are contextual final-layer vectors")
    args = parser.parse_args()

    args.annotations = resolve_annotations(args.annotations, args.omics_root)
    if args.output is None:
        args.output = args.omics_root / "gene seq" / "human_gene_gobert_embeddings.npz"
    print(f"Annotations: {args.annotations}")
    print(f"Output archive: {args.output}")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    gene_terms, gene_symbols = read_gene_go(args.annotations)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False, trust_remote_code=True)
    model = BertForPreTraining.from_pretrained(args.model_name).to(device)
    model.eval()

    # GO identifiers are already atomic vocabulary entries in GoBERT. Map them
    # directly instead of calling WordPiece.encode(), which requires an [UNK]
    # vocabulary entry that this checkpoint's vocabulary does not provide.
    vocab = tokenizer.get_vocab()
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id
    if cls_id is None or sep_id is None or pad_id is None:
        raise ValueError(
            "GoBERT tokenizer must define CLS, SEP, and PAD token IDs. "
            f"Got cls={cls_id}, sep={sep_id}, pad={pad_id}."
        )
    embedding_rows = model.get_input_embeddings().num_embeddings
    if any(token_id >= embedding_rows for token_id in vocab.values()):
        raise ValueError(
            f"Tokenizer has token IDs outside the model embedding table ({embedding_rows} rows). "
            "Load the tokenizer and model from the same GoBERT checkpoint."
        )
    if args.max_length < 3:
        raise ValueError("--max-length must be at least 3 to fit CLS, one GO token, and SEP")

    sequences, sequence_go_ids, gene_ids, symbols, retained_counts = [], [], [], [], []
    genes_dropped_no_vocab_terms = 0
    for gene_id in sorted(gene_terms, key=lambda value: int(value)):
        terms = sorted(gene_terms[gene_id])
        recognized = [term for term in terms if term in vocab]
        if not recognized:
            genes_dropped_no_vocab_terms += 1
            continue
        gene_ids.append(gene_id)
        symbols.append(gene_symbols[gene_id])
        sequence_go_ids.append(recognized)
        sequences.append([vocab[term] for term in recognized])
        retained_counts.append(len(recognized))
    if not sequences:
        raise RuntimeError("No gene GO terms matched the GoBERT tokenizer vocabulary")

    term_vectors = []
    term_go_ids = []
    offsets = [0]
    truncated_genes = 0
    term_limit = args.max_length - 2
    for start in tqdm(range(0, len(sequences), args.batch_size), desc="Encoding genes"):
        batch_terms = sequences[start:start + args.batch_size]
        batch_go_ids = sequence_go_ids[start:start + args.batch_size]
        clipped_ids = [seq[:term_limit] for seq in batch_terms]
        clipped_go_ids = [seq[:term_limit] for seq in batch_go_ids]
        truncated_genes += sum(len(seq) > term_limit for seq in batch_terms)
        batch_len = max(len(seq) for seq in clipped_ids) + 2
        input_ids = torch.full((len(clipped_ids), batch_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(clipped_ids), batch_len), dtype=torch.long)
        for row, seq in enumerate(clipped_ids):
            full_seq = [cls_id, *seq, sep_id]
            input_ids[row, :len(full_seq)] = torch.tensor(full_seq, dtype=torch.long)
            attention_mask[row, :len(full_seq)] = 1
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        with torch.inference_mode():
            output = model(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True, return_dict=True)
            hidden = output.hidden_states[-1]
            for row, (go_ids, seq) in enumerate(zip(clipped_go_ids, clipped_ids)):
                vectors = hidden[row, 1:1 + len(seq)].float().cpu().numpy()
                term_vectors.append(vectors.astype(np.float16 if args.embedding_dtype == "float16" else np.float32))
                term_go_ids.extend(go_ids)
                offsets.append(offsets[-1] + len(go_ids))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.concatenate(term_vectors, axis=0)
    np.savez_compressed(
        args.output,
        entrez_gene_ids=np.asarray(gene_ids, dtype=str),
        gene_symbols=np.asarray(symbols, dtype=str),
        offsets=np.asarray(offsets, dtype=np.int64),
        go_ids=np.asarray(term_go_ids, dtype=str),
        term_embeddings=matrix,
    )
    args.output.with_suffix(".json").write_text(json.dumps({
        "model_name": args.model_name,
        "storage": "packed ragged per-gene GO-token matrices; offsets delimit each gene",
        "embedding_dtype": args.embedding_dtype,
        "max_length": args.max_length,
        "genes_exported": len(gene_ids),
        "genes_without_vocab_terms": genes_dropped_no_vocab_terms,
        "go_term_vectors_exported": int(matrix.shape[0]),
        "unique_go_terms_retained_before_truncation": int(sum(retained_counts)),
        "embedding_dim": int(matrix.shape[1]),
        "truncated_gene_sequences": int(truncated_genes),
    }, indent=2), encoding="utf-8")
    print(f"Saved {len(gene_ids)} genes and {matrix.shape[0]} GO-term vectors, shape={matrix.shape}, to {args.output}")
    print(f"Genes with no recognized GO tokens were skipped: {genes_dropped_no_vocab_terms}")


if __name__ == "__main__":
    main()
