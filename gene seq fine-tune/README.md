# Gene sequence fine-tuning: expression-gated GoBERT baseline

This package implements the first GoBERT-plus-expression drug-response model discussed in the project. It also trains a matched expression-plus-drug model without GoBERT vectors and a drug-mean baseline.

## Model and data flow

1. `scripts/export_gene_embeddings.py` encodes each human gene's distinct positive GO annotations with the released `MM-YY-WW/GoBERT` checkpoint. It saves the final contextual state for every GO token, preserving a separate (N_g \times 1024) matrix for each Entrez Gene ID. This is a frozen feature export, not end-to-end GoBERT fine-tuning.
2. The training script aligns genes with the breast expression matrix by Entrez Gene ID and mean-pools each gene's saved GO-term matrix for this initial baseline. The individual term vectors remain available in the archive for learned pooling later.
3. `GoBERTExpressionFusion` projects each gene vector to 128 dimensions, gates it using that cell line's standardized expression, and mean-pools the gated gene vectors to a cell-line vector. It fuses that vector with a learned drug-ID embedding and a GDSC dataset embedding, then predicts AUC and log2(IC50).
4. `ExpressionDrugFusion` uses expression-only PCA features with the same drug and dataset inputs. This is the no-GoBERT neural comparison.

Evaluation uses five-fold `GroupKFold` by `ModelID`, with a cell-line-grouped validation subset inside each training fold. GDSC1/GDSC2 rows remain separate measurements and carry a dataset indicator.

## Targets and metrics

- Response targets: `AUC_PUBLISHED` and `log2(IC50_PUBLISHED)`, with log2 computed from positive `IC50_PUBLISHED` values.
- Reports: RMSE, MAE, R², Pearson r, and Spearman rho for AUC, log2(IC50), and back-transformed raw IC50.
- Results include global out-of-fold metrics and macro-averaged per-drug metrics, plus per-drug details. `AUC_PUBLISHED` is the dose-response label, not AUROC.

## Setup

```powershell
python -m pip install -r requirements.txt
```

In Colab, training defaults to the mounted Drive files under `/content/drive/MyDrive/OMICS/gene seq`. Output defaults to this project directory's `results/breast_v1` folder. Override the CLI paths if your filenames or folders differ.

## Run the embedding export in Colab

Mount Drive before running the exporter:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Install the export dependencies if needed:

```python
%pip install transformers tqdm
```

Place `export_gene_embeddings.py` in the Colab runtime (for example, at `/content/GoBERT/scripts/export_gene_embeddings.py`), then run:

```python
!python /content/GoBERT/scripts/export_gene_embeddings.py
```

With no path arguments, it reads `/content/drive/MyDrive/OMICS/gene_go_annotations/human_gene_go.tsv` and saves the archive to `/content/drive/MyDrive/OMICS/gene seq/human_gene_gobert_embeddings.npz`. The exporter also accepts `--annotations`, `--output`, and `--omics-root` if your Drive folder names differ. The GoBERT weights are downloaded from Hugging Face on the first run.

## Step 1: export fixed gene vectors

```powershell
python scripts\export_gene_embeddings.py `
  --annotations "..\GoBERT\gene_go_annotations\human_gene_go.tsv" `
  --output "data\human_gene_gobert_embeddings.npz" `
  --model-name "MM-YY-WW/GoBERT" --batch-size 32 --max-length 512
```

The first run downloads the Hugging Face checkpoint. The packed NPZ contains:

- `entrez_gene_ids` and `gene_symbols`: one entry per gene.
- `term_embeddings`: all GO-term vectors concatenated into one `[total_terms, 1024]` array (float16 by default to reduce file size).
- `offsets`: length `number_of_genes + 1`; gene `i`'s matrix is `term_embeddings[offsets[i]:offsets[i+1]]`.
- `go_ids`: GO token corresponding to each row of `term_embeddings`.

For example, `term_embeddings[offsets[i]:offsets[i+1]]` recovers gene `i`'s (N_i \times 1024) matrix. The exporter writes a JSON sidecar with the model, dimensions, and truncation summary. Use `--embedding-dtype float32` if full precision is required.

## Step 2: train and evaluate

For the Colab layout, mount Drive and run from the project directory. This assumes the project files are present at `/content/gene_seq_finetune` and the exported NPZ plus the two CSV files are in `MyDrive/OMICS/gene seq`:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/gene_seq_finetune
%pip install -r requirements.txt
!python scripts/train_eval.py
```

By default, results are written to `/content/gene_seq_finetune/results/breast_v1`. To preserve them in Drive, pass an output directory there:

```python
!python scripts/train_eval.py --output-dir "/content/drive/MyDrive/OMICS/gene seq fine-tune/results/breast_v1"
```

The script checks that the expression, response, and embedding files exist before loading them. You can supply alternative paths with `--expression`, `--responses`, and `--gene-embeddings`.

```powershell
python scripts\train_eval.py `
  --expression "H:\OMICS\datasets_v0\gene seq\gene_expression_breast.csv" `
  --responses "H:\OMICS\datasets_v0\gene seq\drug_response_breast_matched.csv" `
  --gene-embeddings "data\human_gene_gobert_embeddings.npz" `
  --output-dir "results\breast_v1" --device auto
```

Outputs include out-of-fold predictions, metrics, checkpoints, and preprocessing objects. The drug input is a learned `DRUG_ID` embedding, so it predicts for compounds represented during training, not genuinely unseen drugs. Precomputed GoBERT vectors remain frozen; the expression gate, projection, drug/dataset embeddings, and response heads are trained.
