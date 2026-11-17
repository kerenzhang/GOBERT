# GoBERT training on Colab

Follow [COLAB.md](COLAB.md) for installation, a smoke test, full training, and
checkpoint resume commands.

For UT Southwestern Nucleus, see [BIOHPC.md](BIOHPC.md) for storage, a terminal
pull script, environment setup, and Slurm GPU submission.

This folder contains the training model and masking code adapted from
[MM-YY-WW/GoBERT](https://github.com/MM-YY-WW/GoBERT), upstream commit
`5fe2e56`, with a single-GPU Colab entry point and sparse graph storage.

The training inputs are included under `data/`, with the E5-Mistral semantic
embedding table stored in Git LFS. The local tokenizer is under `hf_gobert/`.
The ontology is the GO release dated 2024-01-17.

Run commands from this directory:

```bash
python train_colab.py --smoke_test
python train_colab.py --output_dir /path/to/persistent/GoBERT_runs/e5_mistral
```

The full model uses 10 layers and hidden size 1,024. Defaults select the original
training and validation splits, sequence length 50, fp16, and effective batch
size 64. See COLAB.md for the retained reference-method limitations.
