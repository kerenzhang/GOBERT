# Retrain GoBERT on a Colab GPU

Select **Runtime > Change runtime type > GPU**. The entry point is
`train_colab.py` uses the reference GoBERT model and local training inputs.

## 1. Clone and install

Run these notebook cells in order:

```python
!GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/kerenzhang/GOBERT.git /content/GOBERT
%cd /content/GOBERT
!apt-get -qq update
!apt-get -qq install git-lfs
!git lfs install --local
!git lfs pull --include="GoBERT/data/e5-mistral-7b-instruct.npy"
%cd /content/GOBERT/GoBERT
%pip install -q -r requirements-colab.txt
!nvidia-smi
```

Git LFS fetches the 746 MiB embedding table separately from ordinary source
files. See [GitHub's Git LFS documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage).

The saved table supplies semantic initialization; the 7B text encoder is not
loaded. Its 47,729 rows must follow `hf_gobert/vocab.txt` exactly. The model adds
five special-token rows, projects the 4,096 features to 1,024 dimensions, and
blends semantic and random weights using the reference 0.9/0.1 ratio.
Row count and vocabulary coverage were checked locally; a bare `.npy` file
does not provide evidence of the GO-ID ordering used during its creation.

## 2. Verify the pipeline

```python
!python train_colab.py --smoke_test --output_dir /content/gobert-smoke
```

This uses a tiny model, the real semantic table, and small data subsets. It
checks forward/backward passes, validation, and checkpoint writing. It is not
a full-model training result.

## 3. Save full training to Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
!python train_colab.py \
    --output_dir /content/drive/MyDrive/GoBERT_runs/e5_mistral \
    --num_train_epochs 18 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --learning_rate 2.1e-4 \
    --weight_decay 0.01 \
    --max_seq_length 50 \
    --mixed_precision fp16 \
    --seed 0
```

Defaults already point to the local tokenizer, embedding table, training split,
and **validation** split. The test split is held aside. This command uses the
reference 10-layer model (hidden size 1,024) and an effective batch of 64.
Gradient checkpointing is enabled. If GPU memory is insufficient, use batch
sizes of 1 and accumulation of 64. A Colab session may end before all 18 epochs
finish; resumable checkpoints are written every 500 optimizer steps and after
each validation pass. Adjust this interval with `--checkpointing_steps`.

Outputs:

- `checkpoint_last/`: weights, tokenizer, optimizer, scheduler, scaler, batch
  position, and RNG states for continuing training.
- `checkpoint_best/`: best weights and tokenizer by validation masked-token
  accuracy. This is an export, not a resumable optimizer checkpoint.
- `metrics.jsonl` and `run_config.json`: validation history and settings.

To resume, rerun the same training command with:

```text
--resume_from_checkpoint /content/drive/MyDrive/GoBERT_runs/e5_mistral/checkpoint_last
```

Keep the original epoch count, batch size, accumulation, seed, model, and
optimizer settings when resuming. A partial run can be requested with
`--max_train_steps 10 --max_eval_batches 2`; remove these limits to continue.

## Scope of the Colab changes

The complete graph stays sparse, and only batch rows become dense. Semantic
initialization restores shared input/output embedding weights. Batches with no
masked tokens have zero MLM loss rather than NaN. The launcher handles one GPU
directly without NCCL or duplicate distributed wrappers and uses one consistent
sequence length for tokenization and collation.

The original neighborhood objective and masking probability logic are retained.
In particular, neighborhood labels still use the corrupted input IDs, and the
reference parent-masking exclusion remains imperfect. These are research-method
limitations to resolve before claiming a corrected paper reproduction.
