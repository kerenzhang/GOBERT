# GoBERT on UT Southwestern BioHPC

These instructions are for UTSW's **Nucleus** cluster, not Cornell BioHPC.
The scripts transfer files and define Slurm jobs; they do not submit jobs until
you run the `sbatch` commands below.

## Storage and scheduler references

- [Storage cheat sheet](https://portal.biohpc.utsw.edu/content/guides/storage-cheat-sheet/):
  use `/work` for active data analysis; `/home2` is for configuration and small
  code, while `/project` or `/archive` is appropriate for retaining inactive data.
- [Introduction](https://portal.biohpc.utsw.edu/content/guides/introduction-to-biohpc/):
  personal work storage is `/work/<department>/<username>`. Login to
  `nucleus.biohpc.swmed.edu` and submit analysis jobs rather than training on the
  shared login node.
- [Slurm guide](https://portal.biohpc.utsw.edu/content/guides/slurm/): submit with
  `sbatch`, monitor with `squeue`, request a GPU using `--gres=gpu:1`.
- [Conda guide](https://portal.biohpc.utsw.edu/content/guides/conda-biohpc/):
  Conda is available on compute nodes, not necessarily login nodes; use an
  explicit environment prefix to keep packages out of the home quota.

Reviewed November 17, 2026. Your department path, partition access, and available
module versions must be checked on your account. The default training partition
is `GPUA100`; use `sinfo` to check availability. Resource requests and the 24-hour
training wall time are starting settings, not a runtime estimate.

## 1. Pull the code and embedding into work storage

From your own terminal:

```bash
ssh YOUR_BIOHPC_USERNAME@nucleus.biohpc.swmed.edu
```

Then on Nucleus, replace `YOUR_DEPARTMENT` with your actual directory name:

```bash
export BIOHPC_WORK="/work/YOUR_DEPARTMENT/$USER"
curl --fail --location --retry 3 \
  https://raw.githubusercontent.com/kerenzhang/GOBERT/main/GoBERT/script/pull_biohpc.sh \
  --output "$BIOHPC_WORK/pull_gobert.sh"
bash "$BIOHPC_WORK/pull_gobert.sh" "$BIOHPC_WORK"
source "$BIOHPC_WORK/GO-BERT/biohpc.env"
cd "$GOBERT_REPO"
```

The pull script clones `main`, retrieves the 746 MiB Git LFS embedding, and
verifies its SHA-256 against the repository pointer. If Git LFS is unavailable,
it installs a checksum-verified, portable Linux x86_64 binary under work storage
from the [official v3.8.1 release](https://github.com/git-lfs/git-lfs/releases/tag/v3.8.1).
No sudo or Conda is needed for the pull. Rerunning the script updates a clean
checkout by fast-forward only and leaves local edits intact.

The storage layout is:

```text
/work/DEPARTMENT/USERNAME/GO-BERT/
  GOBERT/         repository and training inputs
  envs/gobert/    Python environment
  cache/          Hugging Face, Conda, pip, and tool downloads
  logs/           Slurm stdout/stderr
  runs/           checkpoints and metrics
  tools/          portable Git LFS if needed
  biohpc.env      storage paths for sourcing in subsequent sessions
```

## 2. Create the environment through Slurm

From `$GOBERT_REPO` on the login node:

```bash
sbatch GoBERT/script/setup_biohpc.sbatch
squeue -u "$USER"
```

Wait for successful completion and inspect `../logs/gobert-setup-JOBID.out` and
`.err` before training. This CPU job creates Python 3.11 under work storage,
installs the locally tested PyTorch 2.6 CUDA 11.8 build and pinned GoBERT
dependencies, and runs small regression tests. Installation requires outbound
package-download access. The wheel source follows
[PyTorch's version-specific installation instructions](https://pytorch.org/get-started/previous-versions/).

If Conda needs a module in your session, find the actual module name on a compute
node (`module avail python`), then export `GOBERT_CONDA_MODULE` to that name
before submitting setup. The public guide's `python/3.12.x-anaconda-mamba` is
an example, not a verified exact module name for your account.

## 3. Submit smoke and full training jobs

```bash
# First verify the complete pipeline on one allocated GPU.
sbatch --time=00:15:00 GoBERT/script/train_biohpc.sbatch --smoke_test
squeue -u "$USER"

# After the smoke job succeeds, start the full 10-layer model.
sbatch GoBERT/script/train_biohpc.sbatch
```

The smoke job writes a separate `runs/smoke-JOBID/` directory. Full training writes
to `runs/e5_mistral/`, uses the existing defaults (18 epochs, microbatch 2,
accumulation 32, fp16), and saves every 500 optimizer steps and after validation.
The trainer retains its historical filename `train_colab.py` and also runs on
BioHPC without any Google Drive or notebook dependency.

To change an allowed partition, pass, for example,
`sbatch --partition=GPUv100s ...`. The setup is intended for A100/V100/H100-class
GPUs; it does not use the old K20/K40 cards in the generic `GPU` partition.
There is no need to request multiple GPUs for this single-GPU trainer.

To continue after a time limit:

```bash
sbatch GoBERT/script/train_biohpc.sbatch \
  --resume_from_checkpoint "$GOBERT_PROJECT_ROOT/runs/e5_mistral/checkpoint_last"
```

Keep the original optimizer, epoch count, seed, and batch settings when resuming.
The job does not automatically requeue. Move completed runs to your lab's agreed
long-term storage when finished; `/work` is an active-work location.
