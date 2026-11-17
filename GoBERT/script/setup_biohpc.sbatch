#!/bin/bash -l
#SBATCH --job-name=gobert-setup
#SBATCH --partition=128GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=../logs/gobert-setup-%j.out
#SBATCH --error=../logs/gobert-setup-%j.err
set -euo pipefail
[[ -n ${SLURM_JOB_ID:-} ]] || { echo 'Submit this script with sbatch.' >&2; exit 1; }
repo=${GOBERT_REPO:-${SLURM_SUBMIT_DIR:?}}
project_root=$(dirname -- "$repo")
source "$project_root/biohpc.env"
export PYTHONNOUSERSITE=1

# BioHPC documents Conda as available by default on compute nodes. An explicit
# module can be selected if your account needs it; do not guess a module version.
if [[ -n ${GOBERT_CONDA_MODULE:-} ]]; then module load "$GOBERT_CONDA_MODULE"; fi
command -v conda >/dev/null || { echo 'Conda is unavailable. Check module avail python on a compute node and set GOBERT_CONDA_MODULE.' >&2; exit 1; }
if [[ ! -f $GOBERT_ENV_PREFIX/conda-meta/history ]]; then
    conda create --yes --prefix "$GOBERT_ENV_PREFIX" --override-channels --channel conda-forge python=3.11 pip
fi
python="$GOBERT_ENV_PREFIX/bin/python"
"$python" -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
"$python" -m pip install --requirement "$GOBERT_REPO/GoBERT/requirements-colab.txt"
cd "$GOBERT_REPO/GoBERT"
"$python" -m unittest discover -s tests -v
"$python" -c 'import torch, transformers; print("torch:", torch.__version__, "transformers:", transformers.__version__)'
printf '\nEnvironment ready: %s\nNext submit a short GPU smoke job; see BIOHPC.md.\n' "$GOBERT_ENV_PREFIX"
