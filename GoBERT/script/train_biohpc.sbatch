#!/bin/bash -l
#SBATCH --job-name=gobert
#SBATCH --partition=GPUA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=../logs/gobert-%j.out
#SBATCH --error=../logs/gobert-%j.err
set -euo pipefail
[[ -n ${SLURM_JOB_ID:-} ]] || { echo 'Submit this script with sbatch.' >&2; exit 1; }
repo=${GOBERT_REPO:-${SLURM_SUBMIT_DIR:?}}
project_root=$(dirname -- "$repo")
source "$project_root/biohpc.env"
export PYTHONNOUSERSITE=1
python="$GOBERT_ENV_PREFIX/bin/python"
[[ -x $python ]] || { echo 'Run setup_biohpc.sbatch first and inspect its log.' >&2; exit 1; }
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
cd "$GOBERT_REPO/GoBERT"
nvidia-smi
"$python" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable in this job"; print(torch.cuda.get_device_name(0))'

output=${GOBERT_OUTPUT_DIR:-$GOBERT_PROJECT_ROOT/runs/e5_mistral}
for arg in "$@"; do
    if [[ $arg == --smoke_test ]]; then output="$GOBERT_PROJECT_ROOT/runs/smoke-$SLURM_JOB_ID"; fi
done
# Slurm sets CUDA_VISIBLE_DEVICES. The trainer uses one GPU; do not override it
# or request multiple GPUs for this entry point.
exec srun "$python" -u train_colab.py --output_dir "$output" "$@"
