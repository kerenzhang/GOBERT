#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python train_colab.py "$@"
