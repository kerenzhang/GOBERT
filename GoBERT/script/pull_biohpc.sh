#!/usr/bin/env bash
# Run on the Nucleus login node. Only transfers files; does not start training.
set -euo pipefail

usage() {
    printf 'Usage: bash pull_biohpc.sh /work/DEPARTMENT/USERNAME\n'
    printf 'Creates GO-BERT/{GOBERT,envs,cache,logs,runs,tools} under that directory.\n'
}
fail() { printf 'Error: %s\n' "$*" >&2; exit 1; }
if [[ ${1:-} == --help || ${1:-} == -h ]]; then usage; exit 0; fi
[[ $# == 1 ]] || { usage >&2; exit 2; }
[[ $1 == /* && -d $1 && -w $1 ]] || fail 'Supply your existing, writable absolute BioHPC storage directory.'
for tool in git curl tar sha256sum; do
    command -v "$tool" >/dev/null || fail "Required command is missing: $tool"
done

work_base=$(cd -- "$1" && pwd -P)
project_root="$work_base/GO-BERT"
repo="$project_root/GOBERT"
repo_url=${GOBERT_REPO_URL:-https://github.com/kerenzhang/GOBERT.git}
mkdir -p "$project_root"/{envs,cache,logs,runs,tools}

# Conda is not guaranteed on login nodes. Bootstrap a verified portable Git LFS
# only if a working installation is absent. No sudo or home-directory install.
lfs_dir="$project_root/tools/git-lfs-3.8.1"
if ! git lfs version >/dev/null 2>&1; then
    if [[ ! -x $lfs_dir/git-lfs ]]; then
        [[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || fail 'Portable Git LFS requires Linux x86_64; load Git LFS for this platform first.'
        archive="$project_root/cache/git-lfs-linux-amd64-v3.8.1.tar.gz"
        curl --fail --location --retry 3 \
            https://github.com/git-lfs/git-lfs/releases/download/v3.8.1/git-lfs-linux-amd64-v3.8.1.tar.gz \
            --output "$archive"
        printf '%s  %s\n' e72cf723a4a660438862282e8aecf3885fc8123e88bfcad534cf6b941a354b75 "$archive" | sha256sum --check --status
        mkdir -p "$lfs_dir"
        tar --extract --gzip --file "$archive" --directory "$lfs_dir" --strip-components=1
    fi
    export PATH="$lfs_dir:$PATH"
    git lfs version
fi

if [[ -e $repo ]]; then
    [[ -d $repo/.git ]] || fail "Destination exists but is not a repository: $repo"
    [[ $(git -C "$repo" remote get-url origin) == "$repo_url" ]] || fail 'Existing origin differs from the requested repository.'
    [[ $(git -C "$repo" symbolic-ref --short HEAD) == main ]] || fail 'Existing checkout must be on main.'
    git -C "$repo" diff --quiet || fail 'Commit or stash local tracked-file edits before updating.'
    git -C "$repo" diff --cached --quiet || fail 'Commit or stash staged changes before updating.'
    GIT_LFS_SKIP_SMUDGE=1 git -C "$repo" pull --ff-only origin main
else
    # Clone pointers without starting LFS filters; install hooks explicitly below.
    # This also avoids hook-installation errors in older Git clone implementations.
    git -c filter.lfs.process= -c filter.lfs.smudge= -c filter.lfs.required=false \
        clone --branch main --single-branch --depth 1 "$repo_url" "$repo"
fi

git -C "$repo" lfs install --local
embedding=GoBERT/data/e5-mistral-7b-instruct.npy
git -C "$repo" lfs pull --include="$embedding" --exclude=''
expected=$(git -C "$repo" show "HEAD:$embedding" | sed -n 's/^oid sha256://p')
[[ $expected =~ ^[0-9a-f]{64}$ ]] || fail 'Embedding is not tracked by a valid Git LFS pointer.'
printf '%s  %s\n' "$expected" "$repo/$embedding" | sha256sum --check --status

# All paths are quoted for safe sourcing, even when storage names contain spaces.
{
    printf 'export GOBERT_PROJECT_ROOT=%q\n' "$project_root"
    printf 'export GOBERT_REPO=%q\n' "$repo"
    printf 'export GOBERT_ENV_PREFIX=%q\n' "$project_root/envs/gobert"
    printf 'export HF_HOME=%q\n' "$project_root/cache/huggingface"
    printf 'export XDG_CACHE_HOME=%q\n' "$project_root/cache"
    printf 'export PIP_CACHE_DIR=%q\n' "$project_root/cache/pip"
    printf 'export CONDA_PKGS_DIRS=%q\n' "$project_root/cache/conda-pkgs"
    if [[ -x $lfs_dir/git-lfs ]]; then printf 'export PATH=%q:"$PATH"\n' "$lfs_dir"; fi
} > "$project_root/biohpc.env"

printf '\nRepository ready: %s\n' "$repo"
printf 'Commit: %s\n' "$(git -C "$repo" rev-parse --short HEAD)"
printf 'Embedding SHA-256 verified.\n\nNext commands:\n'
printf 'source %q\n' "$project_root/biohpc.env"
printf 'cd %q\n' "$repo"
printf 'sbatch GoBERT/script/setup_biohpc.sbatch\n'
printf 'squeue -u "$USER"\n'
