#!/usr/bin/env bash
# Build the pinned OPTIC llama.cpp runtime and run the downloadable CPU GGUF.
set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
    echo "Usage: bash run_gguf_cpu.sh [--threads N] [llama cli options]"
    echo "Example: bash run_gguf_cpu.sh --threads 8 -p 'Explain why the sky is blue.'"
    echo "Requires Linux, Git, CMake, a C++17 compiler, curl and sha256sum."
    echo "Optional: OPTIC_GGUF_CACHE, OPTIC_GGUF_MODEL, OPTIC_BUILD_JOBS."
    exit 0
fi

threads=8
cli_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --threads|-t)
            threads="${2:?--threads requires a value}"
            shift 2 ;;
        *) cli_args+=("$1"); shift ;;
    esac
done
[[ "$threads" =~ ^[1-9][0-9]*$ ]] || { echo 'Threads must be positive.' >&2; exit 2; }

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
patch_file="$script_dir/llama.cpp-optic.patch"
if [[ ! -f "$patch_file" ]]; then
    patch_file="$script_dir/../patches/llama.cpp-optic.patch"
fi
patch_sha=cb0f60fc083697de8f6f5ca655febdf1c13a472442adce603aad0e69e4cb71d1
printf '%s  %s\n' "$patch_sha" "$patch_file" | sha256sum --check --status

runtime_commit=f7c01c5226cac1de1052c17920ff38cdefea0ae6
model_sha=e68651a77b06155670690c69d0eedfb1566ee0af359ec03ee96fad2a33dd2014
model_name=gemma-4-E4B-QAT-Q4_0-OPTIC-100x.gguf
cache_dir="${OPTIC_GGUF_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/optic-gguf}"
mkdir -p "$cache_dir"
runtime_dir="$cache_dir/llama.cpp-${runtime_commit:0:12}-${patch_sha:0:12}"
binary="$runtime_dir/build-cpu/bin/llama"

if [[ ! -x "$binary" ]]; then
    if [[ ! -d "$runtime_dir/.git" ]]; then
        git init "$runtime_dir"
        git -C "$runtime_dir" remote add origin https://github.com/ggml-org/llama.cpp.git
    fi
    if [[ ! -f "$runtime_dir/.optic-patch-applied" ]]; then
        git -C "$runtime_dir" fetch --depth 1 origin "$runtime_commit"
        git -C "$runtime_dir" checkout --detach "$runtime_commit"
        git -C "$runtime_dir" apply "$patch_file"
        touch "$runtime_dir/.optic-patch-applied"
    fi
    cmake -S "$runtime_dir" -B "$runtime_dir/build-cpu" \
        -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF \
        -DGGML_NATIVE=ON -DGGML_OPENMP=ON -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_UI=OFF
    cmake --build "$runtime_dir/build-cpu" --target llama-app -j "${OPTIC_BUILD_JOBS:-8}"
fi

default_model="$cache_dir/$model_name"
if [[ -f "$script_dir/$model_name" ]]; then
    default_model="$script_dir/$model_name"
fi
model="${OPTIC_GGUF_MODEL:-$default_model}"
if [[ ! -f "$model" ]]; then
    curl --fail --location --retry 3 --continue-at - \
        "https://huggingface.co/zcahjl3/gemma-4-E4B-it-OPTIC-100x/resolve/main/gguf/$model_name" \
        --output "$model.part"
    printf '%s  %s\n' "$model_sha" "$model.part" | sha256sum --check --status
    mv "$model.part" "$model"
fi
printf '%s  %s\n' "$model_sha" "$model" | sha256sum --check --status

if [[ ${#cli_args[@]} -eq 0 ]]; then
    cli_args=(-p 'Explain why the sky is blue in two sentences.')
fi
export CUDA_VISIBLE_DEVICES="" OMP_DYNAMIC=FALSE
exec "$binary" cli -m "$model" -ngl 0 -t "$threads" -tb "$threads" \
    --load-mode auto --lazy-mode on -c 512 -b 512 -ub 512 -n 64 \
    --cache-type-k f16 --cache-type-v f16 --flash-attn on \
    --single-turn --simple-io --reasoning off --temp 0 --seed 0 \
    "${cli_args[@]}"
