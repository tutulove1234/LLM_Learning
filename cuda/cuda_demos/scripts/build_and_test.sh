#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${repo_dir}/build"
architecture="${CUDA_ARCHITECTURES:-native}"

cmake -S "${repo_dir}" -B "${build_dir}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="${architecture}"
cmake --build "${build_dir}" -j
ctest --test-dir "${build_dir}" --output-on-failure
