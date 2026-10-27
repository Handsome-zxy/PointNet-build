#!/usr/bin/env bash
# Source this file (or `.venv/bin/activate`) before running PointNet so that
# TensorFlow can find the CUDA/cuDNN libraries installed by pip.
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_nvidia_libs=$(find "$BASE_DIR"/.venv/lib/python*/site-packages/nvidia -maxdepth 2 -type d -name lib 2>/dev/null | paste -sd:)
if [ -n "$_nvidia_libs" ]; then
    export LD_LIBRARY_PATH="$_nvidia_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
unset _nvidia_libs
