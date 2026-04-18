#!/usr/bin/env bash
#
# scripts/refresh_baselines.sh - Regenerate per-model baseline JSON files.
#
# Builds sam3_cli in Release mode and runs `bench all` against each
# available model variant under models/, writing the result JSON to
# benchmarks/baselines/<variant>.json. Missing models are skipped with
# a warning so partial regenerations are OK.
#
# Usage: scripts/refresh_baselines.sh [BUILD_DIR]
#   BUILD_DIR defaults to build-release.
#
# Copyright (c) 2026 Rifky Bujana Bisri
# SPDX-License-Identifier: MIT

set -uo pipefail

BUILD="${1:-${BUILD:-build-release}}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Configuring $BUILD (Release, SAM3_BENCH=ON)"
cmake -S . -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DSAM3_BENCH=ON

echo "==> Building sam3_cli"
cmake --build "$BUILD" --target sam3_cli -j

mkdir -p benchmarks/baselines

# Collect failures so one broken model does not abort the whole batch.
failed=()
for variant in efficient tinyvit hiera; do
    case "$variant" in
        efficient) model="models/efficient.sam3" ;;
        tinyvit)   model="models/tinyvit_l.sam3" ;;
        hiera)     model="models/sam3.sam3" ;;
    esac

    if [ ! -f "$model" ]; then
        echo "skip: $model not found"
        continue
    fi

    out="benchmarks/baselines/${variant}.json"
    echo "==> Running bench all for $variant -> $out"
    if ! "$BUILD/sam3_cli" bench all \
            --model "$model" --backend metal \
            --output "$out"; then
        echo "FAIL: $variant (continuing with next model)"
        failed+=("$variant")
    fi
done

if [ "${#failed[@]}" -gt 0 ]; then
    echo "==> Done with failures: ${failed[*]}"
    exit 1
fi

echo "==> Done"
