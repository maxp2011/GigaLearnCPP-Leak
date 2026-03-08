#!/bin/bash
# One-shot: setup meshes, build, run training on Linux (vast.ai)
set -e
cd "$(dirname "$0")"

echo "=== 1. Pull latest ==="
git pull

echo "=== 2. Setup collision meshes ==="
chmod +x setup_collision_meshes.sh
./setup_collision_meshes.sh

echo "=== 3. Build ==="
cd build && cmake --build . -j$(nproc)

echo "=== 4. Run training ==="
./GigaLearnBot
