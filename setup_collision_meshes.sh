#!/bin/bash
# Clone collision meshes from GitHub into project root (for Linux/VPS)
set -e
cd "$(dirname "$0")"
if [ -d "collision_meshes/soccar" ]; then
    echo "collision_meshes already present."
    exit 0
fi
echo "Cloning Collision_Meshes from GitHub..."
git clone --depth 1 https://github.com/maxp2011/Collision_Meshes.git _cm_tmp
mkdir -p collision_meshes
# Repo has soccar/ and hoops/ at root
[ -d "_cm_tmp/soccar" ] && mv _cm_tmp/soccar collision_meshes/
[ -d "_cm_tmp/hoops" ] && mv _cm_tmp/hoops collision_meshes/
rm -rf _cm_tmp
echo "Done. collision_meshes ready."
