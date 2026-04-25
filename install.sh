#!/usr/bin/env bash
# Tested on a pristine Ubuntu 26.04 VM.

set -euo pipefail

echo "1. Installing system package dependencies..."
sudo apt-get update
sudo apt-get install -y git cmake build-essential python3-venv python3-dev libopenblas-dev liblapack-dev


echo "2. Creating a virtual environment..."
python3 -m venv venv
./venv/bin/python3 -m pip install numpy matplotlib
venv_python="$(./venv/bin/python3 -c 'import sys; print(sys.executable)')"
venv_lib="$("$venv_python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/"


echo "3. fetching xfac with the grouped bit-ordering patch..."
git clone "https://github.com/elv3rs/xfac.git"
git -C ./xfac checkout "61a85334f9d3d2066d8877009f83b63e26e7262c"
git -C ./xfac submodule update --init --recursive --depth 1

echo "4. Configuring..."
cmake -S ./xfac -B ./xfac/build \
  -D CMAKE_BUILD_TYPE=Release \
  -D XFAC_BUILD_PYTHON=ON \
  -D CMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -D Python_EXECUTABLE="$venv_python" \
  -D Python3_EXECUTABLE="$venv_python"

echo "5. compiling..."
cmake --build ./xfac/build --target xfacpy -j "$(nproc)"

echo "6. Installing (copying .so into venv)..."
compiled_so="$(find ./xfac/build -name 'xfacpy*.so' -print -quit)" 
cp "$compiled_so" "$venv_lib" 
