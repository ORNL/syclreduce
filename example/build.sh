#!/bin/bash
set -e

[ -d build ] && rm -fr build
mkdir -p build && cd build

SYCL=/usr/local/spack/opt/spack/darwin-bigsur-skylake/apple-clang-13.0.0/hipsycl-hotfix-cdhe4yqffhy522ltgqqvauhjvrrkprwg
cmake -DCMAKE_INSTALL_PREFIX=$HOME/local \
	  -DCMAKE_PREFIX_PATH=$HOME/local';'$SYCL \
	  -DCMAKE_CXX_COMPILER=$SYCL/bin/syclcc \
	  ..
