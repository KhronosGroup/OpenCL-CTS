#!/bin/sh

mkdir -p build_lnx
cd build_lnx
cmake -g "Unix Makefiles" ../
make --jobs 8
