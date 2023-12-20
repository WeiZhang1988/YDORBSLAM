#!/bin/bash
printf "\e[1;36m----- building and installing DBoW3 -----\e[0m\n"
rm -r build install
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=../install ..
cmake --build . --parallel 16
cmake --install .
cd ..
