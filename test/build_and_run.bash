#!/bin/bash
printf "\e[1;36m----- building and installing test -----\e[0m\n"
rm -r build exe
mkdir build
cd build
cmake ..
cmake --build . --parallel 16
cmake --install .

printf "\e[1;36m----- running test -----\e[0m\n"
cd ../exe
./test
