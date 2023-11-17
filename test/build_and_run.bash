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
./test data/ORBvoc.txt data/TUM1.yaml data/rgbd_dataset/ data/fr1_xyz.txt
