#!/bin/bash
printf "\e[1;36m----- building and installing libslam -----\e[0m\n"
rm -r build include lib
mkdir build
cd build
cmake ..
cmake --build . --parallel 16
cmake --install .
cd ..

printf "\e[1;36m----- building and installing test -----\e[0m\n"
cd test
rm -r build exe
mkdir build
cd build
cmake ..
cmake --build . --parallel 16
cmake --install .

printf "\e[1;36m----- running test -----\e[0m\n"
cd ../exe
./test data/ORBvoc.txt data/TUM1.yaml data/rgbd_dataset/ data/fr1_xyz.txt
