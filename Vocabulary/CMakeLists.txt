cmake_minimum_required( VERSION 2.8 )
project( vocab_traning )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

# opencv 
find_package( OpenCV 4.2.0 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# dbow3 
# dbow3 is a simple lib so I assume you installed it in default directory 
set( DBoW3_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/../thirdParty/DBow3/install/include" )
set( DBoW3_LIBS "${PROJECT_SOURCE_DIR}/../thirdParty/DBow3/install/lib/libDBoW3.so" )

add_executable( feature_training feature_training.cpp )
target_link_libraries( feature_training ${OpenCV_LIBS} ${DBoW3_LIBS} )

add_executable( rgbd_vocab rgbd_vocab.cpp )
target_link_libraries( rgbd_vocab ${OpenCV_LIBS} ${DBoW3_LIBS} )
