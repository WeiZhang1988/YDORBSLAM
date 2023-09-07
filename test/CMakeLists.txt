cmake_minimum_required(VERSION 3.16)
project(test VERSION 1.0.0)
set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_FLAGS "-std=c++11 ${SSE_FLAGS} -g -O3")
set(CMAKE_SKIP_BUILD_RPATH FALSE)  
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)  
set(CMAKE_INSTALL_RPATH ${PROJECT_SOURCE_DIR}/../lib
												${PROJECT_SOURCE_DIR}/../thirdParty/g2o/lib
												${PROJECT_SOURCE_DIR}/../thirdParty/DBow3/build/src)  
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE) 
add_subdirectory(src)
install(DIRECTORY data DESTINATION ${PROJECT_SOURCE_DIR}/exe)
