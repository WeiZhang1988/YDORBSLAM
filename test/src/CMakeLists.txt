set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH ${PROJECT_SOURCE_DIR}/../lib)

find_package (OpenCV REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}
										${PROJECT_SOURCE_DIR}/..
										${PROJECT_SOURCE_DIR}/../include
										${EIGEN3_INCLUDE_DIR}
										${OpenCV_INCLUDE_DIRS})

link_directories(${PROJECT_SOURCE_DIR}/../lib
								 ${PROJECT_SOURCE_DIR}/../thirdParty/g2o/lib
								 ${PROJECT_SOURCE_DIR}/../thirdParty/DBow3/build/src)

add_executable(test test.cpp)

target_link_libraries(test PUBLIC slam
																	pthread
																	${OpenCV_LIBRARIES}
																	DBoW3
																	g2o_core
																	g2o_solver_dense
																	g2o_solver_eigen
																	g2o_types_sba
																	g2o_types_sim3)

target_include_directories(test PUBLIC ${PROJECT_SOURCE_DIR}/../include
																			 ${OpenCV_INCLUDE_DIRS}
																			 ${PROJECT_SOURCE_DIR}/../thirdParty/DBow3/build
																			 ${PROJECT_SOURCE_DIR}/../thirdParty/g2o
																			 ${PROJECT_SOURCE_DIR}/../thirdParty/g2o/build)

install(TARGETS test DESTINATION ${PROJECT_SOURCE_DIR}/exe)
