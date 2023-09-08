find_package (OpenCV REQUIRED)
find_package (Eigen3 3.3.9 EXACT REQUIRED NO_MODULE)

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

target_include_directories(test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
																			 ${PROJECT_SOURCE_DIR}/../include
																			 ${EIGEN3_INCLUDE_DIR}
																			 ${OpenCV_INCLUDE_DIRS}
																			 ${PROJECT_SOURCE_DIR}/../thirdParty/DBow3/src
																			 ${PROJECT_SOURCE_DIR}/../thirdParty/g2o)

install(TARGETS test DESTINATION ${PROJECT_SOURCE_DIR}/exe)
