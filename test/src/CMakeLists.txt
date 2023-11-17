set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH ${PROJECT_SOURCE_DIR}/../lib 
												${PROJECT_SOURCE_DIR}/../thirdParty/g2o/install/lib
								 				${PROJECT_SOURCE_DIR}/../thirdParty/DBow3/install/lib
												${PROJECT_SOURCE_DIR}/../thirdParty/Pangolin/install/lib)

find_package (OpenCV REQUIRED)

link_directories(${PROJECT_SOURCE_DIR}/../lib
								 ${PROJECT_SOURCE_DIR}/../thirdParty/g2o/install/lib
								 ${PROJECT_SOURCE_DIR}/../thirdParty/DBow3/install/lib
								 ${PROJECT_SOURCE_DIR}/../thirdParty/Pangolin/install/lib)

add_executable(test test.cpp)

target_link_libraries(test PUBLIC slam
																	pthread
																	${OpenCV_LIBRARIES}
																	GL
																	GLEW
																	${PROJECT_SOURCE_DIR}/../thirdParty/Pangolin/install/lib/libpangolin.so
																	${PROJECT_SOURCE_DIR}/../thirdParty/DBow3/install/lib/libDBoW3.so
																	${PROJECT_SOURCE_DIR}/../thirdParty/g2o/install/lib/libg2o_core.so
																	${PROJECT_SOURCE_DIR}/../thirdParty/g2o/install/lib/libg2o_solver_dense.so
																	${PROJECT_SOURCE_DIR}/../thirdParty/g2o/install/lib/libg2o_solver_eigen.so
																	${PROJECT_SOURCE_DIR}/../thirdParty/g2o/install/lib/libg2o_types_sba.so
																	${PROJECT_SOURCE_DIR}/../thirdParty/g2o/install/lib/libg2o_types_sim3.so
																	${PROJECT_SOURCE_DIR}/../thirdParty/g2o/install/lib/libg2o_stuff.so)

target_include_directories(test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
																			 ${PROJECT_SOURCE_DIR}/../include
																			 ${OpenCV_INCLUDE_DIRS}
																			 ${PROJECT_SOURCE_DIR}/../thirdParty/eigen/install/include
																			 ${PROJECT_SOURCE_DIR}/../thirdParty/Pangolin/install/include
																			 ${PROJECT_SOURCE_DIR}/../thirdParty/DBow3/install/include
																			 ${PROJECT_SOURCE_DIR}/../thirdParty/g2o/install/include
																			 )

install(TARGETS test DESTINATION ${PROJECT_SOURCE_DIR}/exe)
