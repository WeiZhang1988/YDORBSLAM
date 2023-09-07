find_package (OpenCV REQUIRED)
find_package (Eigen3 3.3.9 EXACT REQUIRED NO_MODULE)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}
										${PROJECT_SOURCE_DIR}
										${PROJECT_SOURCE_DIR}/include
										${EIGEN3_INCLUDE_DIR}
										${OpenCV_INCLUDE_DIRS})

add_library(slam SHARED orbExtractor.cpp
												orbMatcher.cpp
												mapPoint.cpp
												frame.cpp
												keyFrame.cpp
												keyFrameDatabase.cpp
												map.cpp
												converter.cpp
												optimizer.cpp
												loopClosing.cpp)

target_link_libraries(slam PUBLIC pthread
																	${OpenCV_LIBRARIES}
																	${PROJECT_SOURCE_DIR}/thirdParty/DBow3/build/src/libDBoW3.so
																	${PROJECT_SOURCE_DIR}/thirdParty/g2o/lib/libg2o_core.so
																	${PROJECT_SOURCE_DIR}/thirdParty/g2o/lib/libg2o_solver_dense.so
																	${PROJECT_SOURCE_DIR}/thirdParty/g2o/lib/libg2o_solver_eigen.so
																	${PROJECT_SOURCE_DIR}/thirdParty/g2o/lib/libg2o_types_sba.so
																	${PROJECT_SOURCE_DIR}/thirdParty/g2o/lib/libg2o_types_sim3.so)

target_include_directories(slam PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
																			 ${OpenCV_INCLUDE_DIRS}
																			 ${PROJECT_SOURCE_DIR}/thirdParty/DBow3/
																			 ${PROJECT_SOURCE_DIR}/thirdParty/DBow3/build/
																			 ${PROJECT_SOURCE_DIR}/thirdParty/DBow3/build/src/
																			 ${PROJECT_SOURCE_DIR}/thirdParty/g2o/
																			 ${PROJECT_SOURCE_DIR}/thirdParty/g2o/build/)

install(FILES orbExtractor.hpp 
							orbMatcher.hpp
							mapPoint.hpp
							frame.hpp
							keyFrame.hpp
							keyFrameDatabase.hpp
							map.hpp
							converter.hpp
							optimizer.hpp
							loopClosing.hpp
							DESTINATION ${PROJECT_SOURCE_DIR}/include)
install(TARGETS slam LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/lib)
