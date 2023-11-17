set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH ${PROJECT_SOURCE_DIR}/lib 
												${PROJECT_SOURCE_DIR}/thirdParty/g2o/install/lib
								 				${PROJECT_SOURCE_DIR}/thirdParty/DBow3/install/lib
												${PROJECT_SOURCE_DIR}/thirdParty/Pangolin/install/lib)

find_package (OpenCV REQUIRED)
										
link_directories(${PROJECT_SOURCE_DIR}/thirdParty/g2o/install/lib
								 ${PROJECT_SOURCE_DIR}/thirdParty/DBow3/install/lib
								 ${PROJECT_SOURCE_DIR}/thirdParty/Pangolin/install/lib)

add_library(slam SHARED orbExtractor.cpp
												orbMatcher.cpp
												mapPoint.cpp
												frame.cpp
												keyFrame.cpp
												frameDrawer.cpp
												mapDrawer.cpp
												keyFrameDatabase.cpp
												map.cpp
												system.cpp
												viewer.cpp
												converter.cpp
												optimizer.cpp
												pnpSolver.cpp
												sim3Solver.cpp
												tracking.cpp
												localMapping.cpp
												loopClosing.cpp)

target_link_libraries(slam PUBLIC pthread
																	${OpenCV_LIBRARIES}
																	${PROJECT_SOURCE_DIR}/thirdParty/Pangolin/install/lib/libpangolin.so
																	${PROJECT_SOURCE_DIR}/thirdParty/DBow3/install/lib/libDBoW3.so
																	${PROJECT_SOURCE_DIR}/thirdParty/g2o/install/lib/libg2o_core.so
																	${PROJECT_SOURCE_DIR}/thirdParty/g2o/install/lib/libg2o_solver_dense.so
																	${PROJECT_SOURCE_DIR}/thirdParty/g2o/install/lib/libg2o_solver_eigen.so
																	${PROJECT_SOURCE_DIR}/thirdParty/g2o/install/lib/libg2o_types_sba.so
																	${PROJECT_SOURCE_DIR}/thirdParty/g2o/install/lib/libg2o_types_sim3.so
																	${PROJECT_SOURCE_DIR}/thirdParty/g2o/install/lib/libg2o_stuff.so)
																	
target_include_directories(slam BEFORE PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
																			 			  ${PROJECT_SOURCE_DIR}/include
																			 			  ${OpenCV_INCLUDE_DIRS}
																			 			  ${PROJECT_SOURCE_DIR}/thirdParty/Pangolin/install/include
																			 			  ${PROJECT_SOURCE_DIR}/thirdParty/eigen/install/include
																			 			  ${PROJECT_SOURCE_DIR}/thirdParty/DBow3/install/include
																			 			  ${PROJECT_SOURCE_DIR}/thirdParty/g2o/install/include)


install(FILES orbExtractor.hpp 
							orbMatcher.hpp
							mapPoint.hpp
							frame.hpp
							keyFrame.hpp
							frameDrawer.hpp
							mapDrawer.hpp
							keyFrameDatabase.hpp
							map.hpp
							system.hpp
							viewer.hpp
							converter.hpp
							optimizer.hpp
							pnpSolver.hpp
							sim3Solver.hpp
							tracking.hpp
							localMapping.hpp
							loopClosing.hpp
							enumclass.hpp
							stack_compatible_enable_shared_from_this.hpp
							
							DESTINATION ${PROJECT_SOURCE_DIR}/include)
install(TARGETS slam LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/lib)
