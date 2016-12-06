cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_FLAGS "-std=c++0x -fopenmp")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11)
project( kalman_testing )

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -flto")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Ofast -flto")
if (${CMAKE_LIBRARY_ARCHITECTURE} STREQUAL "arm-linux-gnueabihf")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mcpu=cortex-a15 -mfpu=neon-vfpv4 -fvect-cost-model")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -mcpu=cortex-a15 -mfpu=neon-vfpv4 -fvect-cost-model")
else()
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -march=native")
endif()


find_package(OpenCV REQUIRED )
find_package(ZED REQUIRED)
find_package(CUDA REQUIRED)
find_package(Boost COMPONENTS system thread REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} 
	${EIGEN3_INCLUDE_DIR}
	${ZED_INCLUDE_DIRS} 
	${CUDA_INCLUDE_DIRS} 
	/home/ubuntu/2016VisionCode/libfovis/build/include/fovis 
	../fovis 
	../zebravision)
link_directories(${ZED_LIBRARY_DIR})
link_directories(${CUDA_LIBRARY_DIRS})
link_directories(/home/ubuntu/2016VisionCode/libfovis/build/lib/)

add_executable( kalman_testing 
	kalman_testing.cpp 
	../fovis/FovisLocalizer.cpp 
	../zebravision/asyncin.cpp 
	../zebravision/syncin.cpp 
	../zebravision/mediain.cpp 
	../zebravision/zedsvoin.cpp 
	../zebravision/zedcamerain.cpp 
	PositionKF.cpp)
target_link_libraries( kalman_testing ${OpenCV_LIBS} fovis ${ZED_LIBRARIES} ${CUDA_LIBRARIES} ${CUDA_nppi_LIBRARY} ${CUDA_npps_LIBRARY})
