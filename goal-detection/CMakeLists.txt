cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_FLAGS "-std=c++11 -fopenmp")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11)
project( goal_detect )

find_package( OpenCV REQUIRED )
find_package(ZED 0.8.2 REQUIRED)
find_package(CUDA 6.5 REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} ${ZED_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ../threading-test )

link_directories(${ZED_LIBRARY_DIR})
link_directories(${CUDA_LIBRARY_DIRS})

add_executable( goal_detect goal_detect.cpp ../threading-test/zedin.cpp )
target_link_libraries( goal_detect ${OpenCV_LIBS} ${ZED_LIBRARIES} ${CUDA_LIBRARIES} )
