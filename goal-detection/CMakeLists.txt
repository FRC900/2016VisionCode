cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_FLAGS "-std=c++11 -fopenmp -DZED_SUPPORT")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11)
project( goal_detect )

find_package(OpenCV REQUIRED )
find_package(ZED REQUIRED)
find_package(CUDA REQUIRED)


include_directories( ${OpenCV_INCLUDE_DIRS} ${ZED_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ../threading-test )

link_directories(${ZED_LIBRARY_DIR})
link_directories(${CUDA_LIBRARY_DIRS})

add_executable( goal_detect goal_detect.cpp GoalDetector.cpp ../threading-test/zedin.cpp )
target_link_libraries( goal_detect ${OpenCV_LIBS} ${ZED_LIBRARIES} ${CUDA_LIBRARIES} )
