cmake_minimum_required(VERSION 2.8)
set(CMAKE_BUILD_TYPE Release)
project( goal_detect )
set(CMAKE_CXX_FLAGS "-std=c++11 -DZED_SUPPORT -Wall")

find_package(OpenCV REQUIRED )
find_package(ZED REQUIRED)
find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost COMPONENTS filesystem system serialization iostreams REQUIRED)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../bindetection/cmake")
find_package(ZMQ REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} ${ZED_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${Boost_INCLUDE_DIR} ../bindetection . )

link_directories(${ZED_LIBRARY_DIR})
link_directories(${CUDA_LIBRARY_DIRS})

add_executable( goal_detect goal_detect.cpp GoalDetector.cpp ../bindetection/mediain.cpp ../bindetection/zedin.cpp ../bindetection/Utilities.cpp ../bindetection/track3d.cpp ../bindetection/kalman.cpp ../bindetection/hungarian.cpp portable_binary_iarchive.cpp portable_binary_oarchive.cpp)
target_link_libraries( goal_detect ${OpenCV_LIBS} ${ZED_LIBRARIES} ${CUDA_LIBRARIES} ${CUDA_nppi_LIBRARY} ${CUDA_npps_LIBRARY} ${Boost_LIBRARIES} ${ZMQ_LIBRARIES})
