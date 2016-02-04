cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_FLAGS "-std=c++11")
project( goal_detect )

find_package(OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} ../bindetection )


add_executable( C920Cap C920Cap.cpp ../bindetection/mediain.cpp )
target_link_libraries( C920Cap ${OpenCV_LIBS})
