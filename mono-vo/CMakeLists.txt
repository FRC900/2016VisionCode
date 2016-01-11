cmake_minimum_required(VERSION 2.8)
project( vo )
set(CMAKE_BUILD_TYPE Release)
find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( vo visodo.cpp calib.cpp ptsetreg.cpp vo_features.h )
target_link_libraries( vo ${OpenCV_LIBS} )
