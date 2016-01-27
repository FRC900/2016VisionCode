cmake_minimum_required(VERSION 2.8)
set( CMAKE_LEGACY_CYGWIN_WIN32 0) # Remove when CMake >= 2.8.4 is required
project( zv )
set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/" )
set( CMAKE_BUILD_TYPE Release )
set( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -flto" )
set( CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -Ofast -flto" )
find_package( OpenCV REQUIRED )
find_package( Threads )
find_package( ZMQ )
include_directories( ${OpenCV_INCLUDE_DIRS} ${ZMQ_INCLUDE_DIRS} ../C920VideoCap )
add_executable( zv test.cpp imagedetect.cpp videoin_c920.cpp ../C920VideoCap/C920Camera.cpp track.cpp classifierio.cpp Args.cpp WriteOnFrame.cpp )
target_link_libraries( zv ${OpenCV_LIBS} ${LibNT} ${ZMQ_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} )

