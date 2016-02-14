CMake_minimum_required(VERSION 2.8)
project(zv)
set(CMAKE_LEGACY_CYGWIN_WIN32 0) # Remove when CMake >= 2.8.4 is required
set(CMAKE_BUILD_TYPE Release)
add_definitions(-std=c++11 -Wall -Wextra)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -flto")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -Ofast -flto")
if (${CMAKE_LIBRARY_ARCHITECTURE} STREQUAL "arm-linux-gnueabihf")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mcpu=cortex-a15 -mfpu=neon-vfpv4")
  set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -mcpu=cortex-a15 -mfpu=neon-vfpv4")
else()
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
  set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -march=native")
endif()
find_package(ZED 0.9.2 QUIET)
if (${ZED_FOUND})
  add_definitions(-DZED_SUPPORT)
endif()
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
find_package(CUDA REQUIRED)
find_package(Boost COMPONENTS filesystem system REQUIRED)
find_package(Threads)
find_package(OpenCV REQUIRED)
find_package(ZMQ REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} ../caffe_classifier ../C920VideoCap ${ZED_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} /home/ubuntu/caffe/build/install/include /home/ubuntu/caffe/build/include /home/ubuntu-bvlc/caffe/include ${Boost_INCLUDE_DIR} ${ZMQ_INCLUDE_DIRS} ../goal-detection)
link_directories(${ZED_LIBRARY_DIR})
link_directories(${CUDA_LIBRARY_DIRS})

add_executable(zv test.cpp objdetect.cpp ../goal-detection/GoalDetector.cpp ../caffe_classifier/detect.cpp ../caffe_classifier/CaffeBatchPrediction.cpp ../caffe_classifier/fast_nms.cpp ../caffe_classifier/scalefactor.cpp detectstate.cpp mediain.cpp videoin.cpp camerain.cpp ../C920VideoCap/C920Camera.cpp c920camerain.cpp imagein.cpp track.cpp classifierio.cpp Args.cpp WriteOnFrame.cpp zedin.cpp groundtruth.cpp) 

find_library (LibCaffe caffe PATH  /home/ubuntu/caffe/build/lib)
find_library (LibGLOG  glog)
find_library (LibProtobuf protobuf)

target_link_libraries( zv ${OpenCV_LIBS} ${LibCaffe} ${CMAKE_THREAD_LIBS_INIT} ${ZED_LIBRARIES} ${CUDA_LIBRARIES} ${CUDA_nppi_LIBRARY} ${CUDA_npps_LIBRARY} ${Boost_LIBRARIES} ${LibGLOG} ${LibProtobuf} ${ZMQ_LIBRARIES})

# Code to test for bugs in GPU classifier code. Think this is fixed
# so for now don't build this code
#add_executable( gputest gputest.cpp objdetect.cpp classifierio.cpp )
#target_link_libraries( gputest ${OpenCV_LIBS} )

#add_executable(fpstest fpstest.cpp objdetect.cpp mediain.cpp videoin.cpp ../caffe_classifier/detect.cpp ../caffe_classifier/CaffeBatchPrediction.cpp ../caffe_classifier/fast_nms.cpp ../caffe_classifier/scalefactor.cpp)
#target_link_libraries( fpstest ${OpenCV_LIBS} )
