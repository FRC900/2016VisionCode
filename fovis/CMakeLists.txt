cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_FLAGS "-std=c++0x -fopenmp")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11)
project( fovis_zed_test )

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
find_package(ZED 0.8.2 REQUIRED)
find_package(CUDA 6.5 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Boost REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} /usr/local/include /usr/local/include/eigen3 ${ZED_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
link_directories(${ZED_LIBRARY_DIR})
link_directories(${CUDA_LIBRARY_DIRS})

add_executable( fovis_zed_test fovis_testing.cpp zedin.cpp )
target_link_libraries( fovis_zed_test ${OpenCV_LIBS} /usr/local/lib/libfovis.so ${ZED_LIBRARIES} ${CUDA_LIBRARIES} ${CUDA_nppi_LIBRARY} ${CUDA_npps_LIBRARY})
