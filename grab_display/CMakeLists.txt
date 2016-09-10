cmake_minimum_required(VERSION 2.8)
set(CMAKE_LEGACY_CYGWIN_WIN32 0) # Remove when CMake >= 2.8.4 is required
project( grab_chroma )
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -std=c++11 -Wall -Wextra")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -flto")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -Ofast -flto")

find_package( CUDA REQUIRED )
find_package( OpenCV REQUIRED )
find_package( Boost COMPONENTS filesystem system thread program_options REQUIRED )
INCLUDE( FindCUDA )

include_directories(../framegrabber)
include_directories(../zebravision)
include_directories( ${Boost_INCLUDE_DIR} )
include_directories( /usr/local/cuda/include )

set (CUDA_PROPAGATE_HOST_FLAGS off)
LIST(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_52,code=sm_52)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_53,code=sm_53)
LIST(APPEND CUDA_NVCC_FLAGS -Xcompiler ,\"-O3\",\"-DNDEBUG\","-ffast-math")

add_executable( ColorSharpen ColorSharpen.cpp)
target_link_libraries( ColorSharpen ${OpenCV_LIBS} )
add_executable( grab_chroma grab_chroma.cpp chroma_key.cpp image_warp.cpp imageShift.cpp random_subimage.cpp)
target_link_libraries( grab_chroma ${OpenCV_LIBS} )
add_executable( rotate_from_imageclipper rotate_from_imageclipper.cpp chroma_key.cpp image_warp.cpp imageclipper_read.cpp)
target_link_libraries( rotate_from_imageclipper ${OpenCV_LIBS} ${Boost_LIBRARIES} )
add_executable( shift_from_imageclipper shift_from_imageclipper.cpp chroma_key.cpp image_warp.cpp imageShift.cpp imageclipper_read.cpp random_subimage.cpp)
target_link_libraries( shift_from_imageclipper ${OpenCV_LIBS} ${Boost_LIBRARIES} )

CUDA_ADD_EXECUTABLE ( zcacalc zcacalc.cpp ../zebravision/zca.cpp ../zebravision/zca.cu ../framegrabber/utilities_common.cpp random_subimage.cpp )
target_link_libraries( zcacalc ${OpenCV_LIBS} ${Boost_LIBRARIES} )
CUDA_ADD_EXECUTABLE ( zcarun zcarun.cpp ../zebravision/zca.cpp ../zebravision/zca.cu ../framegrabber/utilities_common.cpp )
target_link_libraries( zcarun ${OpenCV_LIBS} ${Boost_LIBRARIES} )
