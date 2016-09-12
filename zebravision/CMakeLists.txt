CMake_minimum_required(VERSION 2.8)
project(zv)
set(CMAKE_LEGACY_CYGWIN_WIN32 0) # Remove when CMake >= 2.8.4 is required
set(CMAKE_BUILD_TYPE Release)
add_definitions(-std=c++11 -Wall -Wextra)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -flto")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -Ofast -flto")
set(CMAKE_MODULE_PATH "/usr/share/cmake-2.8/Modules/")
if (${CMAKE_LIBRARY_ARCHITECTURE} STREQUAL "arm-linux-gnueabihf")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mcpu=cortex-a15 -mfpu=neon-vfpv4")
  set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -mcpu=cortex-a15 -mfpu=neon-vfpv4")
elseif (${CMAKE_LIBRARY_ARCHITECTURE} STREQUAL "aarch64-linux-gnu")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mcpu=cortex-a57")
  set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -mcpu=cortex-a57")
  #link_directories(/home/ubuntu/GIE/lib)
  #find_library (LibNVCaffeParser nvcaffe_parser /home/ubuntu/GIE/lib)
  #find_library (LibNVInfer nvinfer /home/ubuntu/GIE/lib)
  #add_definitions(-DUSE_GIE=1)
else()
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
  set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -march=native")
endif()
find_package(ZED)
if (${ZED_FOUND})
  add_definitions(-DZED_SUPPORT)
endif()
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
find_package(CUDA REQUIRED)
find_package(Boost COMPONENTS filesystem system serialization iostreams thread REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(ZMQ REQUIRED)
find_package(MKL)
if (${MKL_FOUND})
	add_definitions(-DUSE_MKL=1)
endif()

INCLUDE( FindCUDA )

set (CAFFE_BASE_DIR "/home/ubuntu/caffe")

# Caffe headers change the size of various
# objects based on these defines. Best to match
# how caffe itself was compiled
#add_definitions(-DCPU_ONLY=1)
add_definitions(-DUSE_CUDNN=1)

include_directories(
	/home/ubuntu/opencv-2.4.13/build/include
	#${OpenCV_INCLUDE_DIRS}
	${ZED_INCLUDE_DIRS}
	${CUDA_INCLUDE_DIRS}
	${CAFFE_BASE_DIR}/build/install/include
	${CAFFE_BASE_DIR}/build/include
	${CAFFE_BASE_DIR}/include
	${Boost_INCLUDE_DIR}
	${ZMQ_INCLUDE_DIRS}
   	${EIGEN3_INCLUDE_DIR}
	${MKL_INCLUDE_DIR}
	.
	../fovis
	/home/ubuntu/2016VisionCode/libfovis/build/include/fovis
	../../GIE/include)

link_directorieS(/home/ubuntu/GIE/lib)
link_directories(/home/ubuntu/opencv-2.4.13/build/lib)
link_directories(/home/kjaget/opencv-2.4.13/build/lib)
link_directories(${ZED_LIBRARY_DIR})
link_directories(${CUDA_LIBRARY_DIRS})
link_directories(/home/ubuntu/2016VisionCode/libfovis/build/lib/)

# Without this, nvcc grabs --std=c++11 which looks like
# it breaks builds for some reason
set (CUDA_PROPAGATE_HOST_FLAGS off)
LIST(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_52,code=sm_52)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_53,code=sm_53)
LIST(APPEND CUDA_NVCC_FLAGS -Xcompiler ,\"-O3\",\"-DNDEBUG\","-ffast-math")

CUDA_ADD_EXECUTABLE(zv
	objdetect.cpp
	GoalDetector.cpp
	detect.cpp
	Classifier.cpp
	CaffeClassifier.cpp
	GIEClassifier.cpp
	fast_nms.cpp
	scalefactor.cpp
	detectstate.cpp
	mediain.cpp
	videoin.cpp
	camerain.cpp
	C920Camera.cpp
	c920camerain.cpp
	imagein.cpp
	mediaout.cpp
	aviout.cpp
	pngout.cpp
	zmsout.cpp
	portable_binary_iarchive.cpp
	portable_binary_oarchive.cpp
	classifierio.cpp
	Args.cpp
	WriteOnFrame.cpp
	zedin.cpp
	groundtruth.cpp
	Utilities.cpp
	track3d.cpp
	#../fovis/FovisLocalizer.cpp
	FlowLocalizer.cpp
	kalman.cpp
	hungarian.cpp
	zca.cpp
	zca.cu
	ZvSettings.cpp
	zv.cpp)

find_library (LibCaffe caffe PATH  ${CAFFE_BASE_DIR}/build/lib)
find_library (LibGLOG  glog)
find_library (LibProtobuf protobuf /home/ubuntu/GIE/lib)
find_library (LibWconv wconv /home/ubuntu/GIE/lib)
find_library (LibTinyXML2 tinyxml2)
find_library (LibEFence efence)

target_link_libraries( zv
	${LibProtobuf}
	${OpenCV_LIBS}
	${LibCaffe}
	#fovis
	${CMAKE_THREAD_LIBS_INIT}
	${ZED_LIBRARIES}
	${CUDA_LIBRARIES}
	${CUDA_nppi_LIBRARY}
	${CUDA_npps_LIBRARY}
	${Boost_LIBRARIES}
	${LibGLOG}
	${ZMQ_LIBRARIES}
	${LibNVCaffeParser}
	${LibNVInfer}
	${LibTinyXML2}
	${MKL_LIBRARIES}
	#${LibEFence} 
	)

add_executable(convertzms convertzms.cpp mediain.cpp zedin.cpp mediaout.cpp zmsout.cpp portable_binary_oarchive.cpp portable_binary_iarchive.cpp ZvSettings.cpp)
target_link_libraries( convertzms ${Boost_LIBRARIES} ${OpenCV_LIBS} ${ZED_LIBRARIES} ${LibTinyXML2})
add_executable(mergezms mergezms.cpp mediain.cpp zedin.cpp mediaout.cpp zmsout.cpp portable_binary_oarchive.cpp portable_binary_iarchive.cpp ZvSettings.cpp)
target_link_libraries( mergezms ${Boost_LIBRARIES} ${OpenCV_LIBS} ${ZED_LIBRARIES} ${LibTinyXML2})
CUDA_ADD_EXECUTABLE(predict_one predict_one.cpp CaffeClassifier.cpp GIEClassifier.cpp Classifier.cpp zca.cpp zca.cu classifierio.cpp)
target_link_libraries( predict_one ${Boost_LIBRARIES} ${OpenCV_LIBS} ${LibCaffe} ${LibGLOG} ${LibProtobuf} ${CUDA_LIBRARIES} ${LibNVCaffeParser} ${LibNVInfer})
CUDA_ADD_EXECUTABLE(rank_imagelist rank_imagelist.cpp CaffeClassifier.cpp GIEClassifier.cpp Classifier.cpp zca.cpp zca.cu classifierio.cpp)
target_link_libraries( rank_imagelist ${Boost_LIBRARIES} ${OpenCV_LIBS} ${LibCaffe} ${LibGLOG} ${LibProtobuf} ${CUDA_LIBRARIES} ${LibNVCaffeParser} ${LibNVInfer})
#add_executable(depthtest depthtest.cpp)
#target_link_libraries( depthtest ${OpenCV_LIBS} )
