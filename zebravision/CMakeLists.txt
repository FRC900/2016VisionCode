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

set (CAFFE_BASE_DIR "/home/ubuntu/caffe")

include_directories(/home/ubuntu/opencv-2.4.13/build/include
	../caffe_classifier
	${ZED_INCLUDE_DIRS}
	${CUDA_INCLUDE_DIRS}
	${CAFFE_BASE_DIR}/build/install/include
	${CAFFE_BASE_DIR}/build/include
	${CAFFE_BASE_DIR}/include
	${Boost_INCLUDE_DIR}
	${ZMQ_INCLUDE_DIRS}
   	${EIGEN3_INCLUDE_DIR}
	.
	../fovis
	/home/ubuntu/2016VisionCode/libfovis/build/include/fovis)

link_directories(/home/ubuntu/opencv-2.4.13/build/lib)
link_directories(${ZED_LIBRARY_DIR})
link_directories(${CUDA_LIBRARY_DIRS})
link_directories(/home/ubuntu/2016VisionCode/libfovis/build/lib/)

add_executable(zv
	objdetect.cpp
	GoalDetector.cpp
	detect.cpp
	Classifier.cpp
	CaffeClassifier.cpp
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
	ZvSettings.cpp
	zv.cpp)

find_library (LibCaffe caffe PATH  ${CAFFE_BASE_DIR}/build/lib)
find_library (LibGLOG  glog)
find_library (LibProtobuf protobuf)
find_library (LibTinyXML2 tinyxml2)

target_link_libraries( zv
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
	${LibProtobuf}
	${ZMQ_LIBRARIES}
	${LibTinyXML2})

#add_executable(convertzms convertzms.cpp mediain.cpp zedin.cpp mediaout.cpp zmsout.cpp portable_binary_oarchive.cpp portable_binary_iarchive.cpp ZvSettings.cpp)
#target_link_libraries( convertzms ${Boost_LIBRARIES} ${OpenCV_LIBS} ${ZED_LIBRARIES} ${LibTinyXML2})
#add_executable(mergezms mergezms.cpp mediain.cpp zedin.cpp mediaout.cpp zmsout.cpp portable_binary_oarchive.cpp portable_binary_iarchive.cpp ZvSettings.cpp)
#target_link_libraries( mergezms ${Boost_LIBRARIES} ${OpenCV_LIBS} ${ZED_LIBRARIES} ${LibTinyXML2})
add_executable(predict_one predict_one.cpp CaffeClassifier.cpp Classifier.cpp zca.cpp classifierio.cpp)
target_link_libraries( predict_one ${Boost_LIBRARIES} ${OpenCV_LIBS} ${LibCaffe} ${LibGLOG} ${LibProtobuf} )
add_executable(rank_imagelist rank_imagelist.cpp CaffeClassifier.cpp Classifier.cpp zca.cpp classifierio.cpp)
target_link_libraries( rank_imagelist ${Boost_LIBRARIES} ${OpenCV_LIBS} ${LibCaffe} ${LibGLOG} ${LibProtobuf} )
#add_executable(fpstest fpstest.cpp objdetect.cpp mediain.cpp videoin.cpp ../caffe_classifier/detect.cpp ../caffe_classifier/CaffeBatchPrediction.cpp ../caffe_classifier/fast_nms.cpp ../caffe_classifier/scalefactor.cpp)
#target_link_libraries( fpstest ${OpenCV_LIBS} )
