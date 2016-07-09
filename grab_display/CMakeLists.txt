cmake_minimum_required(VERSION 2.8)
set(CMAKE_LEGACY_CYGWIN_WIN32 0) # Remove when CMake >= 2.8.4 is required
project( grab_chroma )
set(CMAKE_BUILD_TYPE Release )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -std=c++11")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -flto")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -Ofast -flto")

find_package( OpenCV REQUIRED )
find_package( Boost COMPONENTS filesystem system thread program_options REQUIRED )

include_directories(../framegrabber)
include_directories( ${Boost_INCLUDE_DIR} )

add_executable( ColorSharpen ColorSharpen.cpp)
target_link_libraries( ColorSharpen ${OpenCV_LIBS} )
add_executable( grab_chroma grab_chroma.cpp chroma_key.cpp image_warp.cpp)
target_link_libraries( grab_chroma ${OpenCV_LIBS} )
add_executable( imageShift imageShift.cpp  chroma_key.cpp image_warp.cpp)
target_link_libraries( imageShift ${OpenCV_LIBS} )

add_executable( zcacalc zcacalc.cpp zca.cpp ../framegrabber/utilities_common.cpp )
target_link_libraries( zcacalc ${OpenCV_LIBS} ${Boost_LIBRARIES} )
add_executable( zcarun zcarun.cpp zca.cpp ../framegrabber/utilities_common.cpp )
target_link_libraries( zcarun ${OpenCV_LIBS} ${Boost_LIBRARIES} )
