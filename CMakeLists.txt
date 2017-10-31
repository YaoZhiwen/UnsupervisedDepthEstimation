cmake_minimum_required( VERSION 2.8 )
project( resTest )

add_executable( resTest resTest.cpp )
set( CMAKE_CXX_FLAGS "-std=c++11" )

find_package( PCL REQUIRED COMPONENTS common io )
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4") # use this in Ubuntu 16.04

add_definitions( ${PCL_DEFINITIONS} )

include_directories( /home/ai/Programs/Caffe/include 
  /home/ai/Programs/Caffe/.build_release/src
  /usr/local/cuda/include
  /usr/local/include
  /usr/include
  ${PCL_INCLUDE_DIRS}
  )
  
link_libraries( ${PCL_LIBRARY_DIRS} )
  
target_link_libraries( resTest
  /home/ai/Programs/Caffe/.build_release/lib/libcaffe.so 
  /usr/lib/x86_64-linux-gnu/libopencv_highgui.so
  /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so
  /usr/lib/x86_64-linux-gnu/libopencv_core.so
  /usr/lib/x86_64-linux-gnu/libboost_system.so  
  /usr/lib/x86_64-linux-gnu/libglog.so 
  ${PCL_LIBRARIES}
  )
