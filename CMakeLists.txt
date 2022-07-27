cmake_minimum_required(VERSION 3.0)
project(BundleFusion)

add_definitions(-w)
add_compile_options(-std=c++11)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fpermissive -pthread -lrt -fPIC")
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

option(VISUALIZATION "Build BundleFusion With Pangolin" ON)

find_package(CUDA REQUIRED)
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_NVCC_FLAGS -arch=sm_61;-std=c++11;-lm)
ADD_DEFINITIONS(-fpermissive)

find_package (Threads)
find_package (Eigen3 REQUIRED)
find_package (OpenCV REQUIRED)

# find_package(Eigen)
include_directories( "/usr/include/eigen3" )


file(GLOB_RECURSE cpp_file src/*.cpp)
file(GLOB_RECURSE cu_file src/*.cu)

if(VISUALIZATION)
  find_package(Pangolin 0.2 QUIET)
  message("-- found PANGOLIN, add visualization plug.")
  if(Pangolin_FOUND)
    find_package(Boost COMPONENTS system thread)
    add_definitions(-DWITH_VISUALIZATION)
    include_directories(
      ${Pangolin_INCLUDE_DIRS}
      ${Boost_INCLUDE_DIRS}
      ${PROJECT_SOURCE_DIR}/include/Visualization/
    )
    
  endif()
else()
  message("-- not found PANGOLIN, visualization not enable.")
  file(GLOB_RECURSE Visualization_CPP_Files src/PangolinOutputWrapper.cpp)
  file(GLOB_RECURSE Visualization_CU_Files src/PangolinOutputWrapper.cu)
  list(REMOVE_ITEM cpp_file ${Visualization_CPP_Files})
  list(REMOVE_ITEM cu_file ${Visualization_CU_Files}) 
endif()

# message("opencv path")
# message(${OpenCV_INCLUDE_DIRS})


add_compile_options(-std=c++11)

include_directories(
  ${CUDA_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PROJECT_SOURCE_DIR}/include/
  ${PROJECT_SOURCE_DIR}/include/mLib/include
  ${PROJECT_SOURCE_DIR}/include/mLibExternal
  ${PROJECT_SOURCE_DIR}/include/cutil/cutil/inc/
  ${PROJECT_SOURCE_DIR}/include/FreeImage/
  ${PROJECT_SOURCE_DIR}/include/interface/
  ${PROJECT_SOURCE_DIR}/include/DepthSensing/
  ${PROJECT_SOURCE_DIR}/include/SiftGPU/
  ${PROJECT_SOURCE_DIR}/include/Solver/
  ${PROJECT_SOURCE_DIR}/include/maskrcnn/
  ${PROJECT_SOURCE_DIR}/include/gSLICr/
  ${PROJECT_SOURCE_DIR}/include/gSLICr/gSLICr_Lib/engines

  #${catkin_INCLUDE_DIRS}
)

 
cuda_add_library(BundleFusion  SHARED
  ${cu_file} 
  ${cpp_file}
)

target_link_libraries(BundleFusion 
  cuda 
  freeimage
  ${CUDA_LIBRARIES}
  ${OpenCV_LIBS}
)

if(Pangolin_FOUND)
  target_link_libraries(BundleFusion
    ${Pangolin_LIBRARIES}
    boost_system 
    boost_thread
  )
endif()

# install(EXPORT gputsdfrecon-targets
#     NAMESPACE gputsdfrecon::
#     FILE gputsdfrecon-config.cmake
#     DESTINATION lib/cmake/gputsdfrecon)

add_executable(bundle_fusion_example example/bundle_fusion_example.cpp)
target_link_libraries(bundle_fusion_example 
   BundleFusion

   "/usr/lib/x86_64-linux-gnu/libnvinfer.so.6"
   "/usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.6.0.1"
   #"/home/yxh/TensorRT/TensorRT-6.0.1.5/lib/libnvinfer.so.6.0.1"
   #"/home/yxh/TensorRT/TensorRT-6.0.1.5/lib/libnvinfer_plugin.so.6.0.1"
   "/home/yxh/TensorRT/TensorRT-6.0.1.5/lib/libnvparsers.so.6.0.1"
   "/home/yxh/TensorRT/TensorRT-6.0.1.5/lib/libnvonnxparser.so.6.0.1"
)

include_directories(/home/yxh/TensorRT/TensorRT-6.0.1.5/include/)
link_directories(/home/yxh/TensorRT/TensorRT-6.0.1.5/lib/)

# 
# include_directories(
# ${PROJECT_SOURCE_DIR}/example/include/)
# 
# add_executable(example_depthfusion example/example_depthfusion.cpp example/native.cpp)
# target_link_libraries(example_depthfusion 
#   gpu_tsdf_bf
#   )
#   
# add_executable(example_rkd_fusion example/example_rkd_fusion.cpp example/native.cpp)
# target_link_libraries(example_rkd_fusion gpu_tsdf_bf)
# 
# add_executable(example_rkd_fusion_cd example/example_rkd_fusion_cd.cpp example/native.cpp)
# 
# target_link_libraries(example_rkd_fusion_cd gpu_tsdf_bf)
  
#add_executable(example_tsdf_integrate_orbslam example/example_orbslam_tsdf_integrate.cpp)
#target_link_libraries(example_tsdf_integrate_orbslam 
#  gpu_tsdf_bf
#  )

