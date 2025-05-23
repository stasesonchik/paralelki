cmake_minimum_required(VERSION 3.17)
project(heat2d_solver)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Boost REQUIRED COMPONENTS program_options)

option(USE_GPU "Build with GPU support" ON)
option(USE_MULTICORE "Build multicore-version for CPU" OFF)

set(CMAKE_CXX_COMPILER nvc++)  # или pgc++ если nvc++ недоступен

set(CUDA_HOME "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4")  # поправь под сервер

if(USE_GPU)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -acc -gpu=cc70 -stdpar=gpu -Minfo=accel -O3 -g")
    message(STATUS "Building with GPU support")
else()
  if(USE_MULTICORE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -acc=multicore -Minfo=all -fast")
    message(STATUS "Building CPU multicore version")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -acc=host -Minfo=all -fast")
    message(STATUS "Building CPU single-core version")
  endif()
endif()

add_executable(heat_solver main.cpp heat_solver.cpp boundary.cpp)

target_include_directories(heat_solver PRIVATE include ${Boost_INCLUDE_DIRS} ${CUDA_HOME}/include)
target_link_libraries(heat_solver PRIVATE Boost::program_options)

target_link_directories(heat_solver PRIVATE ${CUDA_HOME}/lib64)

if (USE_GPU)
  target_link_libraries(heat_solver PRIVATE cudart)
endif()
