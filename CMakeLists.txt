cmake_minimum_required(VERSION 3.17)

set(project_name lab7)
project(${project_name})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_COMPILER "pgc++")

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(CUDAToolkit REQUIRED)

add_executable(${project_name} lab7.cpp)

target_compile_options(${project_name} PRIVATE 
  -acc=gpu 
  -Minfo=all 
  -O2
)

target_link_options(${project_name} PRIVATE 
  -acc=gpu 
  -Minfo=all
  -O2
)

target_link_libraries(${project_name} 
  Boost::program_options 
  CUDA::cublas
)
