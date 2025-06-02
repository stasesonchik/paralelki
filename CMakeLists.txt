cmake_minimum_required(VERSION 3.22)

set(project_name lab8)
project(${project_name} LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_COMPILER "nvc++")

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(CUDAToolkit REQUIRED)

add_executable(${project_name} lab8.cu)

target_compile_options(${project_name} PRIVATE -O2)

target_link_libraries(${project_name} 
  Boost::program_options 
  CUDA::cudart
)
