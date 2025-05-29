cmake_minimum_required(VERSION 3.13)

set(project_name lab6)
project(${project_name})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_COMPILER "pgc++")

find_package(Boost REQUIRED COMPONENTS program_options)
add_executable(${project_name} lab6.cpp)

option(USE_ACC_HOST "Use OpenACC host" OFF)
if(USE_ACC_HOST)
    target_compile_options(${project_name} PRIVATE -acc=host -Minfo=all)
    target_link_options(${project_name} PRIVATE -acc=host -Minfo=all)
else()
    target_compile_options(${project_name} PRIVATE -acc=multicore -Minfo=all)
    target_link_options(${project_name} PRIVATE -acc=multicore -Minfo=all)
endif()

target_compile_options(${project_name} PRIVATE -O2)
target_link_libraries(${project_name} Boost::program_options)
