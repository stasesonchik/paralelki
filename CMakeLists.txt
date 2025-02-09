cmake_minimum_required(VERSION 3.10)
project(lab1)


if(NOT DEFINED TYPE)
    set(TYPE "double")
endif()


add_definitions(-DTYPE=${TYPE})


add_executable(lab1 laba1.cpp)
