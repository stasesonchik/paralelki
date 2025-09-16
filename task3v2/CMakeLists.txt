cmake_minimum_required(VERSION 3.16)
project(paralell2)

set(CMAKE_CXX_STANDARD 17)

# Указываем путь к LLVM из Homebrew
set(LLVM_PATH "/opt/homebrew/opt/llvm")
set(CMAKE_C_COMPILER "${LLVM_PATH}/bin/clang")
set(CMAKE_CXX_COMPILER "${LLVM_PATH}/bin/clang++")

# Добавляем флаги для компиляции и линковки OpenMP
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -I${LLVM_PATH}/include")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${LLVM_PATH}/lib -lomp")

add_executable(task3v2 main.cpp)
