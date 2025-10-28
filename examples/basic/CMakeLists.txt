cmake_minimum_required(VERSION 3.20)

add_axon_compiler(basic compiler.cpp)

axon_compiler_generate(basic basic.h)

add_executable(train
  train.c
  "${CMAKE_CURRENT_BINARY_DIR}/basic.h"
)

target_include_directories(train PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
