cmake_minimum_required(VERSION 3.20)

project(axon)

option(AXON_EXAMPLES "Whether to build the examples." OFF)

add_subdirectory(compiler)

if(AXON_EXAMPLES)
  add_subdirectory(examples/basic)
  add_subdirectory(examples/image_encoder)
endif()
