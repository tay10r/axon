cmake_minimum_required(VERSION 3.20)

add_axon_compiler(image_encoder compiler.cpp)

axon_compiler_generate(image_encoder image_encoder.h)

add_executable(axon_train_image_encoder
  train.c
  random.h
  random.cpp
  deps/stb_image.h
  deps/stb_image.c
  deps/stb_image_write.h
  deps/stb_image_write.c
  "${CMAKE_CURRENT_BINARY_DIR}/image_encoder.h"
)

target_include_directories(axon_train_image_encoder PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")

target_link_libraries(axon_train_image_encoder PRIVATE m)

if(CMAKE_COMPILER_IS_GNUCC)
  #target_compile_options(axon_train_image_encoder PRIVATE -ffast-math)
endif()
