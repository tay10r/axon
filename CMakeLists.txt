cmake_minimum_required(VERSION 3.20)

project(axon)

option(AXON_WARNINGS    "Whether to enable compiler warnings."          OFF)
option(AXON_EXAMPLES    "Whether to build the example programs."        OFF)
option(AXON_GPU_SUPPORT "Whether to build with GPU support."             ON)
option(AXON_IPO         "Whether to enable interprocedural optimziation" ON)

include(cmake/CMakeRC.cmake)

if(AXON_GPU_SUPPORT)

  find_package(OpenGL REQUIRED COMPONENTS EGL)

  set(gpu_sources
    include/axon/Device.h
    include/axon/DeviceBuffer.h
    include/axon/DeviceFactory.h
    include/axon/DeviceProgram.h
    src/Device.h
    src/Device.cpp
    src/DeviceBuffer.cpp
    src/DeviceFactory.cpp
    src/DeviceProgram.cpp
    src/OpenGL.h
    src/OpenGL.cpp
  )

  cmrc_add_resource_library(axon_shaders
    shaders/row_reduce.glsl
    shaders/interpolate.glsl
  )

endif()

add_library(axon
  include/axon/Exception.h
  include/axon/Expr.h
  include/axon/ExprPrinter.h
  include/axon/ExprVisitor.h
  include/axon/Matrix.h
  include/axon/Module.h
  include/axon/ModuleBuilder.h
  include/axon/Value.h
  include/axon/Optimizer.h
  include/axon/Dataset.h
  include/axon/Interpreter.h
  src/VFloat.h
  src/Exception.cpp
  src/Expr.cpp
  src/ExprPrinter.cpp
  src/ExprVisitor.cpp
  src/Module.cpp
  src/Value.cpp
  src/Optimizer.cpp
  src/Dataset.cpp
  src/Interpreter.cpp
  ${gpu_sources}
)

if(AXON_IPO)
  set_target_properties(axon
    PROPERTIES
      INTERPROCEDURAL_OPTIMIZATION ON
  )
endif()

target_compile_features(axon PRIVATE cxx_std_20)

if(AXON_GPU_SUPPORT)

  target_compile_definitions(axon PRIVATE AXON_GPU_SUPPORT=1)

  target_link_libraries(axon PUBLIC OpenGL::EGL axon_shaders)

endif()

add_library(axon::axon ALIAS axon)

if(AXON_WARNINGS)

  if(CMAKE_COMPILER_IS_GNUCXX)
    target_compile_options(axon PRIVATE -Wall -Wextra -Werror -Wfatal-errors -Wconversion)
  endif()

endif()

target_include_directories(axon
  PUBLIC
    include
)

if(AXON_EXAMPLES)

  function(add_example name)
    set(target axon_example_${name})
    add_executable(${target} ${ARGN})
    target_link_libraries(${target} PRIVATE axon)
    set_target_properties(${target} PROPERTIES OUTPUT_NAME ${name})
    target_compile_definitions(${target} PRIVATE "CMAKE_CURRENT_SOURCE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"")
    target_compile_features(${target} PRIVATE cxx_std_20)
    if(AXON_IPO)
      set_target_properties(${target} PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON)
    endif()
  endfunction()

  add_example(minimal examples/minimal/main.cpp)

  add_example(image_encoder
    examples/image_encoder/main.cpp
    examples/image_encoder/deps/stb_image.h
    examples/image_encoder/deps/stb_image.c
    examples/image_encoder/deps/stb_image_write.h
    examples/image_encoder/deps/stb_image_write.c
  )

  add_example(device_api examples/device_api/main.cpp)

endif()
