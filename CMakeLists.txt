cmake_minimum_required(VERSION 3.20)

project(axon)

option(AXON_WARNINGS "Whether to enable compiler warnings."   OFF)
option(AXON_EXAMPLES "Whether to build the example programs." OFF)
option(AXON_DEBUGGER "Whether to build the debugger."         OFF)

if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

add_library(axon
  include/axon/Expr.h
  include/axon/ExprPrinter.h
  include/axon/ExprVisitor.h
  include/axon/Module.h
  include/axon/ModuleBuilder.h
  include/axon/Value.h
  include/axon/Optimizer.h
  include/axon/Dataset.h
  include/axon/Interpreter.h
  src/Expr.cpp
  src/ExprPrinter.cpp
  src/ExprVisitor.cpp
  src/Module.cpp
  src/Value.cpp
  src/Optimizer.cpp
  src/Dataset.cpp
  src/Interpreter.cpp
)

if(AXON_WARNINGS)
  if(CMAKE_COMPILER_IS_GNUCXX)
    target_compile_options(axon PRIVATE -Wall -Wextra -Werror -Wfatal-errors -Wconversion)
  endif()
endif()

target_include_directories(axon
  PUBLIC
    include
)

add_library(axon::axon ALIAS axon)

if(AXON_EXAMPLES)

  function(add_example name)
    set(target axon_example_${name})
    add_executable(${target} ${ARGN})
    target_link_libraries(${target} PRIVATE axon)
    set_target_properties(${target} PROPERTIES OUTPUT_NAME ${name})
    target_compile_definitions(${target} PRIVATE "CMAKE_CURRENT_SOURCE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"")
  endfunction()

  add_example(minimal examples/minimal/main.cpp)

  add_example(image_encoder
    examples/image_encoder/main.cpp
    examples/image_encoder/deps/stb_image.h
    examples/image_encoder/deps/stb_image.c
    examples/image_encoder/deps/stb_image_write.h
    examples/image_encoder/deps/stb_image_write.c
  )

endif()
