cmake_minimum_required(VERSION 3.20)

project(axon)

option(AXON_WARNINGS "Whether or not to enable compiler warnings." OFF)

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
  src/Expr.cpp
  src/ExprPrinter.cpp
  src/ExprVisitor.cpp
  src/Module.cpp
  src/Value.cpp
  src/Optimizer.cpp
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

add_executable(main
  main.cpp
)
target_link_libraries(main PRIVATE axon)
