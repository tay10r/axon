cmake_minimum_required(VERSION 3.20)

project(axon_compiler)

#==========#
# Compiler #
#==========#

add_library(axon_compiler
  include/axon/compiler.hpp
  include/axon/matrix.hpp
  include/axon/module.hpp
  include/axon/module_builder.hpp
  include/axon/value.hpp
  include/axon/exception.hpp
  include/axon/expr.hpp
  include/axon/expr_visitor.hpp
  include/axon/exporter.hpp
  src/compiler.cpp
  src/module.cpp
  src/exception.cpp
  src/expr.cpp
  src/expr_visitor.cpp
  src/value.cpp
  src/exporter.cpp
  src/main.cpp
  src/c_exporter.hpp
  src/c_exporter.cpp
)

target_include_directories(axon_compiler PUBLIC include)

target_compile_Features(axon_compiler PUBLIC cxx_std_20)

add_library(axon::compiler ALIAS axon_compiler)

include(cmake/helpers.cmake)
