include_guard()

function(add_axon_compiler name)
  set(target axon_compiler_${name})
  add_executable(${target} ${ARGN})
  target_link_libraries(${target} PRIVATE axon::compiler)

  add_executable(axon::compiler::${name} ALIAS ${target})
endfunction()

macro(axon_compiler_generate name header)
  if(NOT IS_ABSOLUTE "${header}")
    set(header "${CMAKE_CURRENT_BINARY_DIR}/${header}")
  endif()
  add_custom_command(OUTPUT "${header}"
    COMMAND $<TARGET_FILE:axon::compiler::${name}> -o "${header}"
    DEPENDS axon::compiler::${name}
  )
endmacro()
