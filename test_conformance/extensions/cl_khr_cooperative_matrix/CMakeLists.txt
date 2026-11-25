if(CLConform_SPIRV_TOOLS_TARGET)
  set(MODULE_NAME CL_KHR_COOPERATIVE_MATRIX)

  set(${MODULE_NAME}_SOURCES
      cooperative_matrix.cpp
      main.cpp
      program.cpp
  )

  include(../../CMakeCommon.txt)
  target_link_libraries(${${MODULE_NAME}_OUT} ${CLConform_SPIRV_TOOLS_TARGET})
else()
  message(STATUS "SPIRV-Tools not found; cl_khr_cooperative_matrix tests will not be built")
endif()
