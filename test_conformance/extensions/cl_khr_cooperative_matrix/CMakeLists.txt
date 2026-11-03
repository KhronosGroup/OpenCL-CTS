include(FindPkgConfig)

# first try locating SPIRV-Tools via pkgconfig (the old way)
pkg_search_module(SPIRV_TOOLS SPIRV-Tools)
if (NOT SPIRV_TOOLS_FOUND)
  # then try locating SPIRV-Tools via cmake (the new way)
  find_package(SPIRV-Tools)
  find_package(SPIRV-Tools-tools)
  if (SPIRV-Tools_FOUND AND SPIRV-Tools-tools_FOUND)
    set(SPIRV_TOOLS_FOUND TRUE)
    # check for the existence of library targets in the found packages
    if(TARGET SPIRV-Tools-shared)
      # use the shared libary target if present
      set(SPIRV-Tools-library SPIRV-Tools-shared)
    elseif(TARGET SPIRV-Tools-static)
      # otherwise fallback to the static library target
      set(SPIRV-Tools-library SPIRV-Tools-static)
    else()
      message(FATAL_ERROR "Found SPIRV-Tools package but neither "
              "SPIRV-Tools-shared or SPIRV-Tools-static targets exist.")
    endif()
    set(SPIRV_TOOLS_LDFLAGS ${SPIRV-Tools-library})
    get_target_property(SPIRV_TOOLS_INCLUDE_DIRS ${SPIRV-Tools-library} INTERFACE_INCLUDE_DIRECTORIES)
  endif()
endif()

if (SPIRV_TOOLS_FOUND)
  set(MODULE_NAME CL_KHR_COOPERATIVE_MATRIX)

  set(${MODULE_NAME}_SOURCES
      cooperative_matrix.cpp
      main.cpp
      program.cpp
  )

  include(../../CMakeCommon.txt)
  target_include_directories(${${MODULE_NAME}_OUT} PRIVATE ${SPIRV_TOOLS_INCLUDE_DIRS})
  target_link_libraries(${${MODULE_NAME}_OUT} ${SPIRV_TOOLS_LDFLAGS})
else()
  message(STATUS "SPIRV-Tools not found; cl_khr_cooperative_matrix tests will not be built")
endif()
