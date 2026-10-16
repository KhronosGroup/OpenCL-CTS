set(MODULE_NAME D3D10)

set(${MODULE_NAME}_SOURCES main.cpp)

if(WIN32 AND D3D10_IS_SUPPORTED)
  list(APPEND ${MODULE_NAME}_SOURCES
    buffer.cpp
    texture2d.cpp
    texture3d.cpp
    misc.cpp
    harness.cpp
  )
  list(APPEND CLConform_LIBRARIES directx_wrapper)
  add_compile_definitions(D3D10_IS_SUPPORTED)
endif()

include(../CMakeCommon.txt)
