set(MODULE_NAME D3D11)

set(${MODULE_NAME}_SOURCES main.cpp)

if(WIN32 AND D3D11_IS_SUPPORTED)
  list(APPEND ${MODULE_NAME}_SOURCES
    buffer.cpp
    texture2d.cpp
    texture3d.cpp
    misc.cpp
    harness.cpp
  )
  list(APPEND CLConform_LIBRARIES d3d11 dxgi)
  add_compile_definitions(D3D11_IS_SUPPORTED)
endif()

include(../CMakeCommon.txt)
