if(WIN32)
set(MODULE_NAME D3D11)

set(${MODULE_NAME}_SOURCES
    buffer.cpp
    texture2d.cpp
    texture3d.cpp
    misc.cpp
    main.cpp
    harness.cpp
)

list(APPEND CLConform_LIBRARIES d3d11 dxgi)

include(../CMakeCommon.txt)
else()
message(STATUS "D3D11 tests are only supported on Windows.")
endif()
