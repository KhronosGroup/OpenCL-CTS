if(WIN32)
set(MODULE_NAME D3D10)

set(${MODULE_NAME}_SOURCES
    buffer.cpp
    texture2d.cpp
    texture3d.cpp
    misc.cpp
    main.cpp
    harness.cpp
)

list(APPEND CLConform_LIBRARIES d3d10 dxgi)

include(../CMakeCommon.txt)
else()
message(STATUS "D3D10 tests are only supported on Windows.")
endif()
