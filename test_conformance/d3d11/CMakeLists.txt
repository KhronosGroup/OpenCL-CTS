set(MODULE_NAME D3D11)

if(WIN32 AND D3D11_IS_SUPPORTED)
set(${MODULE_NAME}_SOURCES
    buffer.cpp
    texture2d.cpp
    texture3d.cpp
    misc.cpp
    main.cpp
    harness.cpp
)

list(APPEND CLConform_LIBRARIES d3d11 dxgi)
else()
build_as_stub(${MODULE_NAME})
endif()

include(../CMakeCommon.txt)
