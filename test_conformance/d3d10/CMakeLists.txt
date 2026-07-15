set(MODULE_NAME D3D10)

if(WIN32 AND D3D10_IS_SUPPORTED)
set(${MODULE_NAME}_SOURCES
    buffer.cpp
    texture2d.cpp
    texture3d.cpp
    misc.cpp
    main.cpp
    harness.cpp
)

list(APPEND CLConform_LIBRARIES directx_wrapper)
else()
build_as_stub(${MODULE_NAME})
endif()

include(../CMakeCommon.txt)
