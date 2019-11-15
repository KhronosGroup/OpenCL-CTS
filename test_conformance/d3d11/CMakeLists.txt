if(WIN32)

set(D3D11_INCLUDE_DIR $ENV{NV_TOOLS}/sdk/DirectX_Aug2009/Include)

if(${ARCH} STREQUAL "i686")
set(D3D11_LIB_DIR $ENV{NV_TOOLS}/sdk/DirectX_Aug2009/Lib/x86)
endif(${ARCH} STREQUAL "i686")

if(${ARCH} STREQUAL "x86_64")
set(D3D11_LIB_DIR $ENV{NV_TOOLS}/sdk/DirectX_Aug2009/Lib/x64)
endif(${ARCH} STREQUAL "x86_64")

list(APPEND CLConform_INCLUDE_DIR ${D3D11_INCLUDE_DIR})
include_directories (${CLConform_SOURCE_DIR}/test_common/harness
                     ${CLConform_INCLUDE_DIR} )
link_directories(${CL_LIB_DIR}, ${D3D11_LIB_DIR})

list(APPEND CLConform_LIBRARIES d3d11 dxgi)

set(MODULE_NAME D3D11)

set(${MODULE_NAME}_SOURCES
    buffer.cpp
    texture2d.cpp
    texture3d.cpp
    misc.cpp
    main.cpp
    harness.cpp
)

set_source_files_properties(
    ${MODULE_NAME}_SOURCES
    PROPERTIES LANGUAGE CXX)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include(../CMakeCommon.txt)
endif(WIN32)
