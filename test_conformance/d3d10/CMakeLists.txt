if(WIN32)

set(D3D10_INCLUDE_DIR $ENV{NV_TOOLS}/sdk/DirectX_Aug2009/Include)

if(${ARCH} STREQUAL "i686")
set(D3D10_LIB_DIR $ENV{NV_TOOLS}/sdk/DirectX_Aug2009/Lib/x86)
endif(${ARCH} STREQUAL "i686")

if(${ARCH} STREQUAL "x86_64")
set(D3D10_LIB_DIR $ENV{NV_TOOLS}/sdk/DirectX_Aug2009/Lib/x64)
endif(${ARCH} STREQUAL "x86_64")

list(APPEND CLConform_INCLUDE_DIR ${D3D10_INCLUDE_DIR})
include_directories (${CLConform_SOURCE_DIR}/test_common/harness
                     ${CLConform_INCLUDE_DIR} )
link_directories(${CL_LIB_DIR}, ${D3D10_LIB_DIR})

list(APPEND CLConform_LIBRARIES d3d10 dxgi)


set(D3D10_SOURCES
    buffer.cpp
    texture2d.cpp
    texture3d.cpp
    misc.cpp
    main.cpp
    harness.cpp
)

add_executable(conformance_test_d3d10
    ${D3D10_SOURCES})

set_source_files_properties(
    ${D3D10_SOURCES}
    PROPERTIES LANGUAGE CXX)

TARGET_LINK_LIBRARIES(conformance_test_d3d10 harness
    ${CLConform_LIBRARIES})

endif(WIN32)
