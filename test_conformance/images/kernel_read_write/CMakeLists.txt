set(MODULE_NAME IMAGE_STREAMS)

set(${MODULE_NAME}_SOURCES
    main.cpp
    test_common.cpp
    test_iterations.cpp
    test_loops.cpp
    test_read_1D.cpp
    test_read_1D_array.cpp
    test_read_2D_array.cpp
    test_read_3D.cpp
    test_write_image.cpp
    test_write_1D.cpp
    test_write_1D_array.cpp
    test_write_2D_array.cpp
    test_write_3D.cpp
    test_cl_ext_image_requirements_info.cpp
    test_cl_ext_image_from_buffer.cpp
    ../common.cpp
)

# Make unused variables not fatal in this module; see
# https://github.com/KhronosGroup/OpenCL-CTS/issues/1484
if(CMAKE_COMPILER_IS_GNUCC OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "(Apple)?Clang")
  SET_SOURCE_FILES_PROPERTIES(
    ${${MODULE_NAME}_SOURCES}
    PROPERTIES
    COMPILE_FLAGS "-Wno-error=unused-variable"
  )
endif()

include(../../CMakeCommon.txt)

