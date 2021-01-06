set(MODULE_NAME SAMPLERLESS_READS)

set(${MODULE_NAME}_SOURCES
    main.cpp
    test_iterations.cpp
    test_loops.cpp
    test_read_1D.cpp
    test_read_3D.cpp
    test_read_1D_buffer.cpp
    test_read_1D_array.cpp
    test_read_2D_array.cpp
    ../common.cpp
)

include(../../CMakeCommon.txt)
