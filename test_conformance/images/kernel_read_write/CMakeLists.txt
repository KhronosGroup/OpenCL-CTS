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
    ../common.cpp
)

include(../../CMakeCommon.txt)

