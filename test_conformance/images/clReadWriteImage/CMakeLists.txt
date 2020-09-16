set(MODULE_NAME CL_READ_WRITE_IMAGES)

set(${MODULE_NAME}_SOURCES
    main.cpp
    test_read_1D.cpp
    test_read_1D_array.cpp
    test_read_2D.cpp
    test_read_2D_array.cpp
    test_loops.cpp
    test_read_3D.cpp
    ../common.cpp
)

include(../../CMakeCommon.txt)

