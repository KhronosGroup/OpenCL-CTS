set(MODULE_NAME CL_READ_WRITE_IMAGES)

set(${MODULE_NAME}_SOURCES
    main.cpp
    test_read_1D.cpp
    test_read_1D_buffer.cpp
    test_read_1D_array.cpp
    test_read_2D.cpp
    test_read_2D_array.cpp
    test_loops.cpp
    test_read_3D.cpp
    ../common.cpp
)

set_gnulike_module_compile_flags("-Wno-unused-but-set-variable")

include(../../CMakeCommon.txt)

