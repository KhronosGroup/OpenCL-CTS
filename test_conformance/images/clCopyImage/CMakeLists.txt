set(MODULE_NAME CL_COPY_IMAGES)

set(${MODULE_NAME}_SOURCES
    main.cpp
    test_copy_1D.cpp
    test_copy_1D_array.cpp
    test_copy_2D.cpp
    test_copy_2D_2D_array.cpp
    test_copy_2D_3D.cpp
    test_copy_2D_array.cpp
    test_copy_3D.cpp
    test_copy_3D_2D_array.cpp
    test_copy_1D_buffer.cpp
    test_copy_generic.cpp
    test_loops.cpp
    ../common.cpp
)

set_gnulike_module_compile_flags("-Wno-unused-but-set-variable")

include(../../CMakeCommon.txt)

