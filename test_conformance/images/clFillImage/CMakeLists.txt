
set(MODULE_NAME CL_FILL_IMAGES)

set(${MODULE_NAME}_SOURCES
    main.cpp
    test_fill_1D.cpp
    test_fill_1D_array.cpp
    test_fill_2D.cpp
    test_fill_2D_array.cpp
    test_fill_generic.cpp
    test_loops.cpp
    test_fill_3D.cpp
#    test_fill_2D_3D.cpp
    ../common.cpp
)


include(../../CMakeCommon.txt)
