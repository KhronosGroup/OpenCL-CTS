set(MODULE_NAME GEOMETRICS)

set(${MODULE_NAME}_SOURCES
    main.cpp
    test_geometrics_double.cpp
    test_geometrics.cpp
)

set(${MODULE_NAME}_KERNEL_LANG clc)
set(${MODULE_NAME}_TESTS
geom_cross
geom_dot
geom_distance
geom_fast_distance
geom_length
geom_fast_length
geom_normalize
geom_fast_normalize
)

include(../CMakeCommon.cmake)

