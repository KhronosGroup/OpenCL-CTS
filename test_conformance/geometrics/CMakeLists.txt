set(MODULE_NAME GEOMETRICS)

set(${MODULE_NAME}_SOURCES
    main.cpp
    test_geometrics_base.cpp
    test_geometrics_cross.cpp
    test_geometrics_two_vec_to_scalar.cpp
    test_geometrics_vec_to_vec.cpp
    test_geometrics_vec_to_scalar.cpp
)

set_gnulike_module_compile_flags("-Wno-sign-compare")

include(../CMakeCommon.txt)

