set(MODULE_NAME GEOMETRICS)

set(${MODULE_NAME}_SOURCES
    main.cpp
    test_geometrics_double.cpp
    test_geometrics.cpp
)

set_gnulike_module_compile_flags("-Wno-sign-compare")

include(../CMakeCommon.txt)

