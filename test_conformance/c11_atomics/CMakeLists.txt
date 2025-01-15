set(MODULE_NAME C11_ATOMICS)

set(${MODULE_NAME}_SOURCES
    common.cpp
    host_atomics.cpp
    main.cpp
    test_atomics.cpp
)

set_gnulike_module_compile_flags("-Wno-sign-compare -Wno-format")

include(../CMakeCommon.txt)
