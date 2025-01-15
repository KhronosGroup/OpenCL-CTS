set(MODULE_NAME RELATIONALS)

set(${MODULE_NAME}_SOURCES
    main.cpp
    test_relationals.cpp
    test_comparisons_fp.cpp
    test_shuffles.cpp
)

set_gnulike_module_compile_flags("-Wno-format")

include(../CMakeCommon.txt)

