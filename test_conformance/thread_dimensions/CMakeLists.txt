set(MODULE_NAME THREAD_DIMENSIONS)

set(${MODULE_NAME}_SOURCES
        main.cpp
    test_thread_dimensions.cpp
)

set_source_files_properties(
        COMPILE_FLAGS -msse2)

include(../CMakeCommon.txt)
