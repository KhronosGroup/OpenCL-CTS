set(MODULE_NAME THREAD_DIMENSIONS)

set(${MODULE_NAME}_SOURCES
        main.c
    test_thread_dimensions.c
)

set_source_files_properties(
        COMPILE_FLAGS -msse2)

include(../CMakeCommon.txt)
