set(MODULE_NAME multiples)

set(${MODULE_NAME}_SOURCES
    main.c
    test_multiple_contexts.c
    test_multiple_devices.cpp
)


set_source_files_properties(
        COMPILE_FLAGS -msse2)

set_source_files_properties(
    main.c
    test_multiple_contexts.c
        PROPERTIES LANGUAGE CXX)

include(../CMakeCommon.txt)
