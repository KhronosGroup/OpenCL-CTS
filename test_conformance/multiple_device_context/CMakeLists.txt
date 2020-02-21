set(MODULE_NAME multiples)

set(${MODULE_NAME}_SOURCES
    main.cpp
    test_multiple_contexts.cpp
    test_multiple_devices.cpp
)


set_source_files_properties(
        COMPILE_FLAGS -msse2)

include(../CMakeCommon.txt)
