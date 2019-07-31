set(MODULE_NAME THREAD_DIMENSIONS)

set(${MODULE_NAME}_SOURCES
        main.c
    test_thread_dimensions.c
    ../../test_common/harness/errorHelpers.c
    ../../test_common/harness/threadTesting.c
    ../../test_common/harness/testHarness.c
    ../../test_common/harness/kernelHelpers.c
    ../../test_common/harness/typeWrappers.cpp
    ../../test_common/harness/mt19937.c
    ../../test_common/harness/conversions.c
    ../../test_common/harness/msvc9.c
    ../../test_common/harness/parseParameters.cpp
    ../../test_common/harness/crc32.c
)

set_source_files_properties(
        COMPILE_FLAGS -msse2)

include(../CMakeCommon.txt)
