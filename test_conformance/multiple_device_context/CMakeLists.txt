set(MODULE_NAME multiples)

set(${MODULE_NAME}_SOURCES
    main.c
    test_multiple_contexts.c
    test_multiple_devices.cpp
    ../../test_common/harness/errorHelpers.c
    ../../test_common/harness/threadTesting.c
    ../../test_common/harness/testHarness.c
    ../../test_common/harness/kernelHelpers.c
    ../../test_common/harness/genericThread.cpp
    ../../test_common/harness/typeWrappers.cpp
    ../../test_common/harness/conversions.c
    ../../test_common/harness/mt19937.c
    ../../test_common/harness/msvc9.c
)


set_source_files_properties(
        COMPILE_FLAGS -msse2)

set_source_files_properties(
    main.c
    test_multiple_contexts.c
    ../../test_common/harness/errorHelpers.c
    ../../test_common/harness/threadTesting.c
    ../../test_common/harness/testHarness.c
    ../../test_common/harness/kernelHelpers.c
    ../../test_common/harness/conversions.c
    ../../test_common/harness/msvc9.c
        PROPERTIES LANGUAGE CXX)

include(../CMakeCommon.txt)
