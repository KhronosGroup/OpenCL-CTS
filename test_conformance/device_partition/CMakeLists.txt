set(MODULE_NAME DEVICE_PARTITION)

set(${MODULE_NAME}_SOURCES
    main.c
    test_device_partition.cpp
    ../../test_common/harness/errorHelpers.c
    ../../test_common/harness/threadTesting.c
    ../../test_common/harness/testHarness.c
    ../../test_common/harness/kernelHelpers.c
    ../../test_common/harness/genericThread.cpp
    ../../test_common/harness/mt19937.c
    ../../test_common/harness/conversions.c
    ../../test_common/harness/typeWrappers.cpp
    ../../test_common/harness/msvc9.c
)
include(../CMakeCommon.txt)
