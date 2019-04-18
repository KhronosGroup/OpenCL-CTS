set(MODULE_NAME ALLOCATIONS)

set(${MODULE_NAME}_SOURCES
        main.cpp
        allocation_execute.cpp
        allocation_fill.cpp
        allocation_functions.cpp
        allocation_utils.cpp
        ../../test_common/harness/errorHelpers.c
        ../../test_common/harness/threadTesting.c
        ../../test_common/harness/kernelHelpers.c
        ../../test_common/harness/testHarness.c
        ../../test_common/harness/typeWrappers.cpp
        ../../test_common/harness/mt19937.c
        ../../test_common/harness/msvc9.c
        ../../test_common/harness/parseParameters.cpp
        ../../test_common/harness/crc32.c
)

include(../CMakeCommon.txt)
