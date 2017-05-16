set(MODULE_NAME SELECT)

set(${MODULE_NAME}_SOURCES
        test_select.c
        util_select.c
        ../../test_common/harness/testHarness.c
        ../../test_common/harness/mt19937.c
        ../../test_common/harness/msvc9.c
        ../../test_common/harness/kernelHelpers.c
        ../../test_common/harness/errorHelpers.c
)

include(../CMakeCommon.txt)
