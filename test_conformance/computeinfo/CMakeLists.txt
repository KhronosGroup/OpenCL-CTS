set(MODULE_NAME computeinfo)

set(${MODULE_NAME}_SOURCES
        main.c
        ../../test_common/harness/errorHelpers.c
        ../../test_common/harness/msvc9.c
        ../../test_common/harness/kernelHelpers.c
        ../../test_common/harness/testHarness.c
)

include(../CMakeCommon.txt)
