set(MODULE_NAME HALF)

set(${MODULE_NAME}_SOURCES
        cl_utils.c
        Test_vLoadHalf.c
        Test_roundTrip.c
        Test_vStoreHalf.c main.c
        ../../test_common/harness/msvc9.c
        ../../test_common/harness/mingw_compat.c
        ../../test_common/harness/errorHelpers.c
        ../../test_common/harness/kernelHelpers.c
        ../../test_common/harness/ThreadPool.c
        ../../test_common/harness/testHarness.c
        ../../test_common/harness/parseParameters.cpp
)

include(../CMakeCommon.txt)

