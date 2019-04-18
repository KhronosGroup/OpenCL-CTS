set(MODULE_NAME CONTRACTIONS)

set(${MODULE_NAME}_SOURCES
        contractions.c
        ../../test_common/harness/mt19937.c
        ../../test_common/harness/msvc9.c
        ../../test_common/harness/mingw_compat.c
        ../../test_common/harness/errorHelpers.c
        ../../test_common/harness/rounding_mode.c
        ../../test_common/harness/kernelHelpers.c
        ../../test_common/harness/testHarness.c
        ../../test_common/harness/parseParameters.cpp
        ../../test_common/harness/crc32.c
)

include(../CMakeCommon.txt)
