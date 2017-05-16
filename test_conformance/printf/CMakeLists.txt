set(MODULE_NAME PRINTF)

set(${MODULE_NAME}_SOURCES
         test_printf.c
         util_printf.c
         ../../test_common/harness/errorHelpers.c
         ../../test_common/harness/threadTesting.c
         ../../test_common/harness/kernelHelpers.c
         ../../test_common/harness/typeWrappers.cpp
         ../../test_common/harness/conversions.c
         ../../test_common/harness/mt19937.c
         ../../test_common/harness/msvc9.c
         #../../test_common/harness/imageHelpers.cpp
         ../../test_common/harness/parseParameters.cpp
         ../../test_common/harness/kernelHelpers.c
         )

include(../CMakeCommon.txt)
