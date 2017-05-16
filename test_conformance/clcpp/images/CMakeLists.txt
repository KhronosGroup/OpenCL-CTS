set(MODULE_NAME CPP_IMAGES)

set(${MODULE_NAME}_SOURCES
    main.cpp
    ../../../test_common/harness/errorHelpers.c
    ../../../test_common/harness/testHarness.c
    ../../../test_common/harness/kernelHelpers.c
    ../../../test_common/harness/msvc9.c
    ../../../test_common/harness/parseParameters.cpp
    ../../../test_common/harness/mt19937.c
    ../../../test_common/harness/conversions.c
    ../../../test_common/harness/imageHelpers.cpp
)

include(../../CMakeCommon.txt)
