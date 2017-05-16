set(MODULE_NAME CPP_ATOMICS)

set(${MODULE_NAME}_SOURCES
    main.cpp
    ../../../test_common/harness/errorHelpers.c
    ../../../test_common/harness/testHarness.c
    ../../../test_common/harness/kernelHelpers.c
    ../../../test_common/harness/msvc9.c
    ../../../test_common/harness/parseParameters.cpp
)

include(../../CMakeCommon.txt)
