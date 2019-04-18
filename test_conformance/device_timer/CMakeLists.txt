set(MODULE_NAME DEVICE_TIMER)

set(${MODULE_NAME}_SOURCES
    main.c
    test_device_timer.c
    ../../test_common/harness/errorHelpers.c
    ../../test_common/harness/kernelHelpers.c
    ../../test_common/harness/testHarness.c
    ../../test_common/harness/parseParameters.cpp
    ../../test_common/harness/msvc9.c
    ../../test_common/harness/crc32.c
)

include(../CMakeCommon.txt)
