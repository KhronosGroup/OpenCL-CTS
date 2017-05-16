set(MODULE_NAME PROFILING)

set(${MODULE_NAME}_SOURCES
    main.c
    readArray.c
    writeArray.c
    readImage.c
    writeImage.c
    copy.c
    execute.c
    execute_multipass.c
    ../../test_common/harness/testHarness.c
    ../../test_common/harness/errorHelpers.c
    ../../test_common/harness/typeWrappers.cpp
    ../../test_common/harness/imageHelpers.cpp
    ../../test_common/harness/kernelHelpers.c
    ../../test_common/harness/mt19937.c
    ../../test_common/harness/conversions.c
    ../../test_common/harness/msvc9.c
)

include(../CMakeCommon.txt)
