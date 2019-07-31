set(MODULE_NAME C11_ATOMICS)

set(${MODULE_NAME}_SOURCES
    common.cpp
    host_atomics.cpp
    main.cpp
    test_atomics.cpp
    ../../test_common/harness/ThreadPool.c
    ../../test_common/harness/errorHelpers.c
    ../../test_common/harness/testHarness.c
    ../../test_common/harness/kernelHelpers.c
    ../../test_common/harness/mt19937.c
    ../../test_common/harness/msvc9.c
    ../../test_common/harness/parseParameters.cpp
    ../../test_common/harness/crc32.c
)

include(../CMakeCommon.txt)
