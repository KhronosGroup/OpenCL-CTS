set(MODULE_NAME SUBGROUPS)

set(${MODULE_NAME}_SOURCES
    main.cpp
    test_barrier.cpp
    test_queries.cpp
    test_workitem.cpp
    test_workgroup.cpp
    ../../test_common/harness/errorHelpers.c
    ../../test_common/harness/testHarness.c
    ../../test_common/harness/kernelHelpers.c
    ../../test_common/harness/typeWrappers.cpp
    ../../test_common/harness/mt19937.c
    ../../test_common/harness/msvc9.c
    ../../test_common/harness/ThreadPool.c
    ../../test_common/harness/conversions.c
    ../../test_common/harness/parseParameters.cpp
    ../../test_common/harness/crc32.c
)

include(../CMakeCommon.txt)
