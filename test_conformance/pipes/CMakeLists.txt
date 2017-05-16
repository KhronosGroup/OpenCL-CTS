set(MODULE_NAME pipes)

set(${MODULE_NAME}_SOURCES
    main.c
    test_pipe_read_write.c
    test_pipe_info.c
    test_pipe_limits.c
    test_pipe_query_functions.c
    test_pipe_readwrite_errors.c
    test_pipe_subgroups.c
    ../../test_common/harness/errorHelpers.c
    ../../test_common/harness/threadTesting.c
    ../../test_common/harness/testHarness.c
    ../../test_common/harness/kernelHelpers.c
    ../../test_common/harness/typeWrappers.cpp
    ../../test_common/harness/mt19937.c
    ../../test_common/harness/conversions.c
    ../../test_common/harness/msvc9.c
    ../../test_common/harness/parseParameters.cpp
)

include(../CMakeCommon.txt)
