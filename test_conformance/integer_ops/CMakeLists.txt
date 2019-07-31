set(MODULE_NAME INTEGER_OPS)

set(${MODULE_NAME}_SOURCES
    main.c
    test_int_basic_ops.c
    test_integers.cpp
    test_upsample.cpp
    test_intmul24.c test_intmad24.c
    test_sub_sat.c test_add_sat.c
    test_abs.c test_absdiff.c
    test_unary_ops.cpp
    verification_and_generation_functions.c
    test_popcount.c
    ../../test_common/harness/ThreadPool.c
    ../../test_common/harness/mt19937.c
    ../../test_common/harness/conversions.c
    ../../test_common/harness/errorHelpers.c
    ../../test_common/harness/threadTesting.c
    ../../test_common/harness/testHarness.c
    ../../test_common/harness/kernelHelpers.c
    ../../test_common/harness/msvc9.c
    ../../test_common/harness/parseParameters.cpp
    ../../test_common/harness/crc32.c
)

include(../CMakeCommon.txt)

