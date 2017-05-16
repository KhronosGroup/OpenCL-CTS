set(MODULE_NAME RELATIONALS)

set(${MODULE_NAME}_SOURCES
    main.c
    test_relationals.cpp
    test_comparisons_float.cpp
    test_comparisons_double.cpp
    test_shuffles.cpp
    ../../test_common/harness/errorHelpers.c
    ../../test_common/harness/threadTesting.c
    ../../test_common/harness/testHarness.c
    ../../test_common/harness/kernelHelpers.c
    ../../test_common/harness/mt19937.c
    ../../test_common/harness/conversions.c
    ../../test_common/harness/msvc9.c
    ../../test_common/harness/parseParameters.cpp
)

include(../CMakeCommon.txt)

