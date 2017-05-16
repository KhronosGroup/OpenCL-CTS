set(MODULE_NAME GENERIC_ADDRESS_SPACE)

set(${MODULE_NAME}_SOURCES
    advanced_tests.cpp
    basic_tests.cpp
    main.cpp
    stress_tests.cpp
    ../../test_common/harness/testHarness.c
    ../../test_common/harness/kernelHelpers.c
    ../../test_common/harness/errorHelpers.c
    ../../test_common/harness/mt19937.c
    ../../test_common/harness/msvc9.c
)

include(../CMakeCommon.txt)

# end of file #
