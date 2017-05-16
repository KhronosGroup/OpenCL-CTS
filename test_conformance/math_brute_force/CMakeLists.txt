set(MODULE_NAME BRUTEFORCE)

set(${MODULE_NAME}_SOURCES
    FunctionList.c
    Sleep.c
    binary.c
    binaryOperator.c
    Utility.c
    binary_i.c
    binary_two_results_i.c
    i_unary.c
    macro_binary.c
    macro_unary.c
    mad.c
    main.c
    reference_math.c
    ternary.c
    unary.c
    unary_two_results.c
    unary_two_results_i.c
    unary_u.c
    ../../test_common/harness/rounding_mode.c
    ../../test_common/harness/ThreadPool.c
    ../../test_common/harness/mt19937.c
    ../../test_common/harness/msvc9.c
    ../../test_common/harness/parseParameters.cpp
    ../../test_common/harness/kernelHelpers.c
    ../../test_common/harness/errorHelpers.c
)


if (NOT ANDROID)
set_source_files_properties(
    FunctionList.c
    Sleep.c
    binary.c
    binaryOperator.c
    Utility.c
    binary_i.c
    binary_two_results_i.c
    i_unary.c
    macro_binary.c
    macro_unary.c
    mad.c
    main.c     reference_math.c
    ternary.c     unary.c     unary_two_results.c
    unary_two_results_i.c unary_u.c
    COMPILE_FLAGS -msse2    )
endif(NOT ANDROID)

set_source_files_properties(
    FunctionList.c
    Sleep.c
    binary.c
    binaryOperator.c
    Utility.c
    binary_i.c
    binary_two_results_i.c
    i_unary.c
    macro_binary.c
    macro_unary.c
    mad.c
    main.c
    reference_math.c
    ternary.c
    unary.c
    unary_two_results.c
    unary_two_results_i.c unary_u.c
    ../../test_common/harness/rounding_mode.c
    ../../test_common/harness/ThreadPool.c
    ../../test_common/harness/msvc9.c
    ../../test_common/harness/parseParameters.cpp
    PROPERTIES LANGUAGE CXX)

if (NOT CMAKE_CL_64 AND NOT MSVC AND NOT ANDROID)
set_source_files_properties(
${MODULE_NAME}_SOURCES
    COMPILE_FLAGS -march=i686)
endif (NOT CMAKE_CL_64 AND NOT MSVC AND NOT ANDROID)

if(CMAKE_COMPILER_IS_GNUCC)
set_source_files_properties(
        COMPILE_FLAGS -O0)
endif(CMAKE_COMPILER_IS_GNUCC)

include(../CMakeCommon.txt)
