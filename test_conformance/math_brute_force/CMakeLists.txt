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
