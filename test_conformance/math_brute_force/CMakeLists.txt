set(MODULE_NAME BRUTEFORCE)

set(${MODULE_NAME}_SOURCES
    binary_double.cpp
    binary_float.cpp
    binary_i_double.cpp
    binary_i_float.cpp
    binary_operator_double.cpp
    binary_operator_float.cpp
    binary_two_results_i_double.cpp
    binary_two_results_i_float.cpp
    function_list.cpp
    i_unary_double.cpp
    i_unary_float.cpp
    macro_binary_double.cpp
    macro_binary_float.cpp
    macro_unary_double.cpp
    macro_unary_float.cpp
    mad_double.cpp
    mad_float.cpp
    main.cpp
    reference_math.cpp
    sleep.cpp
    ternary_double.cpp
    ternary_float.cpp
    unary_double.cpp
    unary_float.cpp
    unary_two_results_double.cpp
    unary_two_results_float.cpp
    unary_two_results_i_double.cpp
    unary_two_results_i_float.cpp
    unary_u_double.cpp
    unary_u_float.cpp
    utility.cpp
)

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
