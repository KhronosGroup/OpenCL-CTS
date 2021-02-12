set(MODULE_NAME BRUTEFORCE)

set(${MODULE_NAME}_SOURCES
    binary.cpp
    binary_i.cpp
    binary_operator.cpp
    binary_two_results_i.cpp
    function_list.cpp
    i_unary.cpp
    macro_binary.cpp
    macro_unary.cpp
    mad.cpp
    main.cpp
    reference_math.cpp
    sleep.cpp
    ternary.cpp
    unary.cpp
    unary_two_results.cpp
    unary_two_results_i.cpp
    unary_u.cpp
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
