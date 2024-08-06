set(MODULE_NAME INTEGER_OPS)

if(CMAKE_COMPILER_IS_GNUCC OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "(Apple)?Clang")
    add_cxx_flag_if_supported(-Wno-narrowing)
endif()

set(${MODULE_NAME}_SOURCES
    main.cpp
    test_int_basic_ops.cpp
    test_integers.cpp
    test_upsample.cpp
    test_intmul24.cpp test_intmad24.cpp
    test_sub_sat.cpp test_add_sat.cpp
    test_abs.cpp test_absdiff.cpp
    test_unary_ops.cpp
    verification_and_generation_functions.cpp
    test_popcount.cpp
    test_integer_dot_product.cpp
    test_extended_bit_ops_extract.cpp
    test_extended_bit_ops_insert.cpp
    test_extended_bit_ops_reverse.cpp
)

include(../CMakeCommon.txt)

