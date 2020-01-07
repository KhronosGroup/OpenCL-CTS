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
)

include(../CMakeCommon.txt)

