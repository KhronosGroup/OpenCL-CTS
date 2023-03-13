set(MODULE_NAME RELATIONALS)

set(${MODULE_NAME}_SOURCES
    main.cpp
    test_relationals.cpp
    test_comparisons_float.cpp
    test_comparisons_double.cpp
    test_shuffles.cpp
)

set(${MODULE_NAME}_KERNEL_LANG clc)
set(${MODULE_NAME}_TESTS
relational_any
relational_all
relational_bitselect
relational_select_signed
relational_select_unsigned
relational_isequal
relational_isnotequal
relational_isgreater
relational_isgreaterequal
relational_isless
relational_islessequal
relational_islessgreater
shuffle_copy
shuffle_function_call
shuffle_array_cast
shuffle_built_in
shuffle_built_in_dual_input
)

include(../CMakeCommon.cmake)

