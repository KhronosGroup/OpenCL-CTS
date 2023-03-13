set(MODULE_NAME GENERIC_ADDRESS_SPACE)

set(${MODULE_NAME}_SOURCES
    advanced_tests.cpp
    basic_tests.cpp
    main.cpp
    stress_tests.cpp
)

set(${MODULE_NAME}_KERNEL_LANG clc)
set(${MODULE_NAME}_TESTS
function_get_fence
function_to_address_space
variable_get_fence
variable_to_address_space
casting
conditional_casting
chain_casting
ternary_operator_casting
language_struct
language_union
multiple_calls_same_function
compare_pointers
library_function
generic_variable_volatile
generic_variable_const
generic_variable_gentype
builtin_functions
generic_advanced_casting
generic_ptr_to_host_mem
generic_ptr_to_host_mem_svm
max_number_of_params
)

include(../CMakeCommon.cmake)

# end of file #
