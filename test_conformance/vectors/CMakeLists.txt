set(MODULE_NAME VECTORS)

set(${MODULE_NAME}_SOURCES
    globals.cpp
        main.cpp
        structs.cpp
        test_step.cpp
        test_vec_align.cpp
        type_replacer.cpp
)

set(${MODULE_NAME}_KERNEL_LANG clc)
set(${MODULE_NAME}_TESTS
step_type
step_var
step_typedef_type
step_typedef_var
vec_align_array
vec_align_struct
vec_align_packed_struct
vec_align_struct_arr
vec_align_packed_struct_arr
)

include(../CMakeCommon.cmake)
