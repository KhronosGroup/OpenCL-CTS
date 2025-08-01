set(MODULE_NAME SPIRV_NEW)

set(${MODULE_NAME}_SOURCES
  main.cpp
  test_basic_versions.cpp
  test_cl_khr_expect_assume.cpp
  test_decorate.cpp
  test_extinst_printf.cpp
  test_get_program_il.cpp
  test_linkage.cpp
  test_no_integer_wrap_decoration.cpp
  test_op_atomic.cpp
  test_op_branch_conditional.cpp
  test_op_branch.cpp
  test_op_composite_construct.cpp
  test_op_constant.cpp
  test_op_copy_object.cpp
  test_op_fmath.cpp
  test_op_function.cpp
  test_op_lifetime.cpp
  test_op_loop_merge.cpp
  test_op_negate.cpp
  test_op_opaque.cpp
  test_op_phi.cpp
  test_op_selection_merge.cpp
  test_op_spec_constant.cpp
  test_op_undef.cpp
  test_op_vector_extract.cpp
  test_op_vector_insert.cpp
  test_op_vector_times_scalar.cpp
  test_spirv_14.cpp
  test_spirv_15.cpp
  test_spirv_16.cpp
)

set(TEST_HARNESS_SOURCES
  ../../test_conformance/math_brute_force/reference_math.cpp
  ../../test_conformance/math_brute_force/utility.cpp
)

set(${MODULE_NAME}_SOURCES ${${MODULE_NAME}_SOURCES} ${TEST_HARNESS_SOURCES})

include(../CMakeCommon.txt)

add_subdirectory(spirv_asm)
add_dependencies(${${MODULE_NAME}_OUT} spirv_new_binaries)
