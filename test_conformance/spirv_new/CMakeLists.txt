# #####################################################################################################
# Copyright (c) 2016 The Khronos Group Inc. All Rights Reserved.
#
# This code is protected by copyright laws and contains material proprietary to the Khronos Group, Inc.
# This is UNPUBLISHED PROPRIETARY SOURCE CODE that may not be disclosed in whole or in part to
# third parties, and may not be reproduced, republished, distributed, transmitted, displayed,
# broadcast or otherwise exploited in any manner without the express prior written permission
# of Khronos Group. The receipt or possession of this code does not convey any rights to reproduce,
# disclose, or distribute its contents, or to manufacture, use, or sell anything that it may describe,
# in whole or in part other than under the terms of the Khronos Adopters Agreement
# or Khronos Conformance Test Source License Agreement as executed between Khronos and the recipient.
# #####################################################################################################

set(MODULE_NAME SPIRV_NEW)

file(GLOB SPIRV_NEW_SOURCES "*.cpp")

set(TEST_HARNESS_SOURCES
  ../../test_conformance/math_brute_force/reference_math.cpp
  ../../test_conformance/math_brute_force/utility.cpp
)

set(${MODULE_NAME}_SOURCES ${SPIRV_NEW_SOURCES} ${TEST_HARNESS_SOURCES})

set(${MODULE_NAME}_KERNEL_LANG clc)
set(${MODULE_NAME}_TESTS
  Reading SPIR-V files from default 'spirv_bin' path.
  In case you want to set other directory use '--spirv-binaries-path' argument.
  ext_cl_khr_spirv_no_integer_wrap_decoration_fadd_int
  ext_cl_khr_spirv_no_integer_wrap_decoration_fsub_int
  ext_cl_khr_spirv_no_integer_wrap_decoration_fmul_int
  ext_cl_khr_spirv_no_integer_wrap_decoration_fshiftleft_int
  ext_cl_khr_spirv_no_integer_wrap_decoration_fnegate_int
  ext_cl_khr_spirv_no_integer_wrap_decoration_fadd_uint
  ext_cl_khr_spirv_no_integer_wrap_decoration_fsub_uint
  ext_cl_khr_spirv_no_integer_wrap_decoration_fmul_uint
  ext_cl_khr_spirv_no_integer_wrap_decoration_fshiftleft_uint
  decorate_restrict
  decorate_aliased
  decorate_alignment
  decorate_constant
  decorate_cpacked
  decorate_saturated_conversion_char
  decorate_saturated_conversion_uchar
  decorate_saturated_conversion_short
  decorate_saturated_conversion_ushort
  decorate_saturated_conversion_int
  decorate_saturated_conversion_uint
  decorate_fp_rounding_mode_rte_float_int
  decorate_fp_rounding_mode_rtz_float_int
  decorate_fp_rounding_mode_rtp_float_int
  decorate_fp_rounding_mode_rtn_float_int
  decorate_fp_rounding_mode_rte_double_long
  decorate_fp_rounding_mode_rtz_double_long
  decorate_fp_rounding_mode_rtp_double_long
  decorate_fp_rounding_mode_rtn_double_long
  get_program_il
  linkage_export_function_compile
  linkage_import_function_compile
  linkage_import_function_link
  op_atomic_inc_global
  op_atomic_dec_global
  op_label_simple
  op_branch_simple
  op_unreachable_simple
  op_branch_conditional
  op_branch_conditional_weighted
  op_composite_construct_int4
  op_composite_construct_struct
  op_constant_true_simple
  op_constant_false_simple
  op_constant_int_simple
  op_constant_uint_simple
  op_constant_char_simple
  op_constant_uchar_simple
  op_constant_ushort_simple
  op_constant_long_simple
  op_constant_ulong_simple
  op_constant_short_simple
  op_constant_float_simple
  op_constant_double_simple
  op_constant_int4_simple
  op_constant_int3_simple
  op_constant_struct_int_float_simple
  op_constant_struct_int_char_simple
  op_constant_struct_struct_simple
  op_constant_half_simple
  op_copy_int_simple
  op_copy_uint_simple
  op_copy_char_simple
  op_copy_uchar_simple
  op_copy_ushort_simple
  op_copy_long_simple
  op_copy_ulong_simple
  op_copy_short_simple
  op_copy_float_simple
  op_copy_double_simple
  op_copy_int4_simple
  op_copy_int3_simple
  op_copy_struct_int_float_simple
  op_copy_struct_int_char_simple
  op_copy_struct_struct_simple
  op_copy_half_simple
  op_fadd_float_regular
  op_fsub_float_regular
  op_fmul_float_regular
  op_fdiv_float_regular
  op_fadd_float_fast
  op_fsub_float_fast
  op_fmul_float_fast
  op_fdiv_float_fast
  op_fadd_double_regular
  op_fsub_double_regular
  op_fmul_double_regular
  op_fdiv_double_regular
  op_fadd_double_fast
  op_fsub_double_fast
  op_fmul_double_fast
  op_fdiv_double_fast
  op_fadd_float4_regular
  op_fsub_float4_regular
  op_fmul_float4_regular
  op_fdiv_float4_regular
  op_fadd_float4_fast
  op_fsub_float4_fast
  op_fmul_float4_fast
  op_fdiv_float4_fast
  op_fadd_double2_regular
  op_fsub_double2_regular
  op_fmul_double2_regular
  op_fdiv_double2_regular
  op_fadd_double2_fast
  op_fsub_double2_fast
  op_fmul_double2_fast
  op_fdiv_double2_fast
  op_fadd_half_regular
  op_fsub_half_regular
  op_fmul_half_regular
  op_fdiv_half_regular
  op_fadd_half_fast
  op_fsub_half_fast
  op_fmul_half_fast
  op_fdiv_half_fast
  function_none
  function_inline
  function_noinline
  function_pure
  function_const
  function_pure_ptr
  op_lifetime_simple
  op_loop_merge_branch_none
  op_loop_merge_branch_unroll
  op_loop_merge_branch_dont_unroll
  op_loop_merge_branch_conditional_none
  op_loop_merge_branch_conditional_unroll
  op_loop_merge_branch_conditional_dont_unroll
  op_neg_float
  op_neg_double
  op_neg_int
  op_neg_long
  op_not_int
  op_not_long
  op_neg_short
  op_not_short
  op_neg_float4
  op_neg_int4
  op_not_int4
  op_type_opaque_simple
  op_phi_2_blocks
  op_phi_3_blocks
  op_phi_4_blocks
  op_selection_merge_if_none
  op_selection_merge_if_flatten
  op_selection_merge_if_dont_flatten
  op_selection_merge_swith_none
  op_selection_merge_swith_flatten
  op_selection_merge_swith_dont_flatten
  op_spec_constant_uint_simple
  op_spec_constant_uchar_simple
  op_spec_constant_ushort_simple
  op_spec_constant_ulong_simple
  op_spec_constant_float_simple
  op_spec_constant_half_simple
  op_spec_constant_double_simple
  op_spec_constant_true_simple
  op_spec_constant_false_simple
  op_undef_true_simple
  op_undef_false_simple
  op_undef_int_simple
  op_undef_uint_simple
  op_undef_char_simple
  op_undef_uchar_simple
  op_undef_ushort_simple
  op_undef_long_simple
  op_undef_ulong_simple
  op_undef_short_simple
  op_undef_float_simple
  op_undef_double_simple
  op_undef_int4_simple
  op_undef_int3_simple
  op_undef_struct_int_float_simple
  op_undef_struct_int_char_simple
  op_undef_struct_struct_simple
  op_undef_half_simple
  op_vector_int4_extract
  op_vector_float4_extract
  op_vector_long2_extract
  op_vector_double2_extract
  op_vector_char16_extract
  op_vector_int4_insert
  op_vector_float4_insert
  op_vector_long2_insert
  op_vector_double2_insert
  op_vector_char16_insert
  op_vector_times_scalar_float
  op_vector_times_scalar_double
)
