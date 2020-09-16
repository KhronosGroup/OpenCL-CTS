//
// Copyright (c) 2017 The Khronos Group Inc.
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
#include "harness/errorHelpers.h"
#include "harness/kernelHelpers.h"
#include "harness/threadTesting.h"
#include "harness/typeWrappers.h"
#include "harness/testHarness.h"
#include "harness/mt19937.h"


// The number of errors to print out for each test
#define MAX_ERRORS_TO_PRINT 10

extern const size_t vector_aligns[];

extern int      create_program_and_kernel(const char *source, const char *kernel_name, cl_program *program_ret, cl_kernel *kernel_ret);
extern void fill_test_values( cl_long *outBufferA, cl_long *outBufferB, size_t numElements, MTdata d );


extern int test_popcount(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements);
extern int test_integer_clz(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_integer_ctz(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_integer_hadd(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_integer_rhadd(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_integer_mul_hi(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_integer_rotate(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_integer_clamp(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_integer_mad_sat(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_integer_mad_hi(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_integer_min(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_integer_max(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_integer_upsample(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);

extern int test_integer_addAssign(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_integer_subtractAssign(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_integer_multiplyAssign(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_integer_divideAssign(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_integer_moduloAssign(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_integer_andAssign(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_integer_orAssign(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_integer_exclusiveOrAssign(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);

extern int test_integer_abs(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_integer_abs_diff(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_integer_add_sat(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_integer_sub_sat(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);

extern int test_integer_mul24(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_integer_mad24(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);


extern int test_long_math(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_long_logic(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_long_shift(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_long_compare(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_ulong_math(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_ulong_logic(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_ulong_shift(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_ulong_compare(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);

extern int test_int_math(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_int_logic(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_int_shift(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_int_compare(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_uint_math(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_uint_logic(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_uint_shift(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_uint_compare(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);

extern int test_short_math(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_short_logic(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_short_shift(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_short_compare(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_ushort_math(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_ushort_logic(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_ushort_shift(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_ushort_compare(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);

extern int test_char_math(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_char_logic(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_char_shift(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_char_compare(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_uchar_math(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_uchar_logic(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_uchar_shift(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_uchar_compare(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);


extern int test_quick_long_math(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_quick_long_logic(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_quick_long_shift(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_quick_long_compare(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_quick_ulong_math(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_quick_ulong_logic(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_quick_ulong_shift(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_quick_ulong_compare(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);

extern int test_quick_int_math(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_quick_int_logic(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_quick_int_shift(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_quick_int_compare(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_quick_uint_math(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_quick_uint_logic(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_quick_uint_shift(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_quick_uint_compare(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);

extern int test_quick_short_math(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_quick_short_logic(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_quick_short_shift(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_quick_short_compare(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_quick_ushort_math(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_quick_ushort_logic(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_quick_ushort_shift(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_quick_ushort_compare(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);

extern int test_quick_char_math(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_quick_char_logic(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_quick_char_shift(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_quick_char_compare(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_quick_uchar_math(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_quick_uchar_logic(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_quick_uchar_shift(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_quick_uchar_compare(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);

extern int test_unary_ops_full(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_unary_ops_increment(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_unary_ops_decrement(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);

extern int test_vector_scalar(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);

