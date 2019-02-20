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
#if !defined(_WIN32)
#include <unistd.h>
#endif

#include <stdio.h>
#include <stdlib.h>

#if !defined(_WIN32)
#include <stdbool.h>
#endif

#include <math.h>
#include <string.h>
#include "../../test_common/harness/testHarness.h"
#include "procs.h"

basefn    basefn_list[] = {
    test_hostptr,
    test_fpmath_float,
    test_fpmath_float2,
    test_fpmath_float4,
    test_intmath_int,
    test_intmath_int2,
    test_intmath_int4,
    test_intmath_long,
    test_intmath_long2,
    test_intmath_long4,
    test_hiloeo,
    test_if,
    test_sizeof,
    test_loop,
    test_pointer_cast,
    test_local_arg_def,
    test_local_kernel_def,
    test_local_kernel_scope,
    test_constant,
    test_constant_source,
    test_readimage,
    test_readimage_int16,
    test_readimage_fp32,
    test_writeimage,
    test_writeimage_int16,
    test_writeimage_fp32,
    test_multireadimageonefmt,

    test_multireadimagemultifmt,
    test_image_r8,
    test_barrier,
    test_int2float,
    test_float2int,
    test_imagereadwrite,
    test_imagereadwrite3d,
    test_readimage3d,
    test_readimage3d_int16,
    test_readimage3d_fp32,
    test_bufferreadwriterect,
    test_arrayreadwrite,
    test_arraycopy,
    test_imagearraycopy,
    test_imagearraycopy3d,
    test_imagecopy,
    test_imagecopy3d,
    test_imagerandomcopy,
    test_arrayimagecopy,
    test_arrayimagecopy3d,
    test_imagenpot,

    test_vload_global,
    test_vload_local,
    test_vload_constant,
    test_vload_private,
    test_vstore_global,
    test_vstore_local,
    test_vstore_private,

    test_createkernelsinprogram,
    test_imagedim_pow2,
    test_imagedim_non_pow2,
    test_image_param,
    test_image_multipass_integer_coord,
    test_image_multipass_float_coord,
    test_explicit_s2v_bool,
    test_explicit_s2v_char,
    test_explicit_s2v_uchar,
    test_explicit_s2v_short,
    test_explicit_s2v_ushort,
    test_explicit_s2v_int,
    test_explicit_s2v_uint,
    test_explicit_s2v_long,
    test_explicit_s2v_ulong,
    test_explicit_s2v_float,
    test_explicit_s2v_double,

    test_enqueue_map_buffer,
    test_enqueue_map_image,

    test_work_item_functions,

    test_astype,

    test_async_copy_global_to_local,
    test_async_copy_local_to_global,
    test_async_strided_copy_global_to_local,
    test_async_strided_copy_local_to_global,
    test_prefetch,

    test_kernel_call_kernel_function,
    test_host_numeric_constants,
    test_kernel_numeric_constants,
    test_kernel_limit_constants,
    test_kernel_preprocessor_macros,

    test_basic_parameter_types,
    test_vector_creation,
    test_vec_type_hint,
    test_kernel_memory_alignment_local,
    test_kernel_memory_alignment_global,
    test_kernel_memory_alignment_constant,
    test_kernel_memory_alignment_private,

    test_global_work_offsets,
    test_get_global_offset
};

const char    *basefn_names[] = {
    "hostptr",
    "fpmath_float",
    "fpmath_float2",
    "fpmath_float4",
    "intmath_int",
    "intmath_int2",
    "intmath_int4",
    "intmath_long",
    "intmath_long2",
    "intmath_long4",
    "hiloeo",
    "if",
    "sizeof",
    "loop",
    "pointer_cast",
    "local_arg_def",
    "local_kernel_def",
    "local_kernel_scope",
    "constant",
    "constant_source",
    "readimage",
    "readimage_int16",
    "readimage_fp32",
    "writeimage",
    "writeimage_int16",
    "writeimage_fp32",
    "mri_one",

    "mri_multiple",
    "image_r8",
    "barrier",
    "int2float",
    "float2int",
    "imagereadwrite",
    "imagereadwrite3d",
    "readimage3d",
    "readimage3d_int16",
    "readimage3d_fp32",
    "bufferreadwriterect",
    "arrayreadwrite",
    "arraycopy",
    "imagearraycopy",
    "imagearraycopy3d",
    "imagecopy",
    "imagecopy3d",
    "imagerandomcopy",
    "arrayimagecopy",
    "arrayimagecopy3d",
    "imagenpot",

    "vload_global",
    "vload_local",
    "vload_constant",
    "vload_private",
    "vstore_global",
    "vstore_local",
    "vstore_private",

    "createkernelsinprogram",
    "imagedim_pow2",
    "imagedim_non_pow2",
    "image_param",
    "image_multipass_integer_coord",
    "image_multipass_float_coord",
    "explicit_s2v_bool",
    "explicit_s2v_char",
    "explicit_s2v_uchar",
    "explicit_s2v_short",
    "explicit_s2v_ushort",
    "explicit_s2v_int",
    "explicit_s2v_uint",
    "explicit_s2v_long",
    "explicit_s2v_ulong",
    "explicit_s2v_float",
    "explicit_s2v_double",

    "enqueue_map_buffer",
    "enqueue_map_image",

    "work_item_functions",

    "astype",

    "async_copy_global_to_local",
    "async_copy_local_to_global",
    "async_strided_copy_global_to_local",
    "async_strided_copy_local_to_global",
    "prefetch",

    "kernel_call_kernel_function",
    "host_numeric_constants",
    "kernel_numeric_constants",
    "kernel_limit_constants",
    "kernel_preprocessor_macros",

    "parameter_types",

    "vector_creation",
    "vec_type_hint",

    "kernel_memory_alignment_local",
    "kernel_memory_alignment_global",
    "kernel_memory_alignment_constant",
    "kernel_memory_alignment_private",

    "global_work_offsets",
    "get_global_offset",

    "all",
};

ct_assert((sizeof(basefn_names) / sizeof(basefn_names[0]) - 1) == (sizeof(basefn_list) / sizeof(basefn_list[0])));

int    num_fns = sizeof(basefn_names) / sizeof(char *);


int main(int argc, const char *argv[])
{
    int err = runTestHarness( argc, argv, num_fns, basefn_list, basefn_names, false, false, 0 );
    return err;
}



