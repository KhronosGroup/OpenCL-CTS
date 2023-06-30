//
// Copyright (c) 2023 The Khronos Group Inc.
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
#include "harness/compat.h"

#if !defined(_WIN32)
#include <unistd.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <CL/cl_half.h>

#include "harness/testHarness.h"
#include "procs.h"

test_definition test_list[] = {
    ADD_TEST(hostptr),
    ADD_TEST(fpmath),
    ADD_TEST(intmath_int),
    ADD_TEST(intmath_int2),
    ADD_TEST(intmath_int4),
    ADD_TEST(intmath_long),
    ADD_TEST(intmath_long2),
    ADD_TEST(intmath_long4),
    ADD_TEST(hiloeo),
    ADD_TEST(if),
    ADD_TEST(sizeof),
    ADD_TEST(loop),
    ADD_TEST(pointer_cast),
    ADD_TEST(local_arg_def),
    ADD_TEST(local_kernel_def),
    ADD_TEST(local_kernel_scope),
    ADD_TEST(constant),
    ADD_TEST(constant_source),
    ADD_TEST(readimage),
    ADD_TEST(readimage_int16),
    ADD_TEST(readimage_fp32),
    ADD_TEST(writeimage),
    ADD_TEST(writeimage_int16),
    ADD_TEST(writeimage_fp32),
    ADD_TEST(mri_one),

    ADD_TEST(mri_multiple),
    ADD_TEST(image_r8),
    ADD_TEST(barrier),
    ADD_TEST_VERSION(wg_barrier, Version(2, 0)),
    ADD_TEST(int2fp),
    ADD_TEST(fp2int),
    ADD_TEST(imagereadwrite),
    ADD_TEST(imagereadwrite3d),
    ADD_TEST(readimage3d),
    ADD_TEST(readimage3d_int16),
    ADD_TEST(readimage3d_fp32),
    ADD_TEST(bufferreadwriterect),
    ADD_TEST(arrayreadwrite),
    ADD_TEST(arraycopy),
    ADD_TEST(imagearraycopy),
    ADD_TEST(imagearraycopy3d),
    ADD_TEST(imagecopy),
    ADD_TEST(imagecopy3d),
    ADD_TEST(imagerandomcopy),
    ADD_TEST(arrayimagecopy),
    ADD_TEST(arrayimagecopy3d),
    ADD_TEST(imagenpot),

    ADD_TEST(vload_global),
    ADD_TEST(vload_local),
    ADD_TEST(vload_constant),
    ADD_TEST(vload_private),
    ADD_TEST(vstore_global),
    ADD_TEST(vstore_local),
    ADD_TEST(vstore_private),

    ADD_TEST(createkernelsinprogram),
    ADD_TEST(imagedim_pow2),
    ADD_TEST(imagedim_non_pow2),
    ADD_TEST(image_param),
    ADD_TEST(image_multipass_integer_coord),
    ADD_TEST(image_multipass_float_coord),
    ADD_TEST(explicit_s2v_char),
    ADD_TEST(explicit_s2v_uchar),
    ADD_TEST(explicit_s2v_short),
    ADD_TEST(explicit_s2v_ushort),
    ADD_TEST(explicit_s2v_int),
    ADD_TEST(explicit_s2v_uint),
    ADD_TEST(explicit_s2v_long),
    ADD_TEST(explicit_s2v_ulong),
    ADD_TEST(explicit_s2v_float),
    ADD_TEST(explicit_s2v_double),

    ADD_TEST(enqueue_map_buffer),
    ADD_TEST(enqueue_map_image),

    ADD_TEST(work_item_functions),

    ADD_TEST(astype),

    ADD_TEST(async_copy_global_to_local),
    ADD_TEST(async_copy_local_to_global),
    ADD_TEST(async_strided_copy_global_to_local),
    ADD_TEST(async_strided_copy_local_to_global),
    ADD_TEST(async_copy_global_to_local2D),
    ADD_TEST(async_copy_local_to_global2D),
    ADD_TEST(async_copy_global_to_local3D),
    ADD_TEST(async_copy_local_to_global3D),
    ADD_TEST(async_work_group_copy_fence_import_after_export_aliased_local),
    ADD_TEST(async_work_group_copy_fence_import_after_export_aliased_global),
    ADD_TEST(
        async_work_group_copy_fence_import_after_export_aliased_global_and_local),
    ADD_TEST(async_work_group_copy_fence_export_after_import_aliased_local),
    ADD_TEST(async_work_group_copy_fence_export_after_import_aliased_global),
    ADD_TEST(
        async_work_group_copy_fence_export_after_import_aliased_global_and_local),
    ADD_TEST(prefetch),
    ADD_TEST(kernel_call_kernel_function),
    ADD_TEST(host_numeric_constants),
    ADD_TEST(kernel_numeric_constants),
    ADD_TEST(kernel_limit_constants),
    ADD_TEST(kernel_preprocessor_macros),
    ADD_TEST(parameter_types),
    ADD_TEST(vector_creation),
    ADD_TEST(vector_swizzle),
    ADD_TEST(vec_type_hint),
    ADD_TEST(kernel_memory_alignment_local),
    ADD_TEST(kernel_memory_alignment_global),
    ADD_TEST(kernel_memory_alignment_constant),
    ADD_TEST(kernel_memory_alignment_private),

    ADD_TEST_VERSION(progvar_prog_scope_misc, Version(2, 0)),
    ADD_TEST_VERSION(progvar_prog_scope_uninit, Version(2, 0)),
    ADD_TEST_VERSION(progvar_prog_scope_init, Version(2, 0)),
    ADD_TEST_VERSION(progvar_func_scope, Version(2, 0)),

    ADD_TEST(global_work_offsets),
    ADD_TEST(get_global_offset),

    ADD_TEST_VERSION(global_linear_id, Version(2, 0)),
    ADD_TEST_VERSION(local_linear_id, Version(2, 0)),
    ADD_TEST_VERSION(enqueued_local_size, Version(2, 0)),

    ADD_TEST(simple_read_image_pitch),
    ADD_TEST(simple_write_image_pitch),

#if defined(__APPLE__)
    ADD_TEST(queue_priority),
#endif

    ADD_TEST_VERSION(get_linear_ids, Version(2, 0)),
    ADD_TEST_VERSION(rw_image_access_qualifier, Version(2, 0)),
};

const int test_num = ARRAY_SIZE( test_list );
cl_half_rounding_mode halfRoundingMode = CL_HALF_RTE;

test_status InitCL(cl_device_id device)
{
    if (is_extension_available(device, "cl_khr_fp16"))
    {
        const cl_device_fp_config fpConfigHalf =
            get_default_rounding_mode(device, CL_DEVICE_HALF_FP_CONFIG);
        if ((fpConfigHalf & CL_FP_ROUND_TO_NEAREST) != 0)
        {
            halfRoundingMode = CL_HALF_RTE;
        }
        else if ((fpConfigHalf & CL_FP_ROUND_TO_ZERO) != 0)
        {
            halfRoundingMode = CL_HALF_RTZ;
        }
        else
        {
            log_error("Error while acquiring half rounding mode");
            return TEST_FAIL;
        }
    }

    return TEST_PASS;
}

int main(int argc, const char *argv[])
{
    return runTestHarnessWithCheck(argc, argv, test_num, test_list, false, 0,
                                   InitCL);
}

