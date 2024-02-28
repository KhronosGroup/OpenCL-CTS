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

#include "harness/kernelHelpers.h"
#include "harness/testHarness.h"
#include "harness/errorHelpers.h"
#include "harness/typeWrappers.h"
#include "harness/conversions.h"
#include "harness/rounding_mode.h"

extern int      test_hostptr(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_fpmath(cl_device_id deviceID, cl_context context,
                       cl_command_queue queue, int num_elements);
extern int      test_intmath_int(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_intmath_int2(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_intmath_int4(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_intmath_long(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_intmath_long2(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_intmath_long4(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_hiloeo(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_if(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_sizeof(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_loop(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_pointer_cast(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_local_arg_def(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_local_kernel_def(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_local_kernel_scope(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_constant(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_constant_source(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_readimage(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_readimage_int16(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_readimage_fp32(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_writeimage(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_writeimage_int16(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_writeimage_fp32(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_mri_one(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_mri_multiple(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_image_r8(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_simplebarrier(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_barrier(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_wg_barrier(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_int2fp(cl_device_id deviceID, cl_context context,
                       cl_command_queue queue, int num_elements);
extern int test_fp2int(cl_device_id deviceID, cl_context context,
                       cl_command_queue queue, int num_elements);
extern int      test_imagearraycopy(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_imagearraycopy3d(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_imagereadwrite(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_imagereadwrite3d(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_readimage3d(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_readimage3d_int16(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_readimage3d_fp32(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_bufferreadwriterect(cl_device_id device, cl_context context, cl_command_queue queue_, int num_elements);
extern int      test_imagecopy(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_imagecopy3d(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_imagerandomcopy(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_arraycopy(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems);
extern int      test_arrayimagecopy(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_arrayimagecopy3d(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_imagenpot(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_sampler_float(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_sampler_int(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_createkernelsinprogram(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_single_large_allocation(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_multiple_max_allocation(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_arrayreadwrite(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_imagedim_pow2(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_imagedim_non_pow2(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_image_param(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_image_multipass_integer_coord(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_image_multipass_float_coord(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);

extern int      test_vload_global(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_vload_local(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_vload_constant(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_vload_private(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_vstore_global(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_vstore_local(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_vstore_private(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);

extern int test_explicit_s2v(cl_device_id deviceID, cl_context context,
                             cl_command_queue queue, int num_elements);

extern int      test_enqueue_map_buffer(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_enqueue_map_image(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);

extern int      test_work_item_functions(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);

extern int      test_astype(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);

extern int      test_native_kernel(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);

extern int      test_async_copy_global_to_local(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_async_copy_local_to_global(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_async_strided_copy_global_to_local(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_async_strided_copy_local_to_global(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_async_copy_global_to_local2D(cl_device_id deviceID,
                                             cl_context context,
                                             cl_command_queue queue,
                                             int num_elements);
extern int test_async_copy_local_to_global2D(cl_device_id deviceID,
                                             cl_context context,
                                             cl_command_queue queue,
                                             int num_elements);
extern int test_async_copy_global_to_local3D(cl_device_id deviceID,
                                             cl_context context,
                                             cl_command_queue queue,
                                             int num_elements);
extern int test_async_copy_local_to_global3D(cl_device_id deviceID,
                                             cl_context context,
                                             cl_command_queue queue,
                                             int num_elements);
extern int test_async_work_group_copy_fence_import_after_export_aliased_local(
    cl_device_id deviceID, cl_context context, cl_command_queue queue,
    int num_elements);
extern int test_async_work_group_copy_fence_import_after_export_aliased_global(
    cl_device_id deviceID, cl_context context, cl_command_queue queue,
    int num_elements);
extern int
test_async_work_group_copy_fence_import_after_export_aliased_global_and_local(
    cl_device_id deviceID, cl_context context, cl_command_queue queue,
    int num_elements);
extern int test_async_work_group_copy_fence_export_after_import_aliased_local(
    cl_device_id deviceID, cl_context context, cl_command_queue queue,
    int num_elements);
extern int test_async_work_group_copy_fence_export_after_import_aliased_global(
    cl_device_id deviceID, cl_context context, cl_command_queue queue,
    int num_elements);
extern int
test_async_work_group_copy_fence_export_after_import_aliased_global_and_local(
    cl_device_id deviceID, cl_context context, cl_command_queue queue,
    int num_elements);
extern int      test_prefetch(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);

extern int      test_host_numeric_constants(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_kernel_numeric_constants(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_kernel_limit_constants(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);

extern int    test_kernel_preprocessor_macros(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);

extern int      test_kernel_call_kernel_function(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);

extern int      test_parameter_types(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements);
extern int test_vector_creation(cl_device_id deviceID, cl_context context,
                                cl_command_queue queue, int num_elements);
extern int test_vector_swizzle(cl_device_id deviceID, cl_context context,
                               cl_command_queue queue, int num_elements);
extern int test_vec_type_hint(cl_device_id deviceID, cl_context context,
                              cl_command_queue queue, int num_elements);


extern int test_kernel_memory_alignment_local(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems );
extern int test_kernel_memory_alignment_global(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems );
extern int test_kernel_memory_alignment_constant(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems );
extern int test_kernel_memory_alignment_private(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems );

extern int test_progvar_prog_scope_misc(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems );
extern int test_progvar_prog_scope_uninit(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems );
extern int test_progvar_prog_scope_init(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems );
extern int test_progvar_func_scope(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems );

extern int test_global_work_offsets(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems );
extern int test_get_global_offset(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems );

extern int test_global_linear_id(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems );
extern int test_local_linear_id(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems );
extern int test_enqueued_local_size(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems );

extern int test_simple_read_image_pitch(cl_device_id device, cl_context cl_context_, cl_command_queue q, int num_elements);
extern int test_simple_write_image_pitch(cl_device_id device, cl_context cl_context_, cl_command_queue q, int num_elements);

#if defined( __APPLE__ )
extern int test_queue_priority(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements);
#endif

extern int test_get_linear_ids(cl_device_id device, cl_context cl_context_, cl_command_queue q, int num_elements);
extern int test_rw_image_access_qualifier(cl_device_id device_id, cl_context context, cl_command_queue commands, int num_elements);

