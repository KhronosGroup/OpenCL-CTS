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
#include "harness/typeWrappers.h"
#include "harness/clImageHelper.h"
#include "harness/imageHelpers.h"
extern float    calculate_ulperror(float a, float b);

extern int        test_load_single_kernel(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_load_two_kernels(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_load_two_kernels_in_one(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_load_two_kernels_manually(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_get_program_info_kernel_names( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_create_kernels_in_program(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_enqueue_task(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_repeated_setup_cleanup(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);

extern int        test_bool_type(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_platform_extensions(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_get_platform_info(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_get_sampler_info(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_get_sampler_info_compatibility(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_get_command_queue_info(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_get_command_queue_info_compatibility(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_get_context_info(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_get_device_info(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_kernel_required_group_size(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);

extern int        test_binary_get(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_binary_create(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);

extern int        test_release_kernel_order(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_release_during_execute(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);

extern int        test_get_kernel_info(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_execute_kernel_local_sizes(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_set_kernel_arg_by_index(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_set_kernel_arg_struct(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_set_kernel_arg_constant(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_set_kernel_arg_struct_array(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_kernel_global_constant(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);

extern int        test_min_max_thread_dimensions(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_min_max_work_items_sizes(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_min_max_work_group_size(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_min_max_read_image_args(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_min_max_write_image_args(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_min_max_mem_alloc_size(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_min_max_image_2d_width(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_min_max_image_2d_height(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_min_max_image_3d_width(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_min_max_image_3d_height(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_min_max_image_3d_depth(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_min_max_image_array_size(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_min_max_image_buffer_size(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_min_max_parameter_size(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_min_max_samplers(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_min_max_constant_buffer_size(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_min_max_constant_args(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_min_max_compute_units(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_min_max_address_bits(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_min_max_single_fp_config(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_min_max_double_fp_config(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_min_max_local_mem_size(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_min_max_kernel_preferred_work_group_size_multiple(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_min_max_execution_capabilities(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_min_max_queue_properties(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_min_max_device_version(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_min_max_language_version(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);

extern int        test_native_kernel(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems );

extern int      test_create_context_from_type(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);

extern int      test_get_platform_ids(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);

extern int        test_kernel_arg_changes(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_kernel_arg_multi_setup_random(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);

extern int        test_retain_queue_single(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_retain_queue_multiple(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_retain_mem_object_single(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_retain_mem_object_multiple(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_retain_mem_object_set_kernel_arg(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_min_data_type_align_size_alignment(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems );

extern int        test_mem_object_destructor_callback(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);

extern int      test_null_buffer_arg( cl_device_id device_id, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_get_buffer_info( cl_device_id deviceID, cl_context context, cl_command_queue ignoreQueue, int num_elements );
extern int      test_get_image2d_info( cl_device_id deviceID, cl_context context, cl_command_queue ignoreQueue, int num_elements );
extern int      test_get_image3d_info( cl_device_id deviceID, cl_context context, cl_command_queue ignoreQueue, int num_elements );
extern int      test_get_image1d_info( cl_device_id deviceID, cl_context context, cl_command_queue ignoreQueue, int num_elements );
extern int      test_get_image1d_array_info( cl_device_id deviceID, cl_context context, cl_command_queue ignoreQueue, int num_elements );
extern int      test_get_image2d_array_info( cl_device_id deviceID, cl_context context, cl_command_queue ignoreQueue, int num_elements );
extern int      test_get_kernel_arg_info( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_get_kernel_arg_info_compatibility( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int      test_queue_hint(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_sub_group_dispatch(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_clone_kernel(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_zero_sized_enqueue(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int      test_queue_properties( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
