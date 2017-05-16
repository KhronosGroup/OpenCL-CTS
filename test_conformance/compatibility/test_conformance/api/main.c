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
#include "../../test_common/harness/compat.h"

#include <stdio.h>
#include <string.h>
#include "procs.h"
#include "../../test_common/harness/testHarness.h"

#if !defined(_WIN32)
#include <unistd.h>
#endif

// FIXME: To use certain functions in ../../test_common/harness/imageHelpers.h
// (for example, generate_random_image_data()), the tests are required to declare
// the following variables:
cl_device_type gDeviceType = CL_DEVICE_TYPE_DEFAULT;
bool gTestRounding = false;

basefn    basefn_list[] = {
    test_get_platform_info,
    test_get_sampler_info,
    test_get_command_queue_info,
    test_get_context_info,
    test_get_device_info,
    test_enqueue_task,
    test_binary_get,
    test_program_binary_create,
    test_kernel_required_group_size,

    test_release_kernel_order,
    test_release_during_execute,

    test_load_single_kernel,
    test_load_two_kernels,
    test_load_two_kernels_in_one,
    test_load_two_kernels_manually,
    test_get_program_info_kernel_names,
    test_get_kernel_arg_info,
    test_create_kernels_in_program,
    test_get_kernel_info,
    test_execute_kernel_local_sizes,
    test_set_kernel_arg_by_index,
    test_set_kernel_arg_constant,
    test_set_kernel_arg_struct_array,
    test_kernel_global_constant,

    test_min_max_thread_dimensions,
    test_min_max_work_items_sizes,
    test_min_max_work_group_size,
    test_min_max_read_image_args,
    test_min_max_write_image_args,
    test_min_max_mem_alloc_size,
    test_min_max_image_2d_width,
    test_min_max_image_2d_height,
    test_min_max_image_3d_width,
    test_min_max_image_3d_height,
    test_min_max_image_3d_depth,
    test_min_max_image_array_size,
    test_min_max_image_buffer_size,
    test_min_max_parameter_size,
    test_min_max_samplers,
    test_min_max_constant_buffer_size,
    test_min_max_constant_args,
    test_min_max_compute_units,
    test_min_max_address_bits,
    test_min_max_single_fp_config,
    test_min_max_double_fp_config,
    test_min_max_local_mem_size,
    test_min_max_kernel_preferred_work_group_size_multiple,
    test_min_max_execution_capabilities,
    test_min_max_queue_properties,
    test_min_max_device_version,
    test_min_max_language_version,

    test_kernel_arg_changes,
    test_kernel_arg_multi_setup_random,

    test_native_kernel,

    test_create_context_from_type,

    test_platform_extensions,
    test_get_platform_ids,
    test_for_bool_type,

    test_repeated_setup_cleanup,

    test_retain_queue_single,
    test_retain_queue_multiple,
    test_retain_mem_object_single,
    test_retain_mem_object_multiple,
    test_min_data_type_align_size_alignment,

    test_mem_object_destructor_callback,
    test_null_buffer_arg,
    test_get_buffer_info,
    test_get_image2d_info,
    test_get_image3d_info,
    test_get_image1d_info,
    test_get_image1d_array_info,
    test_get_image2d_array_info,
};


const char    *basefn_names[] = {
    "get_platform_info",
    "get_sampler_info",
    "get_command_queue_info",
    "get_context_info",
    "get_device_info",
    "enqueue_task",
    "binary_get",
    "binary_create",
    "kernel_required_group_size",

    "release_kernel_order",
    "release_during_execute",

    "load_single_kernel",
    "load_two_kernels",
    "load_two_kernels_in_one",
    "load_two_kernels_manually",
    "get_program_info_kernel_names",
    "get_kernel_arg_info",
    "create_kernels_in_program",
    "get_kernel_info",
    "execute_kernel_local_sizes",
    "set_kernel_arg_by_index",
    "set_kernel_arg_constant",
    "set_kernel_arg_struct_array",
    "kernel_global_constant",

    "min_max_thread_dimensions",
    "min_max_work_items_sizes",
    "min_max_work_group_size",
    "min_max_read_image_args",
    "min_max_write_image_args",
    "min_max_mem_alloc_size",
    "min_max_image_2d_width",
    "min_max_image_2d_height",
    "min_max_image_3d_width",
    "min_max_image_3d_height",
    "min_max_image_3d_depth",
    "min_max_image_array_size",
    "min_max_image_buffer_size",
    "min_max_parameter_size",
    "min_max_samplers",
    "min_max_constant_buffer_size",
    "min_max_constant_args",
    "min_max_compute_units",
    "min_max_address_bits",
    "min_max_single_fp_config",
    "min_max_double_fp_config",
    "min_max_local_mem_size",
    "min_max_kernel_preferred_work_group_size_multiple",
    "min_max_execution_capabilities",
    "min_max_queue_properties",
    "min_max_device_version",
    "min_max_language_version",

    "kernel_arg_changes",
    "kernel_arg_multi_setup_random",

    "native_kernel",

    "create_context_from_type",
    "platform_extensions",

    "get_platform_ids",
    "bool_type",

    "repeated_setup_cleanup",

    "retain_queue_single",
    "retain_queue_multiple",
    "retain_mem_object_single",
    "retain_mem_object_multiple",

    "min_data_type_align_size_alignment",

    "mem_object_destructor_callback",
    "null_buffer_arg",
    "get_buffer_info",
    "get_image2d_info",
    "get_image3d_info",
    "get_image1d_info",
    "get_image1d_array_info",
    "get_image2d_array_info",
};

ct_assert((sizeof(basefn_names) / sizeof(basefn_names[0])) == (sizeof(basefn_list) / sizeof(basefn_list[0])));

int    num_fns = sizeof(basefn_names) / sizeof(char *);

int main(int argc, const char *argv[])
{
    return runTestHarness( argc, argv, num_fns, basefn_list, basefn_names, false, false, 0 );
}


