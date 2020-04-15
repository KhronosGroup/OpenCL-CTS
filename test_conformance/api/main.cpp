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
#include "harness/compat.h"
#include <stdio.h>
#include <stdlib.h>

#include <string.h>
#include "procs.h"
#include "harness/testHarness.h"

#if !defined(_WIN32)
#include <unistd.h>
#endif

// FIXME: To use certain functions in harness/imageHelpers.h
// (for example, generate_random_image_data()), the tests are required to declare
// the following variables (<rdar://problem/11111245>):

test_definition test_list[] = {
    ADD_TEST( get_platform_info ),
    ADD_TEST_VERSION( get_sampler_info, Version(2, 0) ),
    ADD_TEST( get_sampler_info_compatibility ),
    ADD_TEST_VERSION( get_command_queue_info, Version(2, 0) ),
    ADD_TEST( get_command_queue_info_compatibility ),
    ADD_TEST( get_context_info ),
    ADD_TEST( get_device_info ),
    ADD_TEST( enqueue_task ),
    ADD_TEST( binary_get ),
    ADD_TEST( binary_create ),
    ADD_TEST( kernel_required_group_size ),

    ADD_TEST( release_kernel_order ),
    ADD_TEST( release_during_execute ),

    ADD_TEST( load_single_kernel ),
    ADD_TEST( load_two_kernels ),
    ADD_TEST( load_two_kernels_in_one ),
    ADD_TEST( load_two_kernels_manually ),
    ADD_TEST( get_program_info_kernel_names ),
    ADD_TEST( get_kernel_arg_info ),
    ADD_TEST( get_kernel_arg_info_compatibility ),
    ADD_TEST( create_kernels_in_program ),
    ADD_TEST( get_kernel_info ),
    ADD_TEST( execute_kernel_local_sizes ),
    ADD_TEST( set_kernel_arg_by_index ),
    ADD_TEST( set_kernel_arg_constant ),
    ADD_TEST( set_kernel_arg_struct_array ),
    ADD_TEST( kernel_global_constant ),

    ADD_TEST( min_max_thread_dimensions ),
    ADD_TEST( min_max_work_items_sizes ),
    ADD_TEST( min_max_work_group_size ),
    ADD_TEST( min_max_read_image_args ),
    ADD_TEST( min_max_write_image_args ),
    ADD_TEST( min_max_mem_alloc_size ),
    ADD_TEST( min_max_image_2d_width ),
    ADD_TEST( min_max_image_2d_height ),
    ADD_TEST( min_max_image_3d_width ),
    ADD_TEST( min_max_image_3d_height ),
    ADD_TEST( min_max_image_3d_depth ),
    ADD_TEST( min_max_image_array_size ),
    ADD_TEST( min_max_image_buffer_size ),
    ADD_TEST( min_max_parameter_size ),
    ADD_TEST( min_max_samplers ),
    ADD_TEST( min_max_constant_buffer_size ),
    ADD_TEST( min_max_constant_args ),
    ADD_TEST( min_max_compute_units ),
    ADD_TEST( min_max_address_bits ),
    ADD_TEST( min_max_single_fp_config ),
    ADD_TEST( min_max_double_fp_config ),
    ADD_TEST( min_max_local_mem_size ),
    ADD_TEST( min_max_kernel_preferred_work_group_size_multiple ),
    ADD_TEST( min_max_execution_capabilities ),
    ADD_TEST( min_max_queue_properties ),
    ADD_TEST( min_max_device_version ),
    ADD_TEST( min_max_language_version ),

    ADD_TEST( kernel_arg_changes ),
    ADD_TEST( kernel_arg_multi_setup_random ),

    ADD_TEST( native_kernel ),

    ADD_TEST( create_context_from_type ),

    ADD_TEST( platform_extensions ),
    ADD_TEST( get_platform_ids ),
    ADD_TEST( bool_type ),

    ADD_TEST( repeated_setup_cleanup ),

    ADD_TEST( retain_queue_single ),
    ADD_TEST( retain_queue_multiple ),
    ADD_TEST( retain_mem_object_single ),
    ADD_TEST( retain_mem_object_multiple ),
    ADD_TEST( retain_mem_object_set_kernel_arg ),
    ADD_TEST( min_data_type_align_size_alignment ),

    ADD_TEST( mem_object_destructor_callback ),
    ADD_TEST( null_buffer_arg ),
    ADD_TEST( get_buffer_info ),
    ADD_TEST( get_image2d_info ),
    ADD_TEST( get_image3d_info ),
    ADD_TEST( get_image1d_info ),
    ADD_TEST( get_image1d_array_info ),
    ADD_TEST( get_image2d_array_info ),
    ADD_TEST( queue_hint ),
    ADD_TEST( queue_properties ),
    ADD_TEST_VERSION( sub_group_dispatch, Version(2, 1) ),
    ADD_TEST_VERSION( clone_kernel, Version(2, 1) ),
    ADD_TEST_VERSION( zero_sized_enqueue, Version(2, 1) ),
};

const int test_num = ARRAY_SIZE( test_list );

int main(int argc, const char *argv[])
{
    return runTestHarness( argc, argv, test_num, test_list, false, false, 0 );
}

