// Copyright (c) 2022 The Khronos Group Inc.
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
#include "procs.h"
#include "harness/testHarness.h"

test_definition test_list[] = {
    ADD_TEST(mutable_command_info_device_query),
    ADD_TEST(mutable_command_info_buffer),
    ADD_TEST(mutable_command_properties_array),
    ADD_TEST(mutable_command_kernel),
    ADD_TEST(mutable_command_dimensions),
    ADD_TEST(mutable_command_info_type),
    ADD_TEST(mutable_command_info_queue),
    ADD_TEST(mutable_command_info_global_work_offset),
    ADD_TEST(mutable_command_info_local_work_size),
    ADD_TEST(mutable_command_info_global_work_size),
    ADD_TEST(mutable_command_full_dispatch),
    ADD_TEST(mutable_command_overwrite_update),
    ADD_TEST(mutable_command_multiple_dispatches),
    ADD_TEST(mutable_command_iterative_arg_update),
    ADD_TEST(mutable_dispatch_image_1d_arguments),
    ADD_TEST(mutable_dispatch_image_2d_arguments),
    ADD_TEST(mutable_dispatch_out_of_order),
    ADD_TEST(mutable_dispatch_simultaneous_out_of_order),
    ADD_TEST(mutable_dispatch_simultaneous_in_order),
    ADD_TEST(mutable_dispatch_simultaneous_cross_queue),
    ADD_TEST(mutable_dispatch_global_size),
    ADD_TEST(mutable_dispatch_local_size),
    ADD_TEST(mutable_dispatch_global_offset),
    ADD_TEST(mutable_dispatch_svm_arguments),
    ADD_TEST(mutable_dispatch_local_arguments),
    ADD_TEST(mutable_dispatch_global_arguments),
    ADD_TEST(mutable_dispatch_pod_arguments),
    ADD_TEST(mutable_dispatch_null_arguments),
    ADD_TEST(command_buffer_with_no_additional_work_groups),
    ADD_TEST(ndrange_with_no_additional_work_groups),
    ADD_TEST(ndrange_command_buffer_with_no_additional_work_groups),
};

int main(int argc, const char *argv[])
{
    // A device may report the required properties of a queue that
    // is compatible with command-buffers via the query
    // CL_DEVICE_COMMAND_BUFFER_REQUIRED_QUEUE_PROPERTIES_KHR. We account
    // for this in the tests themselves, rather than here, where we have a
    // device to query.
    const cl_command_queue_properties queue_properties = 0;
    return runTestHarnessWithCheck(argc, argv, ARRAY_SIZE(test_list), test_list,
                                   false, queue_properties, nullptr);
    return 0;
}
