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
    ADD_TEST(single_ndrange),
    ADD_TEST(interleaved_enqueue),
    ADD_TEST(mixed_commands),
    ADD_TEST(explicit_flush),
    ADD_TEST(out_of_order),
    ADD_TEST(simultaneous_out_of_order),
    ADD_TEST(info_queues),
    ADD_TEST(info_ref_count),
    ADD_TEST(info_state),
    ADD_TEST(info_prop_array),
    ADD_TEST(info_context),
    ADD_TEST(basic_profiling),
    ADD_TEST(simultaneous_profiling),
    ADD_TEST(regular_wait_for_command_buffer),
    ADD_TEST(command_buffer_wait_for_command_buffer),
    ADD_TEST(command_buffer_wait_for_sec_command_buffer),
    ADD_TEST(return_event_callback),
    ADD_TEST(clwaitforevents_single),
    ADD_TEST(clwaitforevents),
    ADD_TEST(command_buffer_wait_for_regular),
    ADD_TEST(wait_for_sec_queue_event),
    ADD_TEST(user_event_wait),
    ADD_TEST(user_events_wait),
    ADD_TEST(user_event_callback),
    ADD_TEST(queue_substitution),
    ADD_TEST(properties_queue_substitution),
    ADD_TEST(simultaneous_queue_substitution),
    ADD_TEST(fill_image),
    ADD_TEST(fill_buffer),
    ADD_TEST(fill_svm_buffer),
    ADD_TEST(copy_image),
    ADD_TEST(copy_buffer),
    ADD_TEST(copy_svm_buffer),
    ADD_TEST(copy_buffer_to_image),
    ADD_TEST(copy_image_to_buffer),
    ADD_TEST(copy_buffer_rect),
    ADD_TEST(barrier_wait_list),
    ADD_TEST(basic_printf),
    ADD_TEST(simultaneous_printf),
    ADD_TEST(basic_set_kernel_arg),
    ADD_TEST(pending_set_kernel_arg),
    ADD_TEST(event_info_command_type),
    ADD_TEST(event_info_command_queue),
    ADD_TEST(event_info_execution_status),
    ADD_TEST(event_info_context),
    ADD_TEST(event_info_reference_count),
    ADD_TEST(finalize_invalid),
    ADD_TEST(finalize_empty),
    // Command-buffer negative tests
    ADD_TEST(negative_retain_command_buffer_invalid_command_buffer),
    ADD_TEST(negative_release_command_buffer_invalid_command_buffer),
    ADD_TEST(negative_finalize_command_buffer_invalid_command_buffer),
    ADD_TEST(negative_finalize_command_buffer_not_recording_state),
    ADD_TEST(negative_command_buffer_command_fill_buffer_queue_not_null),
    ADD_TEST(negative_command_buffer_command_fill_buffer_context_not_same),
    ADD_TEST(
        negative_command_buffer_command_fill_buffer_sync_points_null_or_num_zero),
    ADD_TEST(
        negative_command_buffer_command_fill_buffer_invalid_command_buffer),
    ADD_TEST(
        negative_command_buffer_command_fill_buffer_finalized_command_buffer),
    ADD_TEST(
        negative_command_buffer_command_fill_buffer_mutable_handle_not_null),
    ADD_TEST(negative_command_buffer_command_fill_image_queue_not_null),
    ADD_TEST(negative_command_buffer_command_fill_image_context_not_same),
    ADD_TEST(
        negative_command_buffer_command_fill_image_sync_points_null_or_num_zero),
    ADD_TEST(negative_command_buffer_command_fill_image_invalid_command_buffer),
    ADD_TEST(
        negative_command_buffer_command_fill_image_finalized_command_buffer),
    ADD_TEST(
        negative_command_buffer_command_fill_image_mutable_handle_not_null),
    ADD_TEST(negative_create_command_buffer_num_queues),
    ADD_TEST(negative_create_command_buffer_null_queues),
    ADD_TEST(negative_create_command_buffer_repeated_properties),
    ADD_TEST(negative_create_command_buffer_not_supported_properties),
    ADD_TEST(negative_create_command_buffer_queue_without_min_properties),
    ADD_TEST(
        negative_create_command_buffer_device_does_not_support_out_of_order_queue),
    ADD_TEST(negative_command_ndrange_queue_not_null),
    ADD_TEST(negative_command_ndrange_kernel_with_different_context),
    ADD_TEST(negative_command_ndrange_kernel_sync_points_null_or_num_zero),
    ADD_TEST(negative_command_ndrange_kernel_invalid_command_buffer),
    ADD_TEST(negative_command_ndrange_kernel_invalid_properties),
    ADD_TEST(negative_command_ndrange_kernel_command_buffer_finalized),
    ADD_TEST(negative_command_ndrange_kernel_mutable_handle_not_null),
    ADD_TEST(negative_command_ndrange_kernel_not_support_printf),
    ADD_TEST(negative_command_ndrange_kernel_with_enqueue_call),
    ADD_TEST(negative_command_buffer_command_copy_buffer_queue_not_null),
    ADD_TEST(negative_command_buffer_command_copy_buffer_different_contexts),
    ADD_TEST(
        negative_command_buffer_command_copy_buffer_sync_points_null_or_num_zero),
    ADD_TEST(
        negative_command_buffer_command_copy_buffer_invalid_command_buffer),
    ADD_TEST(
        negative_command_buffer_command_copy_buffer_finalized_command_buffer),
    ADD_TEST(
        negative_command_buffer_command_copy_buffer_mutable_handle_not_null),
    ADD_TEST(negative_command_buffer_command_copy_image_queue_not_null),
    ADD_TEST(negative_command_buffer_command_copy_image_different_contexts),
    ADD_TEST(
        negative_command_buffer_command_copy_image_sync_points_null_or_num_zero),
    ADD_TEST(negative_command_buffer_command_copy_image_invalid_command_buffer),
    ADD_TEST(
        negative_command_buffer_command_copy_image_finalized_command_buffer),
    ADD_TEST(
        negative_command_buffer_command_copy_image_mutable_handle_not_null),
    ADD_TEST(negative_get_command_buffer_info_invalid_command_buffer),
    ADD_TEST(negative_get_command_buffer_info_not_supported_param_name),
    ADD_TEST(negative_get_command_buffer_info_queues),
    ADD_TEST(negative_get_command_buffer_info_ref_count),
    ADD_TEST(negative_get_command_buffer_info_state),
    ADD_TEST(negative_get_command_buffer_info_prop_array),
    ADD_TEST(negative_get_command_buffer_info_context),
    ADD_TEST(negative_command_buffer_command_svm_queue_not_null),
    ADD_TEST(negative_command_buffer_command_svm_sync_points_null_or_num_zero),
    ADD_TEST(negative_command_buffer_command_svm_invalid_command_buffer),
    ADD_TEST(negative_command_buffer_command_svm_finalized_command_buffer),
    ADD_TEST(negative_command_buffer_command_svm_mutable_handle_not_null),
    ADD_TEST(negative_command_buffer_copy_image_queue_not_null),
    ADD_TEST(negative_command_buffer_copy_image_context_not_same),
    ADD_TEST(negative_command_buffer_copy_image_sync_points_null_or_num_zero),
    ADD_TEST(negative_command_buffer_copy_image_invalid_command_buffer),
    ADD_TEST(negative_command_buffer_copy_image_finalized_command_buffer),
    ADD_TEST(negative_command_buffer_copy_image_mutable_handle_not_null),
    ADD_TEST(negative_command_buffer_barrier_not_null_queue),
    ADD_TEST(negative_command_buffer_barrier_invalid_command_buffer),
    ADD_TEST(negative_command_buffer_barrier_buffer_finalized),
    ADD_TEST(negative_command_buffer_barrier_mutable_handle_not_null),
    ADD_TEST(negative_command_buffer_barrier_sync_points_null_or_num_zero),
    ADD_TEST(negative_enqueue_command_buffer_invalid_command_buffer),
    ADD_TEST(negative_enqueue_command_buffer_not_finalized),
    ADD_TEST(
        negative_enqueue_command_buffer_without_simultaneous_no_pending_state),
    ADD_TEST(negative_enqueue_command_buffer_null_queues_num_queues),
    ADD_TEST(
        negative_enqueue_command_buffer_num_queues_not_zero_different_while_buffer_creation),
    ADD_TEST(negative_enqueue_command_buffer_not_valid_queue_in_queues),
    ADD_TEST(negative_enqueue_queue_not_compatible),
    ADD_TEST(negative_enqueue_queue_with_different_context),
    ADD_TEST(negative_enqueue_command_buffer_different_context_than_event),
    ADD_TEST(negative_enqueue_event_wait_list_null_or_events_null),
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
}
