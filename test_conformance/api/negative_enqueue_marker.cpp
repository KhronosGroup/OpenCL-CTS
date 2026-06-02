//
// Copyright (c) 2025 The Khronos Group Inc.
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

#include "testBase.h"
#include "harness/typeWrappers.h"

REGISTER_TEST(negative_enqueue_marker_with_wait_list)
{
    cl_platform_id platform = getPlatformFromDevice(device);
    cl_context_properties props[3] = {
        CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platform),
        0
    };

    cl_int err = CL_SUCCESS;
    clContextWrapper ctx =
        clCreateContext(props, 1, &device, nullptr, nullptr, &err);
    test_error(err, "clCreateContext failed");

    cl_event ret_event = nullptr;

    err = clEnqueueMarkerWithWaitList(nullptr, 0, nullptr, &ret_event);
    test_failure_error_ret(err, CL_INVALID_COMMAND_QUEUE,
                           "clEnqueueMarkerWithWaitList should return "
                           "CL_INVALID_COMMAND_QUEUE when: \"command_queue is "
                           "not a valid host command-queue\" using a nullptr",
                           TEST_FAIL);
    test_assert_error(ret_event == nullptr,
                      "if clEnqueueMarkerWithWaitList failed, no ret_event "
                      "should be created");

    clEventWrapper different_ctx_event = clCreateUserEvent(ctx, &err);
    test_error(err, "clCreateUserEvent failed");

    err =
        clEnqueueMarkerWithWaitList(queue, 1, &different_ctx_event, &ret_event);
    test_failure_error_ret(
        err, CL_INVALID_CONTEXT,
        "clEnqueueMarkerWithWaitList should return CL_INVALID_CONTEXT when: "
        "\"The context of both the command queue and the events in ret_event "
        "wait list are not the same\"",
        TEST_FAIL);
    test_assert_error(ret_event == nullptr,
                      "if clEnqueueMarkerWithWaitList failed, no ret_event "
                      "should be created");

    err = clEnqueueMarkerWithWaitList(queue, 1, nullptr, &ret_event);
    test_failure_error_ret(
        err, CL_INVALID_EVENT_WAIT_LIST,
        "clEnqueueMarkerWithWaitList should return CL_INVALID_EVENT_WAIT_LIST "
        "when: \"num_events_in_wait_list > 0 but event_wait_list is NULL\"",
        TEST_FAIL);
    test_assert_error(ret_event == nullptr,
                      "if clEnqueueMarkerWithWaitList failed, no ret_event "
                      "should be created");


    clEventWrapper event = clCreateUserEvent(context, &err);
    test_error(err, "clCreateUserEvent failed");

    err = clEnqueueMarkerWithWaitList(queue, 0, &event, &ret_event);
    test_failure_error_ret(
        err, CL_INVALID_EVENT_WAIT_LIST,
        "clEnqueueMarkerWithWaitList should return CL_INVALID_EVENT_WAIT_LIST "
        "when: \"num_events_in_wait_list is 0 but event_wait_list is not "
        "NULL\"",
        TEST_FAIL);
    test_assert_error(ret_event == nullptr,
                      "if clEnqueueMarkerWithWaitList failed, no ret_event "
                      "should be created");

    cl_event invalid_event_wait_list[] = { nullptr };
    err = clEnqueueMarkerWithWaitList(queue, 1, invalid_event_wait_list,
                                      &ret_event);
    test_failure_error_ret(
        err, CL_INVALID_EVENT_WAIT_LIST,
        "clEnqueueMarkerWithWaitList should return CL_INVALID_EVENT_WAIT_LIST "
        "when: \"event objects in event_wait_list are not valid events\"",
        TEST_FAIL);
    test_assert_error(ret_event == nullptr,
                      "if clEnqueueMarkerWithWaitList failed, no ret_event "
                      "should be created");

    return TEST_PASS;
}
