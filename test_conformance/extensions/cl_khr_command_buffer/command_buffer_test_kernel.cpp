//
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
#include "basic_command_buffer.h"
#include "harness/typeWrappers.h"
#include "procs.h"

#include <vector>


namespace {

////////////////////////////////////////////////////////////////////////////////
// Command-queue fill tests which handles below cases:
//
// - barrier wait list

struct BarrierWithWaitListKHR : public BasicCommandBufferTest
{

    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        std::array<cl_sync_point_khr, 2> sync_points;
        cl_int error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, nullptr, kernel, 1, nullptr, &num_elements,
            nullptr, 0, nullptr, &sync_points[0], nullptr);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clCommandBarrierWithWaitListKHR(command_buffer, nullptr, 1,
                                                sync_points.data(),
                                                &sync_points[0], nullptr);
        test_error(error, "clCommandBarrierWithWaitListKHR failed");

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clEnqueueFillBuffer(queue, in_mem, &pattern, sizeof(cl_int), 0,
                                    data_size(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueFillBuffer failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 1,
                                          &user_event, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        std::vector<cl_int> output_data(num_elements);
        error = clEnqueueReadBuffer(queue, out_mem, CL_FALSE, 0, data_size(),
                                    output_data.data(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        error = clSetUserEventStatus(user_event, CL_COMPLETE);
        test_error(error, "clSetUserEventStatus failed");

        error = clFinish(queue);
        test_error(error, "clFinish failed");

        for (size_t i = 0; i < num_elements; i++)
        {
            CHECK_VERIFICATION_ERROR(pattern, output_data[i], i);
        }

        return CL_SUCCESS;
    }

    cl_int SetUp(int elements) override
    {
        cl_int error = BasicCommandBufferTest::SetUp(elements);
        test_error(error, "BasicCommandBufferTest::SetUp failed");

        user_event = clCreateUserEvent(context, &error);
        test_error(error, "clCreateUserEvent failed");

        return CL_SUCCESS;
    }

    const cl_int pattern = 0x16;

    clEventWrapper user_event;
};
};


int test_barrier_wait_list_khr(cl_device_id device, cl_context context,
                               cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<BarrierWithWaitListKHR>(device, context, queue,
                                                  num_elements);
}
