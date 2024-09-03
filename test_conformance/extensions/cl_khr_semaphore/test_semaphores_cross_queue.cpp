//
// Copyright (c) 2024 The Khronos Group Inc.
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

#include <vector>

#include "semaphore_base.h"

namespace {

int num_elems = 0;

template <bool in_order> struct SemaphoreCrossQueue : public SemaphoreTestBase
{
    SemaphoreCrossQueue(cl_device_id device, cl_context context,
                        cl_command_queue queue)
        : SemaphoreTestBase(device, context, queue)
    {}

    // Helper function that signals and waits on semaphore across two different
    // queues.
    int semaphore_cross_queue_helper(cl_device_id deviceID, cl_context context,
                                     cl_command_queue queue_1,
                                     cl_command_queue queue_2)
    {
        cl_int err = CL_SUCCESS;
        // Create semaphore
        cl_semaphore_properties_khr sema_props[] = {
            static_cast<cl_semaphore_properties_khr>(CL_SEMAPHORE_TYPE_KHR),
            static_cast<cl_semaphore_properties_khr>(
                CL_SEMAPHORE_TYPE_BINARY_KHR),
            0
        };
        semaphore =
            clCreateSemaphoreWithPropertiesKHR(context, sema_props, &err);
        test_error(err, "Could not create semaphore");

        // Signal semaphore on queue_1
        clEventWrapper signal_event;
        err = clEnqueueSignalSemaphoresKHR(queue_1, 1, semaphore, nullptr, 0,
                                           nullptr, &signal_event);
        test_error(err, "Could not signal semaphore");

        // Wait semaphore on queue_2
        clEventWrapper wait_event;
        err = clEnqueueWaitSemaphoresKHR(queue_2, 1, semaphore, nullptr, 0,
                                         nullptr, &wait_event);
        test_error(err, "Could not wait semaphore");

        // Finish queue_1 and queue_2
        err = clFinish(queue_1);
        test_error(err, "Could not finish queue");

        err = clFinish(queue_2);
        test_error(err, "Could not finish queue");

        // Ensure all events are completed
        test_assert_event_complete(signal_event);
        test_assert_event_complete(wait_event);

        return TEST_PASS;
    }

    cl_int run_in_order()
    {
        cl_int err = CL_SUCCESS;
        // Create in-order queues
        clCommandQueueWrapper queue_1 =
            clCreateCommandQueue(context, device, 0, &err);
        test_error(err, "Could not create command queue");

        clCommandQueueWrapper queue_2 =
            clCreateCommandQueue(context, device, 0, &err);
        test_error(err, "Could not create command queue");

        return semaphore_cross_queue_helper(device, context, queue_1, queue_2);
    }

    cl_int run_out_of_order()
    {
        cl_int err = CL_SUCCESS;
        // Create ooo queues
        clCommandQueueWrapper queue_1 = clCreateCommandQueue(
            context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
        test_error(err, "Could not create command queue");

        clCommandQueueWrapper queue_2 = clCreateCommandQueue(
            context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
        test_error(err, "Could not create command queue");

        return semaphore_cross_queue_helper(device, context, queue_1, queue_2);
    }

    cl_int Run() override
    {
        if (in_order)
            return run_in_order();
        else
            return run_out_of_order();
    }
};

template <bool single_queue>
struct SemaphoreOutOfOrderOps : public SemaphoreTestBase
{
    SemaphoreOutOfOrderOps(cl_device_id device, cl_context context,
                           cl_command_queue queue)
        : SemaphoreTestBase(device, context, queue)
    {}

    bool SetUp()
    {
        cl_int error = CL_SUCCESS;

        const char *kernel_str =
            R"(
          __kernel void copy(__global int* in, __global int* out) {
              size_t id = get_global_id(0);
              out[id] = in[id];
          })";

        error = create_single_kernel_helper_create_program(context, &program, 1,
                                                           &kernel_str);
        test_error(error, "Failed to create program with source");

        error = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        test_error(error, "Failed to build program");

        kernel = clCreateKernel(program, "copy", &error);
        test_error(error, "Failed to create copy kernel");

        // create producer/consumer out-of-order queues
        producer_queue = clCreateCommandQueue(
            context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &error);
        test_error(error, "Could not create command queue");

        if (single_queue)
        {
            consumer_queue = producer_queue;
        }
        else
        {
            consumer_queue = clCreateCommandQueue(
                context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
                &error);
            test_error(error, "Could not create command queue");
        }

        // Create semaphore
        cl_semaphore_properties_khr sema_props[] = {
            static_cast<cl_semaphore_properties_khr>(CL_SEMAPHORE_TYPE_KHR),
            static_cast<cl_semaphore_properties_khr>(
                CL_SEMAPHORE_TYPE_BINARY_KHR),
            0
        };
        semaphore =
            clCreateSemaphoreWithPropertiesKHR(context, sema_props, &error);
        test_error(error, "Could not create semaphore");

        // create memory resources
        in_mem = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                sizeof(cl_int) * num_elems, nullptr, &error);
        test_error(error, "clCreateBuffer failed");

        out_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                 sizeof(cl_int) * num_elems, nullptr, &error);
        test_error(error, "clCreateBuffer failed");

        error = clSetKernelArg(kernel, 0, sizeof(in_mem), &in_mem);
        test_error(error, "clSetKernelArg failed");

        error = clSetKernelArg(kernel, 1, sizeof(out_mem), &out_mem);
        test_error(error, "clSetKernelArg failed");

        return CL_SUCCESS;
    }

    cl_int Run() override
    {
        cl_int err = SetUp();
        test_error(err, "SetUp failed");

        clEventWrapper wait_events[2];
        // enqueue producer operations and signal semaphore
        const cl_int pattern_pri = 42;
        err = clEnqueueFillBuffer(producer_queue, in_mem, &pattern_pri,
                                  sizeof(cl_int), 0, sizeof(cl_int) * num_elems,
                                  0, nullptr, &wait_events[0]);
        test_error(err, "clEnqueueFillBuffer failed");

        size_t threads = (size_t)num_elems;
        err =
            clEnqueueNDRangeKernel(producer_queue, kernel, 1, nullptr, &threads,
                                   nullptr, 1, &wait_events[0], nullptr);
        test_error(err, "clEnqueueNDRangeKernel failed");

        // The semaphore cannot be signaled until the barrier is complete
        err = clEnqueueBarrierWithWaitList(producer_queue, 0, nullptr, nullptr);
        test_error(err, " clEnqueueBarrierWithWaitList ");

        if (single_queue)
        {
            // signal/wait with event dependency
            err = clEnqueueSignalSemaphoresKHR(producer_queue, 1, semaphore,
                                               nullptr, 0, nullptr,
                                               &wait_events[1]);
            test_error(err, "Could not signal semaphore");

            // consumer and producer queues in sync through wait event
            err = clEnqueueWaitSemaphoresKHR(consumer_queue, 1, semaphore,
                                             nullptr, 1, &wait_events[1],
                                             nullptr);
            test_error(err, "Could not wait semaphore");
        }
        else
        {
            err = clEnqueueSignalSemaphoresKHR(producer_queue, 1, semaphore,
                                               nullptr, 0, nullptr, nullptr);
            test_error(err, "Could not signal semaphore");

            err = clEnqueueWaitSemaphoresKHR(consumer_queue, 1, semaphore,
                                             nullptr, 0, nullptr, nullptr);
            test_error(err, "Could not wait semaphore");
        }

        err = clEnqueueBarrierWithWaitList(consumer_queue, 0, nullptr, nullptr);
        test_error(err, " clEnqueueBarrierWithWaitList ");

        const cl_int pattern_sec = 0xACDC;
        err = clEnqueueFillBuffer(consumer_queue, in_mem, &pattern_sec,
                                  sizeof(cl_int), 0, sizeof(cl_int) * num_elems,
                                  0, nullptr, &wait_events[0]);
        test_error(err, "clEnqueueFillBuffer failed");

        err = clEnqueueNDRangeKernel(consumer_queue, kernel, 1, nullptr,
                                     &threads, nullptr, 1, &wait_events[0],
                                     &wait_events[1]);
        test_error(err, "clEnqueueNDRangeKernel failed");

        std::vector<cl_int> output_buffer(num_elems, 0);
        err = clEnqueueReadBuffer(
            consumer_queue, out_mem, CL_TRUE, 0, sizeof(cl_int) * num_elems,
            output_buffer.data(), 1, &wait_events[1], nullptr);
        test_error(err, "clEnqueueReadBuffer failed");

        for (int i = 0; i < num_elems; i++)
        {
            if (pattern_sec != output_buffer[i])
            {
                log_error("Expected %d was %d at index %zu\n", pattern_sec,
                          output_buffer[i], i);
                return TEST_FAIL;
            }
        }

        return CL_SUCCESS;
    }

    clKernelWrapper kernel = nullptr;
    clProgramWrapper program = nullptr;
    clMemWrapper in_mem = nullptr, out_mem = nullptr;
    clCommandQueueWrapper producer_queue = nullptr, consumer_queue = nullptr;
};

} // anonymous namespace

// Confirm that a semaphore works across different ooo queues
int test_semaphores_cross_queues_ooo(cl_device_id deviceID, cl_context context,
                                     cl_command_queue defaultQueue,
                                     int num_elements)
{
    return MakeAndRunTest<SemaphoreCrossQueue<false>>(deviceID, context,
                                                      defaultQueue);
}

// Confirm that a semaphore works across different in-order queues
int test_semaphores_cross_queues_io(cl_device_id deviceID, cl_context context,
                                    cl_command_queue defaultQueue,
                                    int num_elements)
{
    return MakeAndRunTest<SemaphoreCrossQueue<true>>(deviceID, context,
                                                     defaultQueue);
}

// Confirm that we can synchronize signal/wait commands in single out-of-order
// queue
int test_semaphores_ooo_ops_single_queue(cl_device_id deviceID,
                                         cl_context context,
                                         cl_command_queue defaultQueue,
                                         int num_elements)
{
    num_elems = num_elements;
    return MakeAndRunTest<SemaphoreOutOfOrderOps<true>>(deviceID, context,
                                                        defaultQueue);
}

// Confirm that we can synchronize signal/wait commands across two out-of-order
// queues
int test_semaphores_ooo_ops_cross_queue(cl_device_id deviceID,
                                        cl_context context,
                                        cl_command_queue defaultQueue,
                                        int num_elements)
{
    num_elems = num_elements;
    return MakeAndRunTest<SemaphoreOutOfOrderOps<false>>(deviceID, context,
                                                         defaultQueue);
}
