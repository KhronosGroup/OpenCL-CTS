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
#ifndef TEST_CONFORMANCE_CLCPP_DEVICE_QUEUE_TEST_ENQUEUE_HPP
#define TEST_CONFORMANCE_CLCPP_DEVICE_QUEUE_TEST_ENQUEUE_HPP

#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

// Common for all OpenCL C++ tests
#include "../common.hpp"


namespace test_enqueue {

struct test_options
{
    int test;
};

struct output_type
{
    cl_int enqueue_kernel1_success;
    cl_int enqueue_kernel2_success;
    cl_int enqueue_kernel3_success;
    cl_int enqueue_marker_success;
    cl_int event1_is_valid;
    cl_int event2_is_valid;
    cl_int event3_is_valid;
    cl_int user_event1_is_valid;
    cl_int user_event2_is_valid;
    cl_int values[10000];
};

const std::string source_common = R"(
struct output_type
{
    int enqueue_kernel1_success;
    int enqueue_kernel2_success;
    int enqueue_kernel3_success;
    int enqueue_marker_success;
    int event1_is_valid;
    int event2_is_valid;
    int event3_is_valid;
    int user_event1_is_valid;
    int user_event2_is_valid;
    int values[10000];
};
)";

// -----------------------------------------------------------------------------------
// ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
// -----------------------------------------------------------------------------------
#if defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
std::string generate_source(test_options options)
{
    std::stringstream s;
    s << source_common;
    if (options.test == 0)
    {
        s << R"(
    kernel void test(queue_t queue, global struct output_type *output)
    {
        const ulong gid = get_global_id(0);

        if (gid != 0)
            return;

        output->enqueue_kernel2_success = 1;
        output->enqueue_kernel3_success = 1;
        output->enqueue_marker_success = 1;
        output->event2_is_valid = 1;
        output->event3_is_valid = 1;
        output->user_event1_is_valid = 1;
        output->user_event2_is_valid = 1;

        queue_t default_queue = get_default_queue();

        ndrange_t ndrange1 = ndrange_1D(get_global_size(0));
        clk_event_t event1;
        int status1 = enqueue_kernel(default_queue, CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange1, 0, NULL, &event1,
        ^{
            const ulong gid = get_global_id(0);
            output->values[gid] = 1;
        });
        output->enqueue_kernel1_success = status1 == CLK_SUCCESS;
        output->event1_is_valid = is_valid_event(event1);

        release_event(event1);
    }
    )";
    }
    else if (options.test == 1)
    {
        s << R"(
    kernel void test(queue_t queue, global struct output_type *output)
    {
        const ulong gid = get_global_id(0);

        if (gid != 0)
            return;

        output->enqueue_kernel3_success = 1;
        output->enqueue_marker_success = 1;
        output->event3_is_valid = 1;
        output->user_event1_is_valid = 1;
        output->user_event2_is_valid = 1;

        queue_t default_queue = get_default_queue();

        ndrange_t ndrange1 = ndrange_1D(get_global_size(0) / 2);
        clk_event_t event1;
        int status1 = enqueue_kernel(default_queue, CLK_ENQUEUE_FLAGS_WAIT_WORK_GROUP, ndrange1, 0, NULL, &event1,
        ^{
            const ulong gid = get_global_id(0);
            output->values[gid * 2] = 1;
        });
        output->enqueue_kernel1_success = status1 == CLK_SUCCESS;
        output->event1_is_valid = is_valid_event(event1);

        ndrange_t ndrange2 = ndrange_1D(1, get_global_size(0) / 2, 1);
        clk_event_t event2;
        int status2 = enqueue_kernel(queue, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange2, 1, &event1, &event2,
        ^{
            const ulong gid = get_global_id(0);
            output->values[(gid - 1) * 2 + 1] = 1;
        });
        output->enqueue_kernel2_success = status2 == CLK_SUCCESS;
        output->event2_is_valid = is_valid_event(event2);

        release_event(event1);
        release_event(event2);
    }
    )";
    }
    else if (options.test == 2)
    {
        s << R"(
    kernel void test(queue_t queue, global struct output_type *output)
    {
        const ulong gid = get_global_id(0);

        if (gid != 0)
            return;

        output->enqueue_marker_success = 1;
        output->event3_is_valid = 1;
        output->enqueue_kernel3_success = 1;

        queue_t default_queue = get_default_queue();

        clk_event_t user_event1 = create_user_event();
        retain_event(user_event1);
        output->user_event1_is_valid = is_valid_event(user_event1);

        ndrange_t ndrange1 = ndrange_1D(get_global_size(0) / 2);
        clk_event_t event1;
        int status1 = enqueue_kernel(queue, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange1, 1, &user_event1, &event1,
        ^{
            const ulong gid = get_global_id(0);
            output->values[gid * 2] = 1;
        });
        output->enqueue_kernel1_success = status1 == CLK_SUCCESS;
        output->event1_is_valid = is_valid_event(event1);
        release_event(user_event1);

        clk_event_t user_event2 = create_user_event();
        output->user_event2_is_valid = is_valid_event(user_event2);

        clk_event_t events[2];
        events[0] = user_event2;
        events[1] = user_event1;

        ndrange_t ndrange2 = ndrange_1D(1, get_global_size(0) / 2, get_local_size(0));
        clk_event_t event2;
        int status2 = enqueue_kernel(default_queue, CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange2, 2, events, &event2,
        ^(local void *p0, local void *p1, local void *p2) {
            const ulong gid = get_global_id(0);
            const ulong lid = get_local_id(0);
            local int2 *l0 = (local int2 *)p0;
            local int *l1 = (local int *)p1;
            local int *l2 = (local int *)p2;
            l1[get_local_size(0) - lid - 1] = gid > 0 ? 1 : 0;
            work_group_barrier(CLK_LOCAL_MEM_FENCE);
            if (lid < 5) l0[lid] = (int2)(3, 4);
            if (lid < 3) l2[lid] = 5;
            work_group_barrier(CLK_LOCAL_MEM_FENCE);
            output->values[(gid - 1) * 2 + 1] = min(l1[lid], min(l0[0].x, l2[0]));
        }, sizeof(int2) * 5, sizeof(int) * get_local_size(0), sizeof(int) * 3);
        output->enqueue_kernel2_success = status2 == CLK_SUCCESS;
        output->event2_is_valid = is_valid_event(event2);

        set_user_event_status(user_event1, CL_COMPLETE);
        set_user_event_status(user_event2, CL_COMPLETE);

        release_event(user_event1);
        release_event(user_event2);
        release_event(event1);
        release_event(event2);
    }
    )";
    }
    else if (options.test == 3)
    {
        s << R"(
    kernel void test(queue_t queue, global struct output_type *output)
    {
        const ulong gid = get_global_id(0);

        if (gid != 0)
            return;

        output->user_event2_is_valid = 1;

        queue_t default_queue = get_default_queue();

        ndrange_t ndrange1 = ndrange_1D(get_global_size(0) / 2);
        clk_event_t event1;
        int status1 = enqueue_kernel(default_queue, CLK_ENQUEUE_FLAGS_WAIT_WORK_GROUP, ndrange1, 0, NULL, &event1,
        ^{
            const ulong gid = get_global_id(0);
            output->values[gid * 2] = 20;
        });
        output->enqueue_kernel1_success = status1 == CLK_SUCCESS;
        output->event1_is_valid = is_valid_event(event1);

        ndrange_t ndrange2 = ndrange_1D(1, get_global_size(0) / 2, 1);
        clk_event_t event2;
        int status2 = enqueue_kernel(queue, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange2, 0, NULL, &event2,
        ^{
            const ulong gid = get_global_id(0);
            output->values[(gid - 1) * 2 + 1] = 20;
        });
        output->enqueue_kernel2_success = status2 == CLK_SUCCESS;
        output->event2_is_valid = is_valid_event(event2);

        clk_event_t user_event1 = create_user_event();
        output->user_event1_is_valid = is_valid_event(user_event1);

        clk_event_t events[3];
        events[0] = event2;
        events[1] = user_event1;
        events[2] = event1;

        clk_event_t event3;
        int status3 = enqueue_marker(queue, 3, events, &event3);
        output->enqueue_marker_success = status3 == CLK_SUCCESS;
        output->event3_is_valid = is_valid_event(event3);

        int status4 = enqueue_kernel(default_queue, CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(get_global_size(0)), 1, &event3, NULL,
        ^{
            const ulong gid = get_global_id(0);
            output->values[gid] /= 20;
        });
        output->enqueue_kernel3_success = status4 == CLK_SUCCESS;

        set_user_event_status(user_event1, CL_COMPLETE);

        release_event(user_event1);
        release_event(event1);
        release_event(event2);
        release_event(event3);
    }
    )";
    }

    return s.str();
}
#else
std::string generate_source(test_options options)
{
    std::stringstream s;
    s << R"(
    #include <opencl_memory>
    #include <opencl_common>
    #include <opencl_work_item>
    #include <opencl_synchronization>
    #include <opencl_device_queue>
    using namespace cl;
    )";

    s << source_common;
    if (options.test == 0)
    {
        s << R"(
    kernel void test(device_queue queue, global<output_type> *output)
    {
        const ulong gid = get_global_id(0);

        if (gid != 0)
            return;

        output->enqueue_kernel2_success = 1;
        output->enqueue_kernel3_success = 1;
        output->enqueue_marker_success = 1;
        output->event2_is_valid = 1;
        output->event3_is_valid = 1;
        output->user_event1_is_valid = 1;
        output->user_event2_is_valid = 1;

        device_queue default_queue = get_default_device_queue();

        ndrange ndrange1(get_global_size(0));
        event event1;
        enqueue_status status1 = default_queue.enqueue_kernel(enqueue_policy::no_wait, 0, nullptr, &event1, ndrange1,
        [](global<output_type> *output) {
            const ulong gid = get_global_id(0);
            output->values[gid] = 1;
        }, output);
        output->enqueue_kernel1_success = status1 == enqueue_status::success;
        output->event1_is_valid = event1.is_valid();

        event1.release();
    }
    )";
    }
    else if (options.test == 1)
    {
        s << R"(
    kernel void test(device_queue queue, global<output_type> *output)
    {
        const ulong gid = get_global_id(0);

        if (gid != 0)
            return;

        output->enqueue_kernel3_success = 1;
        output->enqueue_marker_success = 1;
        output->event3_is_valid = 1;
        output->user_event1_is_valid = 1;
        output->user_event2_is_valid = 1;

        device_queue default_queue = get_default_device_queue();

        ndrange ndrange1(get_global_size(0) / 2);
        event event1;
        enqueue_status status1 = default_queue.enqueue_kernel(enqueue_policy::wait_work_group, 0, nullptr, &event1, ndrange1,
        [](global<output_type> *output) {
            const ulong gid = get_global_id(0);
            output->values[gid * 2] = 1;
        }, output);
        output->enqueue_kernel1_success = status1 == enqueue_status::success;
        output->event1_is_valid = event1.is_valid();

        ndrange ndrange2(1, get_global_size(0) / 2, 1);
        event event2;
        enqueue_status status2 = queue.enqueue_kernel(enqueue_policy::wait_kernel, 1, &event1, &event2, ndrange2,
        [](global<output_type> *output) {
            const ulong gid = get_global_id(0);
            output->values[(gid - 1) * 2 + 1] = 1;
        }, output);
        output->enqueue_kernel2_success = status2 == enqueue_status::success;
        output->event2_is_valid = event2.is_valid();

        event1.release();
        event2.release();
    }
    )";
    }
    else if (options.test == 2)
    {
        s << R"(
    kernel void test(device_queue queue, global<output_type> *output)
    {
        const ulong gid = get_global_id(0);

        if (gid != 0)
            return;

        output->enqueue_marker_success = 1;
        output->event3_is_valid = 1;
        output->enqueue_kernel3_success = 1;

        device_queue default_queue = get_default_device_queue();

        event user_event1 = make_user_event();
        user_event1.retain();
        output->user_event1_is_valid = user_event1.is_valid();

        ndrange ndrange1(get_global_size(0) / 2);
        event event1;
        enqueue_status status1 = queue.enqueue_kernel(enqueue_policy::wait_kernel, 1, &user_event1, &event1, ndrange1,
        [](global<output_type> *output){
            const ulong gid = get_global_id(0);
            output->values[gid * 2] = 1;
        }, output);
        output->enqueue_kernel1_success = status1 == enqueue_status::success;
        output->event1_is_valid = event1.is_valid();
        user_event1.release();

        event user_event2 = make_user_event();
        output->user_event2_is_valid = user_event2.is_valid();

        event events[2];
        events[0] = user_event2;
        events[1] = user_event1;

        ndrange ndrange2(1, get_global_size(0) / 2, get_local_size(0));
        event event2;
        enqueue_status status2 = default_queue.enqueue_kernel(enqueue_policy::no_wait, 2, events, &event2, ndrange2,
        [](global<output_type> *output, local_ptr<int2[]> l0, local_ptr<int[]> l1, local_ptr<int[]> l2) {
            const ulong gid = get_global_id(0);
            const ulong lid = get_local_id(0);
            l1[get_local_size(0) - lid - 1] = gid > 0 ? 1 : 0;
            work_group_barrier(mem_fence::local);
            if (lid < 5) l0[lid] = int2(3, 4);
            if (lid < 3) l2[lid] = 5;
            work_group_barrier(mem_fence::local);
            output->values[(gid - 1) * 2 + 1] = min(l1[lid], min(l0[0].x, l2[0]));
        }, output, local_ptr<int2[]>::size_type(5), local_ptr<int[]>::size_type(get_local_size(0)), local_ptr<int[]>::size_type(3));
        output->enqueue_kernel2_success = status2 == enqueue_status::success;
        output->event2_is_valid = event2.is_valid();

        user_event1.set_status(event_status::complete);
        user_event2.set_status(event_status::complete);

        user_event1.release();
        user_event2.release();
        event1.release();
        event2.release();
    }
    )";
    }
    else if (options.test == 3)
    {
        s << R"(
    kernel void test(device_queue queue, global<output_type> *output)
    {
        const ulong gid = get_global_id(0);

        if (gid != 0)
            return;

        output->user_event2_is_valid = 1;

        device_queue default_queue = get_default_device_queue();

        ndrange ndrange1(get_global_size(0) / 2);
        event event1;
        enqueue_status status1 = default_queue.enqueue_kernel(enqueue_policy::wait_work_group, 0, nullptr, &event1, ndrange1,
        [](global<output_type> *output) {
            const ulong gid = get_global_id(0);
            output->values[gid * 2] = 20;
        }, output);
        output->enqueue_kernel1_success = status1 == enqueue_status::success;
        output->event1_is_valid = event1.is_valid();

        ndrange ndrange2(1, get_global_size(0) / 2, 1);
        event event2;
        enqueue_status status2 = queue.enqueue_kernel(enqueue_policy::wait_kernel, 0, nullptr, &event2, ndrange2,
        [](global<output_type> *output) {
            const ulong gid = get_global_id(0);
            output->values[(gid - 1) * 2 + 1] = 20;
        }, output);
        output->enqueue_kernel2_success = status2 == enqueue_status::success;
        output->event2_is_valid = event2.is_valid();

        event user_event1 = make_user_event();
        output->user_event1_is_valid = user_event1.is_valid();

        event events[3];
        events[0] = event2;
        events[1] = user_event1;
        events[2] = event1;

        event event3;
        enqueue_status status3 = queue.enqueue_marker(3, events, &event3);
        output->enqueue_marker_success = status3 == enqueue_status::success;
        output->event3_is_valid = event3.is_valid();

        enqueue_status status4 = default_queue.enqueue_kernel(enqueue_policy::no_wait, 1, &event3, nullptr, ndrange(get_global_size(0)),
        [](global<output_type> *output) {
            const ulong gid = get_global_id(0);
            output->values[gid] /= 20;
        }, output);
        output->enqueue_kernel3_success = status4 == enqueue_status::success;

        user_event1.set_status(event_status::complete);

        user_event1.release();
        event1.release();
        event2.release();
        event3.release();
    }
    )";
    }

    return s.str();
}
#endif

int test(cl_device_id device, cl_context context, cl_command_queue queue, test_options options)
{
    int error = CL_SUCCESS;

    cl_program program;
    cl_kernel kernel;

    std::string kernel_name = "test";
    std::string source = generate_source(options);

// -----------------------------------------------------------------------------------
// ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
// -----------------------------------------------------------------------------------
// Only OpenCL C++ to SPIR-V compilation
#if defined(DEVELOPMENT) && defined(ONLY_SPIRV_COMPILATION)
    error = create_opencl_kernel(
        context, &program, &kernel,
        source, kernel_name
    );
    RETURN_ON_ERROR(error)
    return error;
// Use OpenCL C kernels instead of OpenCL C++ kernels (test C++ host code)
#elif defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
    error = create_opencl_kernel(
        context, &program, &kernel,
        source, kernel_name, "-cl-std=CL2.0", false
    );
    RETURN_ON_ERROR(error)
// Normal run
#else
    error = create_opencl_kernel(
        context, &program, &kernel,
        source, kernel_name
    );
    RETURN_ON_ERROR(error)
#endif

    cl_uint max_queues;
    error = clGetDeviceInfo(device, CL_DEVICE_MAX_ON_DEVICE_QUEUES, sizeof(cl_uint), &max_queues, NULL);
    RETURN_ON_CL_ERROR(error, "clGetDeviceInfo")

    cl_uint max_events;
    error = clGetDeviceInfo(device, CL_DEVICE_MAX_ON_DEVICE_EVENTS, sizeof(cl_uint), &max_events, NULL);
    RETURN_ON_CL_ERROR(error, "clGetDeviceInfo")

    cl_command_queue device_queue1 = NULL;
    cl_command_queue device_queue2 = NULL;

    cl_queue_properties queue_properties1[] =
    {
        CL_QUEUE_PROPERTIES, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_ON_DEVICE | CL_QUEUE_ON_DEVICE_DEFAULT,
        0
    };
    device_queue1 = clCreateCommandQueueWithProperties(context, device, queue_properties1, &error);
    RETURN_ON_CL_ERROR(error, "clCreateCommandQueueWithProperties")

    if (max_queues > 1)
    {
        cl_queue_properties queue_properties2[] =
        {
            CL_QUEUE_PROPERTIES, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_ON_DEVICE,
            0
        };
        device_queue2 = clCreateCommandQueueWithProperties(context, device, queue_properties2, &error);
        RETURN_ON_CL_ERROR(error, "clCreateCommandQueueWithProperties")
    }

    cl_mem output_buffer;
    output_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(output_type), NULL, &error);
    RETURN_ON_CL_ERROR(error, "clCreateBuffer")

    error = clSetKernelArg(kernel, 0, sizeof(cl_command_queue), device_queue2 != NULL ? &device_queue2 : &device_queue1);
    RETURN_ON_CL_ERROR(error, "clSetKernelArg")
    error = clSetKernelArg(kernel, 1, sizeof(output_buffer), &output_buffer);
    RETURN_ON_CL_ERROR(error, "clSetKernelArg")

    const char pattern = 0;
    error = clEnqueueFillBuffer(queue, output_buffer, &pattern, sizeof(pattern), 0, sizeof(output_type), 0, NULL, NULL);
    RETURN_ON_CL_ERROR(error, "clEnqueueFillBuffer")

    size_t max_work_group_size;
    error = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
    RETURN_ON_CL_ERROR(error, "clGetDeviceInfo")

    const size_t local_size = (std::min)((size_t)256, max_work_group_size);
    const size_t global_size = 10000 / local_size * local_size;
    const size_t count = global_size;
    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    RETURN_ON_CL_ERROR(error, "clEnqueueNDRangeKernel")

    output_type output;
    error = clEnqueueReadBuffer(
        queue, output_buffer, CL_TRUE,
        0, sizeof(output_type),
        static_cast<void *>(&output),
        0, NULL, NULL
    );
    RETURN_ON_CL_ERROR(error, "clEnqueueReadBuffer")

    if (!output.enqueue_kernel1_success)
    {
        RETURN_ON_ERROR_MSG(-1, "enqueue_kernel did not succeed")
    }
    if (!output.enqueue_kernel2_success)
    {
        RETURN_ON_ERROR_MSG(-1, "enqueue_kernel did not succeed")
    }
    if (!output.enqueue_kernel3_success)
    {
        RETURN_ON_ERROR_MSG(-1, "enqueue_kernel did not succeed")
    }
    if (!output.enqueue_marker_success)
    {
        RETURN_ON_ERROR_MSG(-1, "enqueue_marker did not succeed")
    }
    if (!output.event1_is_valid)
    {
        RETURN_ON_ERROR_MSG(-1, "event1 is not valid")
    }
    if (!output.event2_is_valid)
    {
        RETURN_ON_ERROR_MSG(-1, "event2 is not valid")
    }
    if (!output.event3_is_valid)
    {
        RETURN_ON_ERROR_MSG(-1, "event3 is not valid")
    }
    if (!output.user_event1_is_valid)
    {
        RETURN_ON_ERROR_MSG(-1, "user_event1 is not valid")
    }
    if (!output.user_event2_is_valid)
    {
        RETURN_ON_ERROR_MSG(-1, "user_event2 is not valid")
    }

    for (size_t i = 0; i < count; i++)
    {
        const cl_int result = output.values[i];
        const cl_int expected = 1;

        if (result != expected)
        {
            RETURN_ON_ERROR_MSG(-1,
                "kernel did not return correct value. Expected: %s, got: %s",
                format_value(expected).c_str(), format_value(result).c_str()
            )
        }
    }

    clReleaseMemObject(output_buffer);
    clReleaseCommandQueue(device_queue1);
    if (device_queue2 != NULL)
        clReleaseCommandQueue(device_queue2);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    return error;
}

AUTO_TEST_CASE(test_enqueue_one_kernel)
(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    test_options options;
    options.test = 0;
    return test(device, context, queue, options);
}

AUTO_TEST_CASE(test_enqueue_two_kernels)
(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    test_options options;
    options.test = 1;
    return test(device, context, queue, options);
}

AUTO_TEST_CASE(test_enqueue_user_events_and_locals)
(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    test_options options;
    options.test = 2;
    return test(device, context, queue, options);
}

AUTO_TEST_CASE(test_enqueue_marker)
(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    test_options options;
    options.test = 3;
    return test(device, context, queue, options);
}

} // namespace

#endif // TEST_CONFORMANCE_CLCPP_DEVICE_QUEUE_TEST_ENQUEUE_HPP
