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
#ifndef TEST_CONFORMANCE_CLCPP_PIPES_TEST_PIPES_HPP
#define TEST_CONFORMANCE_CLCPP_PIPES_TEST_PIPES_HPP

#include <sstream>
#include <string>
#include <tuple>
#include <vector>
#include <algorithm>

// Common for all OpenCL C++ tests
#include "../common.hpp"


namespace test_pipes {

enum class pipe_source
{
    param,
    storage
};

enum class pipe_operation
{
    work_item,
    work_item_reservation,
    work_group_reservation,
    sub_group_reservation
};

struct test_options
{
    pipe_operation operation;
    pipe_source source;
    int max_packets;
    int num_packets;
};

struct output_type
{
    cl_uint write_reservation_is_valid;
    cl_uint write_success;

    cl_uint num_packets;
    cl_uint max_packets;
    cl_uint read_reservation_is_valid;
    cl_uint read_success;

    cl_uint value;
};

const std::string source_common = R"(
struct output_type
{
    uint write_reservation_is_valid;
    uint write_success;

    uint num_packets;
    uint max_packets;
    uint read_reservation_is_valid;
    uint read_success;

    uint value;
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
    if (options.operation == pipe_operation::work_item)
    {
        s << R"(
    kernel void producer(write_only pipe uint out_pipe, global struct output_type *output)
    {
        const ulong gid = get_global_id(0);

        output[gid].write_reservation_is_valid = 1;

        uint value = gid;
        output[gid].write_success = write_pipe(out_pipe, &value) == 0;
    }

    kernel void consumer(read_only pipe uint in_pipe, global struct output_type *output)
    {
        const ulong gid = get_global_id(0);

        output[gid].num_packets = get_pipe_num_packets(in_pipe);
        output[gid].max_packets = get_pipe_max_packets(in_pipe);

        output[gid].read_reservation_is_valid = 1;

        uint value;
        output[gid].read_success = read_pipe(in_pipe, &value) == 0;
        output[gid].value = value;
    }
    )";
    }
    else if (options.operation == pipe_operation::work_item_reservation)
    {
        s << R"(
    kernel void producer(write_only pipe uint out_pipe, global struct output_type *output)
    {
        const ulong gid = get_global_id(0);
        if (gid % 2 == 1) return;

        reserve_id_t reservation = reserve_write_pipe(out_pipe, 2);
        output[gid + 0].write_reservation_is_valid = is_valid_reserve_id(reservation);
        output[gid + 1].write_reservation_is_valid = is_valid_reserve_id(reservation);

        uint value0 = gid + 0;
        uint value1 = gid + 1;
        output[gid + 0].write_success = write_pipe(out_pipe, reservation, 0, &value0) == 0;
        output[gid + 1].write_success = write_pipe(out_pipe, reservation, 1, &value1) == 0;
        commit_write_pipe(out_pipe, reservation);
    }

    kernel void consumer(read_only pipe uint in_pipe, global struct output_type *output)
    {
        const ulong gid = get_global_id(0);
        if (gid % 2 == 1) return;

        output[gid + 0].num_packets = get_pipe_num_packets(in_pipe);
        output[gid + 0].max_packets = get_pipe_max_packets(in_pipe);
        output[gid + 1].num_packets = get_pipe_num_packets(in_pipe);
        output[gid + 1].max_packets = get_pipe_max_packets(in_pipe);

        reserve_id_t reservation = reserve_read_pipe(in_pipe, 2);
        output[gid + 0].read_reservation_is_valid = is_valid_reserve_id(reservation);
        output[gid + 1].read_reservation_is_valid = is_valid_reserve_id(reservation);

        uint value0;
        uint value1;
        output[gid + 0].read_success = read_pipe(in_pipe, reservation, 1, &value0) == 0;
        output[gid + 1].read_success = read_pipe(in_pipe, reservation, 0, &value1) == 0;
        commit_read_pipe(in_pipe, reservation);
        output[gid + 0].value = value0;
        output[gid + 1].value = value1;
    }
    )";
    }
    else if (options.operation == pipe_operation::work_group_reservation)
    {
        s << R"(
    kernel void producer(write_only pipe uint out_pipe, global struct output_type *output)
    {
        const ulong gid = get_global_id(0);

        reserve_id_t reservation = work_group_reserve_write_pipe(out_pipe, get_local_size(0));
        output[gid].write_reservation_is_valid = is_valid_reserve_id(reservation);

        uint value = gid;
        output[gid].write_success = write_pipe(out_pipe, reservation, get_local_id(0), &value) == 0;
        work_group_commit_write_pipe(out_pipe, reservation);
    }

    kernel void consumer(read_only pipe uint in_pipe, global struct output_type *output)
    {
        const ulong gid = get_global_id(0);

        output[gid].num_packets = get_pipe_num_packets(in_pipe);
        output[gid].max_packets = get_pipe_max_packets(in_pipe);

        reserve_id_t reservation = work_group_reserve_read_pipe(in_pipe, get_local_size(0));
        output[gid].read_reservation_is_valid = is_valid_reserve_id(reservation);

        uint value;
        output[gid].read_success = read_pipe(in_pipe, reservation, get_local_size(0) - 1 - get_local_id(0), &value) == 0;
        work_group_commit_read_pipe(in_pipe, reservation);
        output[gid].value = value;
    }
    )";
    }
    else if (options.operation == pipe_operation::sub_group_reservation)
    {
        s << R"(
    #pragma OPENCL EXTENSION cl_khr_subgroups : enable

    kernel void producer(write_only pipe uint out_pipe, global struct output_type *output)
    {
        const ulong gid = get_global_id(0);

        reserve_id_t reservation = sub_group_reserve_write_pipe(out_pipe, get_sub_group_size());
        output[gid].write_reservation_is_valid = is_valid_reserve_id(reservation);

        uint value = gid;
        output[gid].write_success = write_pipe(out_pipe, reservation, get_sub_group_local_id(), &value) == 0;
        sub_group_commit_write_pipe(out_pipe, reservation);
    }

    kernel void consumer(read_only pipe uint in_pipe, global struct output_type *output)
    {
        const ulong gid = get_global_id(0);

        output[gid].num_packets = get_pipe_num_packets(in_pipe);
        output[gid].max_packets = get_pipe_max_packets(in_pipe);

        reserve_id_t reservation = sub_group_reserve_read_pipe(in_pipe, get_sub_group_size());
        output[gid].read_reservation_is_valid = is_valid_reserve_id(reservation);

        uint value;
        output[gid].read_success = read_pipe(in_pipe, reservation, get_sub_group_size() - 1 - get_sub_group_local_id(), &value) == 0;
        sub_group_commit_read_pipe(in_pipe, reservation);
        output[gid].value = value;
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
    #include <opencl_pipe>
    using namespace cl;
    )";

    s << source_common;

    std::string init_out_pipe;
    std::string init_in_pipe;
    if (options.source == pipe_source::param)
    {
        init_out_pipe = "auto out_pipe = pipe_param;";
        init_in_pipe = "auto in_pipe = pipe_param;";
    }
    else if (options.source == pipe_source::storage)
    {
        s << "pipe_storage<uint, " << std::to_string(options.max_packets) << "> storage;";
        init_out_pipe = "auto out_pipe = storage.get<pipe_access::write>();";
        init_in_pipe = "auto in_pipe = make_pipe(storage);";
    }

    if (options.operation == pipe_operation::work_item)
    {
        s << R"(
    kernel void producer(pipe<uint, pipe_access::write> pipe_param, global_ptr<output_type[]> output)
    {
        )" << init_out_pipe << R"(
        const ulong gid = get_global_id(0);

        output[gid].write_reservation_is_valid = 1;

        uint value = gid;
        output[gid].write_success = out_pipe.write(value);
    }

    kernel void consumer(pipe<uint, pipe_access::read> pipe_param, global_ptr<output_type[]> output)
    {
        )" << init_in_pipe << R"(
        const ulong gid = get_global_id(0);

        output[gid].num_packets = in_pipe.num_packets();
        output[gid].max_packets = in_pipe.max_packets();

        output[gid].read_reservation_is_valid = 1;

        uint value;
        output[gid].read_success = in_pipe.read(value);
        output[gid].value = value;
    }
    )";
    }
    else if (options.operation == pipe_operation::work_item_reservation)
    {
        s << R"(
    kernel void producer(pipe<uint, pipe_access::write> pipe_param, global_ptr<output_type[]> output)
    {
        )" << init_out_pipe << R"(
        const ulong gid = get_global_id(0);
        if (gid % 2 == 1) return;

        auto reservation = out_pipe.reserve(2);
        output[gid + 0].write_reservation_is_valid = reservation.is_valid();
        output[gid + 1].write_reservation_is_valid = reservation.is_valid();

        uint value0 = gid + 0;
        uint value1 = gid + 1;
        output[gid + 0].write_success = reservation.write(0, value0);
        output[gid + 1].write_success = reservation.write(1, value1);
        reservation.commit();
    }

    kernel void consumer(pipe<uint, pipe_access::read> pipe_param, global_ptr<output_type[]> output)
    {
        )" << init_in_pipe << R"(
        const ulong gid = get_global_id(0);
        if (gid % 2 == 1) return;

        output[gid + 0].num_packets = in_pipe.num_packets();
        output[gid + 0].max_packets = in_pipe.max_packets();
        output[gid + 1].num_packets = in_pipe.num_packets();
        output[gid + 1].max_packets = in_pipe.max_packets();

        auto reservation = in_pipe.reserve(2);
        output[gid + 0].read_reservation_is_valid = reservation.is_valid();
        output[gid + 1].read_reservation_is_valid = reservation.is_valid();

        uint value0;
        uint value1;
        output[gid + 0].read_success = reservation.read(1, value0);
        output[gid + 1].read_success = reservation.read(0, value1);
        reservation.commit();
        output[gid + 0].value = value0;
        output[gid + 1].value = value1;
    }
    )";
    }
    else if (options.operation == pipe_operation::work_group_reservation)
    {
        s << R"(
    kernel void producer(pipe<uint, pipe_access::write> pipe_param, global_ptr<output_type[]> output)
    {
        )" << init_out_pipe << R"(
        const ulong gid = get_global_id(0);

        auto reservation = out_pipe.work_group_reserve(get_local_size(0));
        output[gid].write_reservation_is_valid = reservation.is_valid();

        uint value = gid;
        output[gid].write_success = reservation.write(get_local_id(0), value);
        reservation.commit();
    }

    kernel void consumer(pipe<uint, pipe_access::read> pipe_param, global_ptr<output_type[]> output)
    {
        )" << init_in_pipe << R"(
        const ulong gid = get_global_id(0);

        output[gid].num_packets = in_pipe.num_packets();
        output[gid].max_packets = in_pipe.max_packets();

        auto reservation = in_pipe.work_group_reserve(get_local_size(0));
        output[gid].read_reservation_is_valid = reservation.is_valid();

        uint value;
        output[gid].read_success = reservation.read(get_local_size(0) - 1 - get_local_id(0), value);
        reservation.commit();
        output[gid].value = value;
    }
    )";
    }
    else if (options.operation == pipe_operation::sub_group_reservation)
    {
        s << R"(
    kernel void producer(pipe<uint, pipe_access::write> pipe_param, global_ptr<output_type[]> output)
    {
        )" << init_out_pipe << R"(
        const ulong gid = get_global_id(0);

        auto reservation = out_pipe.sub_group_reserve(get_sub_group_size());
        output[gid].write_reservation_is_valid = reservation.is_valid();

        uint value = gid;
        output[gid].write_success = reservation.write(get_sub_group_local_id(), value);
        reservation.commit();
    }

    kernel void consumer(pipe<uint, pipe_access::read> pipe_param, global_ptr<output_type[]> output)
    {
        )" << init_in_pipe << R"(
        const ulong gid = get_global_id(0);

        output[gid].num_packets = in_pipe.num_packets();
        output[gid].max_packets = in_pipe.max_packets();

        auto reservation = in_pipe.sub_group_reserve(get_sub_group_size());
        output[gid].read_reservation_is_valid = reservation.is_valid();

        uint value;
        output[gid].read_success = reservation.read(get_sub_group_size() - 1 - get_sub_group_local_id(), value);
        reservation.commit();
        output[gid].value = value;
    }
    )";
    }

    return s.str();
}
#endif

int test(cl_device_id device, cl_context context, cl_command_queue queue, test_options options)
{
    int error = CL_SUCCESS;

    if (options.num_packets % 2 != 0 || options.max_packets < options.num_packets)
    {
        RETURN_ON_ERROR_MSG(-1, "Invalid test options")
    }

#if defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
    if (options.operation == pipe_operation::sub_group_reservation && !is_extension_available(device, "cl_khr_subgroups"))
    {
        log_info("SKIPPED: Extension `cl_khr_subgroups` is not supported. Skipping tests.\n");
        return CL_SUCCESS;
    }
#endif

    cl_program program;
    cl_kernel producer_kernel;
    cl_kernel consumer_kernel;

    std::string producer_kernel_name = "producer";
    std::string consumer_kernel_name = "consumer";
    std::string source = generate_source(options);

// -----------------------------------------------------------------------------------
// ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
// -----------------------------------------------------------------------------------
// Only OpenCL C++ to SPIR-V compilation
#if defined(DEVELOPMENT) && defined(ONLY_SPIRV_COMPILATION)
    error = create_opencl_kernel(
        context, &program, &producer_kernel,
        source, producer_kernel_name
    );
    RETURN_ON_ERROR(error)
    return error;
// Use OpenCL C kernels instead of OpenCL C++ kernels (test C++ host code)
#elif defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
    error = create_opencl_kernel(
        context, &program, &producer_kernel,
        source, producer_kernel_name, "-cl-std=CL2.0", false
    );
    RETURN_ON_ERROR(error)
    consumer_kernel = clCreateKernel(program, consumer_kernel_name.c_str(), &error);
    RETURN_ON_CL_ERROR(error, "clCreateKernel")
// Normal run
#else
    error = create_opencl_kernel(
        context, &program, &producer_kernel,
        source, producer_kernel_name
    );
    RETURN_ON_ERROR(error)
    consumer_kernel = clCreateKernel(program, consumer_kernel_name.c_str(), &error);
    RETURN_ON_CL_ERROR(error, "clCreateKernel")
#endif

    size_t max_work_group_size;
    error = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
    RETURN_ON_CL_ERROR(error, "clGetDeviceInfo")

    const size_t count = options.num_packets;
    const size_t local_size = (std::min)((size_t)256, max_work_group_size);
    const size_t global_size = count;

    const cl_uint packet_size = sizeof(cl_uint);

    cl_mem pipe = clCreatePipe(context, 0, packet_size, options.max_packets, NULL, &error);
    RETURN_ON_CL_ERROR(error, "clCreatePipe")

    cl_mem output_buffer;
    output_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(output_type) * count, NULL, &error);
    RETURN_ON_CL_ERROR(error, "clCreateBuffer")

    const char pattern = 0;
    error = clEnqueueFillBuffer(queue, output_buffer, &pattern, sizeof(pattern), 0, sizeof(output_type) * count, 0, NULL, NULL);
    RETURN_ON_CL_ERROR(error, "clEnqueueFillBuffer")

    error = clSetKernelArg(producer_kernel, 0, sizeof(cl_mem), &pipe);
    RETURN_ON_CL_ERROR(error, "clSetKernelArg")
    error = clSetKernelArg(producer_kernel, 1, sizeof(output_buffer), &output_buffer);
    RETURN_ON_CL_ERROR(error, "clSetKernelArg")

    error = clEnqueueNDRangeKernel(queue, producer_kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    RETURN_ON_CL_ERROR(error, "clEnqueueNDRangeKernel")

    error = clSetKernelArg(consumer_kernel, 0, sizeof(cl_mem), &pipe);
    RETURN_ON_CL_ERROR(error, "clSetKernelArg")
    error = clSetKernelArg(consumer_kernel, 1, sizeof(output_buffer), &output_buffer);
    RETURN_ON_CL_ERROR(error, "clSetKernelArg")

    error = clEnqueueNDRangeKernel(queue, consumer_kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    RETURN_ON_CL_ERROR(error, "clEnqueueNDRangeKernel")

    std::vector<output_type> output(count);
    error = clEnqueueReadBuffer(
        queue, output_buffer, CL_TRUE,
        0, sizeof(output_type) * count,
        static_cast<void *>(output.data()),
        0, NULL, NULL
    );
    RETURN_ON_CL_ERROR(error, "clEnqueueReadBuffer")

    std::vector<bool> existing_values(count, false);
    for (size_t gid = 0; gid < count; gid++)
    {
        const output_type &o = output[gid];

        if (!o.write_reservation_is_valid)
        {
            RETURN_ON_ERROR_MSG(-1, "write reservation is not valid")
        }
        if (!o.write_success)
        {
            RETURN_ON_ERROR_MSG(-1, "write did not succeed")
        }

        if (o.num_packets == 0 || o.num_packets > options.num_packets)
        {
            RETURN_ON_ERROR_MSG(-1, "num_packets did not return correct value")
        }
        if (o.max_packets != options.max_packets)
        {
            RETURN_ON_ERROR_MSG(-1, "max_packets did not return correct value")
        }
        if (!o.read_reservation_is_valid)
        {
            RETURN_ON_ERROR_MSG(-1, "read reservation is not valid")
        }
        if (!o.read_success)
        {
            RETURN_ON_ERROR_MSG(-1, "read did not succeed")
        }

        // Every value must be presented once in any order
        if (o.value >= count || existing_values[o.value])
        {
            RETURN_ON_ERROR_MSG(-1, "kernel did not return correct value")
        }
        existing_values[o.value] = true;
    }

    clReleaseMemObject(pipe);
    clReleaseMemObject(output_buffer);
    clReleaseKernel(producer_kernel);
    clReleaseKernel(consumer_kernel);
    clReleaseProgram(program);
    return error;
}

const pipe_operation pipe_operations[] = {
    pipe_operation::work_item,
    pipe_operation::work_item_reservation,
    pipe_operation::work_group_reservation,
    pipe_operation::sub_group_reservation
};

const std::tuple<int, int> max_and_num_packets[] = {
    std::make_tuple<int, int>(2, 2),
    std::make_tuple<int, int>(10, 8),
    std::make_tuple<int, int>(256, 254),
    std::make_tuple<int, int>(1 << 16, 1 << 16),
    std::make_tuple<int, int>((1 << 16) + 5, 1 << 16),
    std::make_tuple<int, int>(12345, 12344),
    std::make_tuple<int, int>(1 << 18, 1 << 18)
};

AUTO_TEST_CASE(test_pipes_pipe)
(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    std::vector<std::tuple<int, int>> ps;
    for (auto p : max_and_num_packets)
    {
        if (std::get<0>(p) < num_elements)
            ps.push_back(p);
    }
    ps.push_back(std::tuple<int, int>(num_elements, num_elements));

    int error = CL_SUCCESS;

    for (auto operation : pipe_operations)
    for (auto p : ps)
    {
        test_options options;
        options.source = pipe_source::param;
        options.max_packets = std::get<0>(p);
        options.num_packets = std::get<1>(p);
        options.operation = operation;

        error = test(device, context, queue, options);
        RETURN_ON_ERROR(error)
    }

    return error;
}

AUTO_TEST_CASE(test_pipes_pipe_storage)
(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    std::vector<std::tuple<int, int>> ps;
    for (auto p : max_and_num_packets)
    {
        if (std::get<0>(p) < num_elements)
            ps.push_back(p);
    }
    ps.push_back(std::tuple<int, int>(num_elements, num_elements));

    int error = CL_SUCCESS;

    for (auto operation : pipe_operations)
    for (auto p : ps)
    {
        test_options options;
        options.source = pipe_source::storage;
        options.max_packets = std::get<0>(p);
        options.num_packets = std::get<1>(p);
        options.operation = operation;

        error = test(device, context, queue, options);
        RETURN_ON_ERROR(error)
    }

    return error;
}

} // namespace

#endif // TEST_CONFORMANCE_CLCPP_PIPES_TEST_PIPES_HPP
