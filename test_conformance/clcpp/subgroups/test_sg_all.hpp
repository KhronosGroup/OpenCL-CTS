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
#ifndef TEST_CONFORMANCE_CLCPP_SUBGROUPS_TEST_SG_ALL_HPP
#define TEST_CONFORMANCE_CLCPP_SUBGROUPS_TEST_SG_ALL_HPP

#include <vector>
#include <limits>
#include <algorithm>

// Common for all OpenCL C++ tests
#include "../common.hpp"
// Common for tests of sub-group functions
#include "common.hpp"

std::string generate_sg_all_kernel_code()
{
    return "#include <opencl_memory>\n"
           "#include <opencl_work_item>\n"
           "#include <opencl_work_group>\n"
           "using namespace cl;\n"
           "__kernel void test_sg_all(global_ptr<uint[]> input, global_ptr<uint[]> output)\n"
           "{\n"
           "    ulong tid = get_global_id(0);\n"
           "    bool result = sub_group_all(input[tid] < input[tid+1]);\n"
           "    if(!result) {\n        output[tid] = 0;\n        return;\n    }\n"
           "    output[tid] = 1;\n"
           "}\n";
}

int verify_sg_all(const std::vector<cl_uint> &in, const std::vector<cl_uint> &out, size_t count, size_t wg_size, size_t sg_size)
{
    size_t i, j, k;
    for (i = 0; i < count; i += wg_size)
    {
        for (j = 0; j < ((count - i) > wg_size ? wg_size : (count - i)); j+= sg_size)
        {
            // sub-group all
            bool all = true;
            for (k = 0; k < ((wg_size - j) > sg_size ? sg_size : (wg_size - j)); k++)
            {
                if(!(in[i+j+k] < in[i+j+k+1]))
                {
                    all = false;
                    break;
                }
            }

            // Convert bool to uint
            cl_uint all_uint = all ? 1 : 0;
            // Check if all work-items in sub-group stored correct value
            for (k = 0; k < ((wg_size - j) > sg_size ? sg_size : (wg_size - j)); k++)
            {
                if (all_uint != out[i + j + k])
                {
                    log_info(
                        "sub_group_all %s: Error at %lu: expected = %lu, got = %lu\n",
                        type_name<cl_uint>().c_str(),
                        i + j,
                        static_cast<size_t>(all_uint),
                        static_cast<size_t>(out[i + j + k]));
                    return -1;
                }
            }
        }
    }
    return CL_SUCCESS;
}

std::vector<cl_uint> generate_input_sg_all(size_t count, size_t wg_size)
{
    std::vector<cl_uint> input(count, cl_uint(0));
    size_t j = wg_size;
    for(size_t i = 0; i < count; i++)
    {
        input[i] = static_cast<cl_uint>(i);
        // In one place in ~half of work-groups (input[tid] < input[tid+1]) will
        // generate false, it means that for sub_group_all(input[tid] < input[tid+1])
        // should return false for all sub-groups in that work-groups
        if((j == wg_size/2) && (i > count/2))
        {
            input[i] = input[i - 1];
        }
        j--;
        if(j == 0)
        {
            j = wg_size;
        }
    }
    return input;
}

std::vector<cl_uint> generate_output_sg_all(size_t count, size_t wg_size)
{
    (void) wg_size;
    return std::vector<cl_uint>(count, cl_uint(1));
}

int sub_group_all(cl_device_id device, cl_context context, cl_command_queue queue, size_t count)
{
    cl_mem buffers[2];
    cl_program program;
    cl_kernel kernel;
    size_t wg_size;
    size_t sg_max_size;
    size_t work_size[1];
    int err;

    std::string code_str = generate_sg_all_kernel_code();
// -----------------------------------------------------------------------------------
// ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
// -----------------------------------------------------------------------------------
// Only OpenCL C++ to SPIR-V compilation
#if defined(DEVELOPMENT) && defined(ONLY_SPIRV_COMPILATION)
    err = create_opencl_kernel(context, &program, &kernel, code_str, "test_sg_all");
    RETURN_ON_ERROR(err)
    return err;
// Use OpenCL C kernels instead of OpenCL C++ kernels (test C++ host code)
#elif defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
    log_info("SKIPPED: OpenCL C kernels not provided for this test. Skipping the test.\n");
    return CL_SUCCESS;
#else
    err = create_opencl_kernel(context, &program, &kernel, code_str, "test_sg_all");
    RETURN_ON_ERROR(err)
#endif

    err = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &wg_size, NULL);
    RETURN_ON_CL_ERROR(err, "clGetKernelWorkGroupInfo")

    size_t param_value_size = 0;
    err = clGetKernelSubGroupInfo(
        kernel, device, CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE,
        sizeof(size_t), static_cast<void*>(&wg_size),
        sizeof(size_t), static_cast<void*>(&sg_max_size),
        &param_value_size
    );
    RETURN_ON_CL_ERROR(err, "clGetKernelSubGroupInfo")

    // Verify size of returned param
    if(param_value_size != sizeof(size_t))
    {
        RETURN_ON_ERROR_MSG(-1,
            "Returned size of max sub group size not valid! (Expected %lu, got %lu)\n",
            sizeof(size_t),
            param_value_size
        )
    }

    // Calculate global work size
    size_t flat_work_size;
    size_t wg_number = static_cast<size_t>(
        std::ceil(static_cast<double>(count) / wg_size)
    );
    flat_work_size = wg_number * wg_size;
    work_size[0] = flat_work_size;

    std::vector<cl_uint> input = generate_input_sg_all(flat_work_size + 1, wg_size);
    std::vector<cl_uint> output = generate_output_sg_all(flat_work_size, wg_size);

    buffers[0] = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE), sizeof(cl_uint) * input.size(), NULL, &err);
    RETURN_ON_CL_ERROR(err, "clCreateBuffer");

    buffers[1] = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE), sizeof(cl_uint) * output.size(), NULL, &err);
    RETURN_ON_CL_ERROR(err, "clCreateBuffer");

    err = clEnqueueWriteBuffer(
        queue, buffers[0], CL_TRUE, 0, sizeof(cl_uint) * input.size(),
        static_cast<void *>(input.data()), 0, NULL, NULL
    );
    RETURN_ON_CL_ERROR(err, "clEnqueueWriteBuffer");

    err = clSetKernelArg(kernel, 0, sizeof(buffers[0]), &buffers[0]);
    err |= clSetKernelArg(kernel, 1, sizeof(buffers[1]), &buffers[1]);
    RETURN_ON_CL_ERROR(err, "clSetKernelArg");

    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, work_size, &wg_size, 0, NULL, NULL);
    RETURN_ON_CL_ERROR(err, "clEnqueueNDRangeKernel");

    err = clEnqueueReadBuffer(
        queue, buffers[1], CL_TRUE, 0, sizeof(cl_uint) * output.size(),
        static_cast<void *>(output.data()), 0, NULL, NULL
    );
    RETURN_ON_CL_ERROR(err, "clEnqueueReadBuffer");

    if (verify_sg_all(input, output, flat_work_size, wg_size, sg_max_size) != CL_SUCCESS)
    {
        RETURN_ON_ERROR_MSG(-1, "sub_group_all failed");
    }
    log_info("sub_group_all passed\n");

    clReleaseMemObject(buffers[0]);
    clReleaseMemObject(buffers[1]);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    return err;
}

AUTO_TEST_CASE(test_sub_group_all)
(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
    int err = CL_SUCCESS;
    err = sub_group_all(device, context, queue, n_elems);
    CHECK_ERROR(err)
    return err;
}

#endif // TEST_CONFORMANCE_CLCPP_SUBGROUPS_TEST_SG_ALL_HPP
