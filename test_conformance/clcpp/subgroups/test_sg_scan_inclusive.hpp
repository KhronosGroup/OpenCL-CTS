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
#ifndef TEST_CONFORMANCE_CLCPP_SUBGROUPS_TEST_SG_SCAN_INCLUSIVE_HPP
#define TEST_CONFORMANCE_CLCPP_SUBGROUPS_TEST_SG_SCAN_INCLUSIVE_HPP

#include <vector>
#include <algorithm>

// Common for all OpenCL C++ tests
#include "../common.hpp"
// Common for tests of sub-group functions
#include "common.hpp"

template <class CL_INT_TYPE, work_group_op op>
std::string generate_sg_scan_inclusive_kernel_code()
{
    return "#include <opencl_memory>\n"
           "#include <opencl_work_item>\n"
           "#include <opencl_work_group>\n"
           "using namespace cl;\n"
           "__kernel void test_sg_scan_inclusive(global_ptr<" + type_name<CL_INT_TYPE>() + "[]> input, "
                                                "global_ptr<" + type_name<CL_INT_TYPE>() + "[]> output)\n"
           "{\n"
           "    ulong tid = get_global_id(0);\n"
           "    " + type_name<CL_INT_TYPE>() + " result = sub_group_scan_inclusive<work_group_op::" + to_string(op) + ">(input[tid]);\n"
           "    output[tid] = result;\n"
           "}\n";
}

template <class CL_INT_TYPE>
int verify_sg_scan_inclusive_add(const std::vector<CL_INT_TYPE> &in, const std::vector<CL_INT_TYPE> &out, size_t wg_size, size_t sg_size)
{
    size_t i, j, k;
    for (i = 0; i < in.size(); i += wg_size)
    {
        for (j = 0; j < ((in.size() - i) > wg_size ? wg_size : (in.size() - i)); j+= sg_size)
        {
            CL_INT_TYPE sum = 0;
            // Check if all work-items in sub-group stored correct value
            for (k = 0; k < ((wg_size - j) > sg_size ? sg_size : (wg_size - j)); k++)
            {
                sum += in[i + j + k];
                if (sum != out[i + j + k])
                {
                    log_info(
                        "sub_group_scan_exclusive_add %s: Error at %lu: expected = %lu, got = %lu\n",
                        type_name<cl_uint>().c_str(),
                        i + j,
                        static_cast<size_t>(sum),
                        static_cast<size_t>(out[i + j + k]));
                    return -1;
                }
            }
        }
    }
    return CL_SUCCESS;
}

template <class CL_INT_TYPE>
int verify_sg_scan_inclusive_min(const std::vector<CL_INT_TYPE> &in, const std::vector<CL_INT_TYPE> &out, size_t wg_size, size_t sg_size)
{
    size_t i, j, k;
    for (i = 0; i < in.size(); i += wg_size)
    {
        for (j = 0; j < ((in.size() - i) > wg_size ? wg_size : (in.size() - i)); j+= sg_size)
        {
            CL_INT_TYPE min = (std::numeric_limits<CL_INT_TYPE>::max)();
            // Check if all work-items in sub-group stored correct value
            for (k = 0; k < ((wg_size - j) > sg_size ? sg_size : (wg_size - j)); k++)
            {
                min = std::min<CL_INT_TYPE>(min, in[i + j + k]);
                if (min != out[i + j + k])
                {
                    log_info(
                        "sub_group_scan_exclusive_min %s: Error at %lu: expected = %lu, got = %lu\n",
                        type_name<cl_uint>().c_str(),
                        i + j,
                        static_cast<size_t>(min),
                        static_cast<size_t>(out[i + j + k]));
                    return -1;
                }
            }
        }
    }
    return CL_SUCCESS;
}

template <class CL_INT_TYPE>
int verify_sg_scan_inclusive_max(const std::vector<CL_INT_TYPE> &in, const std::vector<CL_INT_TYPE> &out, size_t wg_size, size_t sg_size)
{
    size_t i, j, k;
    for (i = 0; i < in.size(); i += wg_size)
    {
        for (j = 0; j < ((in.size() - i) > wg_size ? wg_size : (in.size() - i)); j+= sg_size)
        {
            CL_INT_TYPE max = (std::numeric_limits<CL_INT_TYPE>::min)();
            // Check if all work-items in sub-group stored correct value
            for (k = 0; k < ((wg_size - j) > sg_size ? sg_size : (wg_size - j)); k++)
            {
                max = std::max<CL_INT_TYPE>(max, in[i + j + k]);
                if (max != out[i + j + k])
                {
                    log_info(
                        "sub_group_scan_exclusive_max %s: Error at %lu: expected = %lu, got = %lu\n",
                        type_name<cl_uint>().c_str(),
                        i + j,
                        static_cast<size_t>(max),
                        static_cast<size_t>(out[i + j + k]));
                    return -1;
                }
            }
        }
    }
    return CL_SUCCESS;
}

template <class CL_INT_TYPE, work_group_op op>
int verify_sg_scan_inclusive(const std::vector<CL_INT_TYPE> &in, const std::vector<CL_INT_TYPE> &out, size_t wg_size, size_t sg_size)
{
    switch (op)
    {
        case work_group_op::add:
            return verify_sg_scan_inclusive_add(in, out, wg_size, sg_size);
        case work_group_op::min:
            return verify_sg_scan_inclusive_min(in, out, wg_size, sg_size);
        case work_group_op::max:
            return verify_sg_scan_inclusive_max(in, out, wg_size, sg_size);
    }
    return -1;
}

template <class CL_INT_TYPE, work_group_op op>
int sub_group_scan_inclusive(cl_device_id device, cl_context context, cl_command_queue queue, size_t count)
{
    // don't run test for unsupported types
    if(!type_supported<CL_INT_TYPE>(device))
    {
        return CL_SUCCESS;
    }

    cl_mem buffers[2];
    cl_program program;
    cl_kernel kernel;
    size_t wg_size;
    size_t sg_max_size;
    size_t work_size[1];
    int err;

    std::string code_str = generate_sg_scan_inclusive_kernel_code<CL_INT_TYPE, op>();
// -----------------------------------------------------------------------------------
// ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
// -----------------------------------------------------------------------------------
// Only OpenCL C++ to SPIR-V compilation
#if defined(DEVELOPMENT) && defined(ONLY_SPIRV_COMPILATION)
    err = create_opencl_kernel(context, &program, &kernel, code_str, "test_sg_scan_inclusive");
    RETURN_ON_ERROR(err)
    return err;
// Use OpenCL C kernels instead of OpenCL C++ kernels (test C++ host code)
#elif defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
    log_info("SKIPPED: OpenCL C kernels not provided for this test. Skipping the test.\n");
    return CL_SUCCESS;
#else
    err = create_opencl_kernel(context, &program, &kernel, code_str, "test_sg_scan_inclusive");
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

    std::vector<CL_INT_TYPE> input = generate_input<CL_INT_TYPE, op>(flat_work_size, wg_size);
    std::vector<CL_INT_TYPE> output = generate_output<CL_INT_TYPE, op>(flat_work_size, wg_size);

    buffers[0] = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE), sizeof(CL_INT_TYPE) * input.size(), NULL, &err);
    RETURN_ON_CL_ERROR(err, "clCreateBuffer");

    buffers[1] = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE), sizeof(CL_INT_TYPE) * output.size(), NULL, &err);
    RETURN_ON_CL_ERROR(err, "clCreateBuffer");

    err = clEnqueueWriteBuffer(
        queue, buffers[0], CL_TRUE, 0, sizeof(CL_INT_TYPE) * input.size(),
        static_cast<void *>(input.data()), 0, NULL, NULL
    );
    RETURN_ON_CL_ERROR(err, "clEnqueueWriteBuffer");

    err = clSetKernelArg(kernel, 0, sizeof(buffers[0]), &buffers[0]);
    err |= clSetKernelArg(kernel, 1, sizeof(buffers[1]), &buffers[1]);
    RETURN_ON_CL_ERROR(err, "clSetKernelArg");

    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, work_size, &wg_size, 0, NULL, NULL);
    RETURN_ON_CL_ERROR(err, "clEnqueueNDRangeKernel");

    err = clEnqueueReadBuffer(
        queue, buffers[1], CL_TRUE, 0, sizeof(CL_INT_TYPE) * output.size(),
        static_cast<void *>(output.data()), 0, NULL, NULL
    );
    RETURN_ON_CL_ERROR(err, "clEnqueueReadBuffer");

    if (verify_sg_scan_inclusive<CL_INT_TYPE, op>(input, output, wg_size, sg_max_size) != CL_SUCCESS)
    {
        RETURN_ON_ERROR_MSG(-1, "sub_group_scan_inclusive_%s %s failed", to_string(op).c_str(), type_name<CL_INT_TYPE>().c_str());
    }
    log_info("sub_group_scan_inclusive_%s %s passed\n", to_string(op).c_str(), type_name<CL_INT_TYPE>().c_str());

    clReleaseMemObject(buffers[0]);
    clReleaseMemObject(buffers[1]);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    return err;
}

AUTO_TEST_CASE(test_sub_group_scan_inclusive_add)
(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
    int error = CL_SUCCESS;
    int local_error = CL_SUCCESS;

    local_error = sub_group_scan_inclusive<cl_int, work_group_op::add>(device, context, queue, n_elems);
    CHECK_ERROR(local_error)
    error |= local_error;

    local_error = sub_group_scan_inclusive<cl_uint, work_group_op::add>(device, context, queue, n_elems);
    CHECK_ERROR(local_error)
    error |= local_error;

    local_error = sub_group_scan_inclusive<cl_long, work_group_op::add>(device, context, queue, n_elems);
    CHECK_ERROR(local_error)
    error |= local_error;

    local_error = sub_group_scan_inclusive<cl_ulong, work_group_op::add>(device, context, queue, n_elems);
    CHECK_ERROR(local_error)
    error |= local_error;

    if(error != CL_SUCCESS)
        return -1;
    return CL_SUCCESS;
}

AUTO_TEST_CASE(test_sub_group_scan_inclusive_min)
(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
    int error = CL_SUCCESS;
    int local_error = CL_SUCCESS;

    local_error = sub_group_scan_inclusive<cl_int, work_group_op::min>(device, context, queue, n_elems);
    CHECK_ERROR(local_error)
    error |= local_error;

    local_error = sub_group_scan_inclusive<cl_uint, work_group_op::min>(device, context, queue, n_elems);
    CHECK_ERROR(local_error)
    error |= local_error;

    local_error = sub_group_scan_inclusive<cl_long, work_group_op::min>(device, context, queue, n_elems);
    CHECK_ERROR(local_error)
    error |= local_error;

    local_error = sub_group_scan_inclusive<cl_ulong, work_group_op::min>(device, context, queue, n_elems);
    CHECK_ERROR(local_error)
    error |= local_error;

    if(error != CL_SUCCESS)
        return -1;
    return CL_SUCCESS;
}

AUTO_TEST_CASE(test_sub_group_scan_inclusive_max)
(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
    int error = CL_SUCCESS;
    int local_error = CL_SUCCESS;

    local_error = sub_group_scan_inclusive<cl_int, work_group_op::max>(device, context, queue, n_elems);
    CHECK_ERROR(local_error)
    error |= local_error;

    local_error = sub_group_scan_inclusive<cl_uint, work_group_op::max>(device, context, queue, n_elems);
    CHECK_ERROR(local_error)
    error |= local_error;

    local_error = sub_group_scan_inclusive<cl_long, work_group_op::max>(device, context, queue, n_elems);
    CHECK_ERROR(local_error)
    error |= local_error;

    local_error = sub_group_scan_inclusive<cl_ulong, work_group_op::max>(device, context, queue, n_elems);
    CHECK_ERROR(local_error)
    error |= local_error;

    if(error != CL_SUCCESS)
        return -1;
    return CL_SUCCESS;
}

#endif // TEST_CONFORMANCE_CLCPP_SUBGROUPS_TEST_SG_SCAN_INCLUSIVE_HPP
