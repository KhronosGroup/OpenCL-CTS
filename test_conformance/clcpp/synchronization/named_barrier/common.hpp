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
#ifndef TEST_CONFORMANCE_CLCPP_SYNCHRONIZATION_NAMED_BARRIER_COMMON_HPP
#define TEST_CONFORMANCE_CLCPP_SYNCHRONIZATION_NAMED_BARRIER_COMMON_HPP

#include <vector>

// Common for all OpenCL C++ tests
#include "../../common.hpp"
#include "../../funcs_test_utils.hpp"

#define RUN_WG_NAMED_BARRIER_TEST_MACRO(TEST_CLASS) \
    last_error = run_work_group_named_barrier_barrier_test(  \
        device, context, queue, num_elements, TEST_CLASS \
    );  \
    CHECK_ERROR(last_error) \
    error |= last_error;

namespace named_barrier {

struct work_group_named_barrier_test_base : public detail::base_func_type<cl_uint>
{
    // Returns test name
    virtual std::string str() = 0;
    // Returns OpenCL program source
    // It's assumed that this program has only one kernel.
    virtual std::string generate_program() = 0;
    // Return value that is expected to be in output_buffer[i]
    virtual cl_uint operator()(size_t i, size_t work_group_size, size_t mas_sub_group_size) = 0;
    // Kernel execution
    // This covers typical case: kernel is executed once, kernel
    // has only one argument which is output buffer
    virtual cl_int execute(const cl_kernel kernel,
                           const cl_mem output_buffer,
                           const cl_command_queue& queue,
                           const size_t work_size,
                           const size_t work_group_size)
    {
        cl_int err;
        err = clSetKernelArg(kernel, 0, sizeof(output_buffer), &output_buffer);
        RETURN_ON_CL_ERROR(err, "clSetKernelArg")

        err = clEnqueueNDRangeKernel(
            queue, kernel, 1,
            NULL, &work_size, &work_group_size,
            0, NULL, NULL
        );
        RETURN_ON_CL_ERROR(err, "clEnqueueNDRangeKernel")
        return err;
    }
    // Calculates maximal work-group size (one dim)
    virtual size_t get_max_local_size(const cl_kernel kernel,
                                      const cl_device_id device,
                                      const size_t work_group_size, // default work-group size
                                      cl_int& error)
    {
        size_t max_wg_size;
        error = clGetKernelWorkGroupInfo(
            kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &max_wg_size, NULL
        );
        RETURN_ON_ERROR(error)
        return (std::min)(work_group_size, max_wg_size);
    }
    // if work-groups should be uniform
    virtual bool enforce_uniform()
    {
        return false;
    }
};

template <class work_group_named_barrier_test>
int run_work_group_named_barrier_barrier_test(cl_device_id device, cl_context context, cl_command_queue queue,
                                              size_t count, work_group_named_barrier_test test)
{
    cl_mem buffers[1];
    cl_program program;
    cl_kernel kernel;
    size_t work_group_size;
    size_t work_size[1];
    cl_int err;

    std::string code_str = test.generate_program();
    std::string kernel_name = test.get_kernel_name();
// -----------------------------------------------------------------------------------
// ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
// -----------------------------------------------------------------------------------
// Only OpenCL C++ to SPIR-V compilation
#if defined(DEVELOPMENT) && defined(ONLY_SPIRV_COMPILATION)
    err = create_opencl_kernel(context, &program, &kernel, code_str, kernel_name);
    return err;
// Use OpenCL C kernels instead of OpenCL C++ kernels (test C++ host code)
#elif defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
    err = create_opencl_kernel(context, &program, &kernel, code_str, kernel_name, "-cl-std=CL2.0", false);
    RETURN_ON_ERROR(err)
#else
    err = create_opencl_kernel(context, &program, &kernel, code_str, kernel_name);
    RETURN_ON_ERROR(err)
#endif

    // Find the max possible wg size for among all the kernels
    work_group_size = test.get_max_local_size(kernel, device, 256, err);
    RETURN_ON_ERROR(err);
    if(work_group_size == 0)
    {
        log_info("SKIPPED: Can't produce local size with enough sub-groups. Skipping tests.\n");
        return CL_SUCCESS;
    }

    work_size[0] = count;
    // uniform work-group
    if(test.enforce_uniform())
    {
        size_t wg_number = static_cast<size_t>(
            std::ceil(static_cast<double>(work_size[0]) / work_group_size)
        );
        work_size[0] = wg_number * work_group_size;
    }

    // host output vector
    std::vector<cl_uint> output = generate_output<cl_uint>(work_size[0], 9999);

    // device output buffer
    buffers[0] = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE), sizeof(cl_uint) * output.size(), NULL, &err);
    RETURN_ON_CL_ERROR(err, "clCreateBuffer")

    // Execute test kernels
    err = test.execute(kernel, buffers[0], queue, work_size[0], work_group_size);
    RETURN_ON_ERROR(err)

    err = clEnqueueReadBuffer(
        queue, buffers[0], CL_TRUE, 0, sizeof(cl_uint) * output.size(),
        static_cast<void *>(output.data()), 0, NULL, NULL
    );
    RETURN_ON_CL_ERROR(err, "clEnqueueReadBuffer")

    // Check output values
    for(size_t i = 0; i < output.size(); i++)
    {
        cl_uint v = test(i, work_group_size, i);
        if(!(are_equal(v, output[i], ::detail::make_value<cl_uint>(0), test)))
        {
            RETURN_ON_ERROR_MSG(-1,
                "test_%s(%s) failed. Expected: %s, got: %s", test.str().c_str(), type_name<cl_uint>().c_str(),
                format_value(v).c_str(), format_value(output[i]).c_str()
            );
        }
    }
    log_info("test_%s(%s) passed\n", test.str().c_str(), type_name<cl_uint>().c_str());

    clReleaseMemObject(buffers[0]);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    return err;
}

} // namespace named_barrier

#endif // TEST_CONFORMANCE_CLCPP_SYNCHRONIZATION_NAMED_BARRIER_COMMON_HPP
