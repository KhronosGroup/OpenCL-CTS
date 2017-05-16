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
#ifndef TEST_CONFORMANCE_CLCPP_SPEC_CONSTANTS_COMMON_HPP
#define TEST_CONFORMANCE_CLCPP_SPEC_CONSTANTS_COMMON_HPP

#include <algorithm>

#include "../common.hpp"
#include "../funcs_test_utils.hpp"

#define RUN_SPEC_CONSTANTS_TEST_MACRO(TEST_CLASS) \
    last_error = run_spec_constants_test(  \
        device, context, queue, n_elems, TEST_CLASS \
    );  \
    CHECK_ERROR(last_error) \
    error |= last_error;

// Base class for all tests of cl::spec_contatnt
template <class T>
struct spec_constants_test : public detail::base_func_type<T>
{
    // Output buffer type
    typedef T type;

    virtual ~spec_constants_test() {};
    // Returns test name
    virtual std::string str() = 0;
    // Returns OpenCL program source
    virtual std::string generate_program() = 0;

    // Return names of test's kernels, in order.
    // Typical case: one kernel.
    virtual std::vector<std::string> get_kernel_names()
    {
        // Typical case, that is, only one kernel
        return { this->get_kernel_name() };
    }

    // If local size has to be set in clEnqueueNDRangeKernel()
    // this should return true; otherwise - false;
    virtual bool set_local_size()
    {
        return false;
    }

    // Calculates maximal work-group size (one dim)
    virtual size_t get_max_local_size(const std::vector<cl_kernel>& kernels,
                                      cl_device_id device,
                                      size_t work_group_size, // default work-group size
                                      cl_int& error)
    {
        size_t wg_size = work_group_size;
        for(auto& k : kernels)
        {
            size_t max_wg_size;
            error = clGetKernelWorkGroupInfo(
                k, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &max_wg_size, NULL
            );
            RETURN_ON_CL_ERROR(error, "clGetKernelWorkGroupInfo")
            wg_size = (std::min)(wg_size, max_wg_size);
        }
        return wg_size;
    }

    // Sets spec constants
    // Typical case: no spec constants to set
    virtual cl_int set_spec_constants(const cl_program& program)
    {
        return CL_SUCCESS;
    }

    // This covers typical case:
    // 1. each kernel is executed once,
    // 2. the only argument in every kernel is output_buffer
    virtual cl_int execute(const std::vector<cl_kernel>& kernels,
                           cl_mem& output_buffer,
                           cl_command_queue& queue,
                           size_t work_size,
                           size_t work_group_size)
    {
        cl_int err;
        for(auto& k : kernels)
        {
            err = clSetKernelArg(k, 0, sizeof(output_buffer), &output_buffer);
            RETURN_ON_CL_ERROR(err, "clSetKernelArg");

            err = clEnqueueNDRangeKernel(
                queue, k, 1,
                NULL, &work_size, this->set_local_size() ? &work_group_size : NULL,
                0, NULL, NULL
            );
            RETURN_ON_CL_ERROR(err, "clEnqueueNDRangeKernel");
        }
        return err;
    }

    // This is a function which performs additional queries and checks
    // if the results are correct. This method is run after checking that
    // test results (output values) are correct.
    virtual cl_int check_queries(const std::vector<cl_kernel>& kernels,
                                 cl_device_id device,
                                 cl_context context,
                                 cl_command_queue queue)
    {
        (void) kernels;
        (void) device;
        (void) context;
        (void) queue;
        return CL_SUCCESS;
    }
};

template <class spec_constants_test>
int run_spec_constants_test(cl_device_id device, cl_context context, cl_command_queue queue, size_t count, spec_constants_test op)
{
    cl_mem buffers[1];
    cl_program program;
    std::vector<cl_kernel> kernels;
    size_t wg_size;
    size_t work_size[1];
    cl_int err;

    typedef typename spec_constants_test::type TYPE;

    // Don't run test for unsupported types
    if(!(type_supported<TYPE>(device)))
    {
        return CL_SUCCESS;
    }

    std::string code_str = op.generate_program();
    std::vector<std::string> kernel_names = op.get_kernel_names();
    if(kernel_names.empty())
    {
        RETURN_ON_ERROR_MSG(-1, "No kernel to run");
    }
    kernels.resize(kernel_names.size());

    std::string options = "";
    if(is_extension_available(device, "cl_khr_fp16"))
    {
        options += " -cl-fp16-enable";
    }
    if(is_extension_available(device, "cl_khr_fp64"))
    {
        options += " -cl-fp64-enable";
    }
// -----------------------------------------------------------------------------------
// ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
// -----------------------------------------------------------------------------------
// Only OpenCL C++ to SPIR-V compilation
#if defined(DEVELOPMENT) && defined(ONLY_SPIRV_COMPILATION)
    err = create_opencl_kernel(context, &program, &(kernels[0]), code_str, kernel_names[0], options);
    return err;
// Use OpenCL C kernels instead of OpenCL C++ kernels (test C++ host code)
#elif defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
    err = create_opencl_kernel(context, &program, &(kernels[0]), code_str, kernel_names[0], "-cl-std=CL2.0", false);
    RETURN_ON_ERROR(err)
    for(size_t i = 1; i < kernels.size(); i++)
    {
        kernels[i] = clCreateKernel(program, kernel_names[i].c_str(), &err);
        RETURN_ON_CL_ERROR(err, "clCreateKernel");
    }
#else
    const char * code_c_str = code_str.c_str();
    err = create_openclcpp_program(context, &program, 1, &(code_c_str), options.c_str());
    RETURN_ON_ERROR_MSG(err, "Creating OpenCL C++ program failed")

    // Set spec constants
    err = op.set_spec_constants(program);
    RETURN_ON_ERROR_MSG(err, "Setting Spec Constants failed")

    // Build program and create 1st kernel
    err = build_program_create_kernel_helper(
        context, &program, &(kernels[0]), 1, &(code_c_str), kernel_names[0].c_str()
    );
    RETURN_ON_ERROR_MSG(err, "Unable to build program or to create kernel")
    // Create other kernels
    for(size_t i = 1; i < kernels.size(); i++)
    {
        kernels[i] = clCreateKernel(program, kernel_names[i].c_str(), &err);
        RETURN_ON_CL_ERROR(err, "clCreateKernel");
    }
#endif

    // Find the max possible wg size for among all the kernels
    wg_size = op.get_max_local_size(kernels, device, 1024, err);
    RETURN_ON_ERROR(err);

    work_size[0] = count;
    if(op.set_local_size())
    {
        size_t wg_number = static_cast<size_t>(
            std::ceil(static_cast<double>(count) / wg_size)
        );
        work_size[0] = wg_number * wg_size;
    }

    // host output vector
    std::vector<TYPE> output = generate_output<TYPE>(work_size[0], 9999);

    // device output buffer
    buffers[0] = clCreateBuffer(
        context, (cl_mem_flags)(CL_MEM_READ_WRITE), sizeof(TYPE) * output.size(), NULL, &err
    );
    RETURN_ON_CL_ERROR(err, "clCreateBuffer");

    // Execute test
    err = op.execute(kernels, buffers[0], queue, work_size[0], wg_size);
    RETURN_ON_ERROR(err)

    err = clEnqueueReadBuffer(
        queue, buffers[0], CL_TRUE, 0, sizeof(TYPE) * output.size(),
        static_cast<void *>(output.data()), 0, NULL, NULL
    );
    RETURN_ON_CL_ERROR(err, "clEnqueueReadBuffer");

    // Check output values
    for(size_t i = 0; i < output.size(); i++)
    {
        TYPE v = op(i, wg_size);
        if(!(are_equal(v, output[i], detail::make_value<TYPE>(0), op)))
        {
            RETURN_ON_ERROR_MSG(-1,
                "test_%s(%s) failed. Expected: %s, got: %s", op.str().c_str(), type_name<cl_uint>().c_str(),
                format_value(v).c_str(), format_value(output[i]).c_str()
            );
        }
    }

    // Check if queries returns correct values
    err = op.check_queries(kernels, device, context, queue);
    RETURN_ON_ERROR(err);

    log_info("test_%s(%s) passed\n", op.str().c_str(), type_name<TYPE>().c_str());

    clReleaseMemObject(buffers[0]);
    for(auto& k : kernels)
        clReleaseKernel(k);
    clReleaseProgram(program);
    return err;
}

#endif // TEST_CONFORMANCE_CLCPP_SPEC_CONSTANTS_COMMON_HPP
