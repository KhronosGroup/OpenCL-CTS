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
#ifndef TEST_CONFORMANCE_CLCPP_PS_CTORS_DTORS_COMMON_HPP
#define TEST_CONFORMANCE_CLCPP_PS_CTORS_DTORS_COMMON_HPP

#include <algorithm>

#include "../common.hpp"
#include "../funcs_test_utils.hpp"

#define RUN_PS_CTORS_DTORS_TEST_MACRO(TEST_CLASS) \
    last_error = run_ps_ctor_dtor_test(  \
        device, context, queue, count, TEST_CLASS \
    );  \
    CHECK_ERROR(last_error) \
    error |= last_error;

// Base class for all tests for kernels with program scope object with
// non-trivial ctors and/or dtors
struct ps_ctors_dtors_test_base : public detail::base_func_type<cl_uint>
{
    // ctor is true, if and only if OpenCL program of this test contains program
    // scope variable with non-trivial ctor.
    // dtor is true, if and only if OpenCL program of this test contains program
    // scope variable with non-trivial dtor.
    ps_ctors_dtors_test_base(const bool ctor,
                             const bool dtor)
        : m_ctor(ctor), m_dtor(dtor)
    {

    }
    virtual ~ps_ctors_dtors_test_base() { };
    // Returns test name
    virtual std::string str() = 0;
    // Returns OpenCL program source
    virtual std::string generate_program() = 0;
    // Returns kernel names IN ORDER
    virtual std::vector<std::string> get_kernel_names()
    {
        // Typical case, that is, only one kernel
        return { this->get_kernel_name() };
    }
    // Returns value that is expected to be in output_buffer[i]
    virtual cl_uint operator()(size_t i) = 0;
    // Executes kernels
    // Typical case: execute every kernel once, every kernel has only
    // one argument, that is, output buffer
    virtual cl_int execute(const std::vector<cl_kernel>& kernels,
                           cl_mem& output_buffer,
                           cl_command_queue& queue,
                           size_t work_size)
    {
        cl_int err;
        for(auto& k : kernels)
        {
            err = clSetKernelArg(k, 0, sizeof(output_buffer), &output_buffer);
            RETURN_ON_CL_ERROR(err, "clSetKernelArg");

            err = clEnqueueNDRangeKernel(
                queue, k, 1,
                NULL, &work_size, NULL,
                0, NULL, NULL
            );
            RETURN_ON_CL_ERROR(err, "clEnqueueNDRangeKernel");
        }
        return err;
    }
    // This method check if queries for CL_PROGRAM_SCOPE_GLOBAL_CTORS_PRESENT
    // and CL_PROGRAM_SCOPE_GLOBAL_DTORS_PRESENT using clGetProgramInfo()
    // return correct values
    virtual cl_int ctors_dtors_present_queries(cl_program program)
    {
        cl_int error = CL_SUCCESS;
        #if defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
            return error;
        #else
            // CL_PROGRAM_SCOPE_GLOBAL_CTORS_PRESENT cl_bool
            // This indicates that the program object contains non-trivial constructor(s) that will be
            // executed by runtime before any kernel from the program is executed.

            // CL_PROGRAM_SCOPE_GLOBAL_DTORS_PRESENT cl_bool
            // This indicates that the program object contains non-trivial destructor(s) that will be
            // executed by runtime when program is destroyed.

            // CL_PROGRAM_SCOPE_GLOBAL_CTORS_PRESENT
            cl_bool ctors_present;
            size_t cl_bool_size;
            error = clGetProgramInfo(
                program,
                CL_PROGRAM_SCOPE_GLOBAL_CTORS_PRESENT,
                sizeof(cl_bool),
                static_cast<void*>(&ctors_present),
                &cl_bool_size
            );
            RETURN_ON_CL_ERROR(error, "clGetProgramInfo")
            if(cl_bool_size != sizeof(cl_bool))
            {
                error = -1;
                CHECK_ERROR_MSG(
                    error,
                    "Test failed, param_value_size_ret != sizeof(cl_bool) (%lu != %lu).\n",
                    cl_bool_size,
                    sizeof(cl_bool)
                );
            }

            // CL_PROGRAM_SCOPE_GLOBAL_DTORS_PRESENT
            cl_bool dtors_present = 0;
            error = clGetProgramInfo(
                program,
                CL_PROGRAM_SCOPE_GLOBAL_DTORS_PRESENT,
                sizeof(cl_bool),
                static_cast<void*>(&ctors_present),
                &cl_bool_size
            );
            RETURN_ON_CL_ERROR(error, "clGetProgramInfo")
            if(cl_bool_size != sizeof(cl_bool))
            {
                error = -1;
                CHECK_ERROR_MSG(
                    error,
                    "Test failed, param_value_size_ret != sizeof(cl_bool) (%lu != %lu).\n",
                    cl_bool_size,
                    sizeof(cl_bool)
                );
            }

            // check constructors
            if(m_ctor && ctors_present != CL_TRUE)
            {
                error = -1;
                CHECK_ERROR_MSG(
                    error,
                    "Test failed, CL_PROGRAM_SCOPE_GLOBAL_CTORS_PRESENT: 0, should be: 1.\n"
                );
            }
            else if(!m_ctor && ctors_present == CL_TRUE)
            {
                error = -1;
                CHECK_ERROR_MSG(
                    error,
                    "Test failed, CL_PROGRAM_SCOPE_GLOBAL_CTORS_PRESENT: 1, should be: 0.\n"
                );
            }

            // check destructors
            if(m_dtor && dtors_present != CL_TRUE)
            {
                error = -1;
                CHECK_ERROR_MSG(
                    error,
                    "Test failed, CL_PROGRAM_SCOPE_GLOBAL_DTORS_PRESENT: 0, should be: 1.\n"
                );
            }
            else if(!m_dtor && dtors_present == CL_TRUE)
            {
                error = -1;
                CHECK_ERROR_MSG(
                    error,
                    "Test failed, CL_PROGRAM_SCOPE_GLOBAL_DTORS_PRESENT: 1, should be: 0.\n"
                );
            }
            return error;
        #endif
    }

private:
    bool m_ctor;
    bool m_dtor;
};

template <class ps_ctor_dtor_test>
int run_ps_ctor_dtor_test(cl_device_id device, cl_context context, cl_command_queue queue, size_t count, ps_ctor_dtor_test op)
{
    cl_mem buffers[1];
    cl_program program;
    std::vector<cl_kernel> kernels;
    size_t work_size[1];
    cl_int err;

    std::string code_str = op.generate_program();
    std::vector<std::string> kernel_names = op.get_kernel_names();
    if(kernel_names.empty())
    {
        RETURN_ON_ERROR_MSG(-1, "No kernel to run");
    }
    kernels.resize(kernel_names.size());

// -----------------------------------------------------------------------------------
// ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
// -----------------------------------------------------------------------------------
// Only OpenCL C++ to SPIR-V compilation
#if defined(DEVELOPMENT) && defined(ONLY_SPIRV_COMPILATION)
    err = create_opencl_kernel(context, &program, &(kernels[0]), code_str, kernel_names[0]);
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
    err = create_opencl_kernel(context, &program, &(kernels[0]), code_str, kernel_names[0]);
    RETURN_ON_ERROR(err)
    for(size_t i = 1; i < kernels.size(); i++)
    {
        kernels[i] = clCreateKernel(program, kernel_names[i].c_str(), &err);
        RETURN_ON_CL_ERROR(err, "clCreateKernel");
    }
#endif

    work_size[0] = count;
    // host output vector
    std::vector<cl_uint> output = generate_output<cl_uint>(work_size[0], 9999);

    // device output buffer
    buffers[0] = clCreateBuffer(
        context, (cl_mem_flags)(CL_MEM_READ_WRITE), sizeof(cl_uint) * output.size(), NULL, &err
    );
    RETURN_ON_CL_ERROR(err, "clCreateBuffer")

    // Execute test
    err = op.execute(kernels, buffers[0], queue, work_size[0]);
    RETURN_ON_ERROR(err)

    // Check if queries returns correct values
    err = op.ctors_dtors_present_queries(program);
    RETURN_ON_ERROR(err);

    // Release kernels and program
    // Destructors should be called now
    for(auto& k : kernels)
    {
        err = clReleaseKernel(k);
        RETURN_ON_CL_ERROR(err, "clReleaseKernel");
    }
    err = clReleaseProgram(program);
    RETURN_ON_CL_ERROR(err, "clReleaseProgram");

    // Finish
    err = clFinish(queue);
    RETURN_ON_CL_ERROR(err, "clFinish");

    err = clEnqueueReadBuffer(
        queue, buffers[0], CL_TRUE, 0, sizeof(cl_uint) * output.size(),
        static_cast<void *>(output.data()), 0, NULL, NULL
    );
    RETURN_ON_CL_ERROR(err, "clEnqueueReadBuffer");

    // Check output values
    for(size_t i = 0; i < output.size(); i++)
    {
        cl_uint v = op(i);
        if(!(are_equal(v, output[i], detail::make_value<cl_uint>(0), op)))
        {
            RETURN_ON_ERROR_MSG(-1,
                "test_%s(%s) failed. Expected: %s, got: %s", op.str().c_str(), type_name<cl_uint>().c_str(),
                format_value(v).c_str(), format_value(output[i]).c_str()
            );
        }
    }
    log_info("test_%s(%s) passed\n", op.str().c_str(), type_name<cl_uint>().c_str());

    clReleaseMemObject(buffers[0]);
    return err;
}

#endif // TEST_CONFORMANCE_CLCPP_PS_CTORS_DTORS_COMMON_HPP
