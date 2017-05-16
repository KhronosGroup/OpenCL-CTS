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
#ifndef TEST_CONFORMANCE_CLCPP_API_TEST_CTORS_HPP
#define TEST_CONFORMANCE_CLCPP_API_TEST_CTORS_HPP

#include <vector>
#include <limits>
#include <algorithm>
#include <numeric>

#include "../common.hpp"

// TEST 1
// Verify that constructors are executed before any kernel is executed.
// Verify that when present, multiple constructors are executed. The order between
// constructors is undefined, but they should all execute.

// -----------------------------------------------------------------------------------
// ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
// -----------------------------------------------------------------------------------
#if defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
const char * kernel_test_ctors_executed =
    "__kernel void test_ctors_executed(global uint *output)\n"
    "{\n"
    "   ulong gid = get_global_id(0);\n"
    "   output[gid] = 0;\n"
    "}\n"
;
const char * kernel_test_ctors_executed_multiple_ctors =
    "__kernel void test_ctors_executed_multiple_ctors(global uint *output)\n"
    "{\n"
    "   ulong gid = get_global_id(0);\n"
    "   output[gid] = 0;\n"
    "}\n"
;
#else
const char * kernel_test_ctors_executed =
    "#include <opencl_memory>\n"
    "#include <opencl_work_item>\n"
    "using namespace cl;\n"
    "struct ctor_test_class {\n"
    // non-trivial ctor
    "   ctor_test_class(int y) { x = y;};\n"
    "   int x;\n"
    "};\n"
    // global scope program variable
    "ctor_test_class global_var(int(0xbeefbeef));\n"
    "__kernel void test_ctors_executed(global_ptr<uint[]> output)\n"
    "{\n"
    "   ulong gid = get_global_id(0);\n"
    "   int result = 0;\n"
    "   if(global_var.x != int(0xbeefbeef)) result = 1;\n"
    "   output[gid] = result;\n"
    "}\n"
;
const char * kernel_test_ctors_executed_multiple_ctors =
    "#include <opencl_memory>\n"
    "#include <opencl_work_item>\n"
    "#include <opencl_limits>\n"
    "using namespace cl;\n"
    "template<class T>\n"
    "struct ctor_test_class {\n"
    // non-trivial ctor
    "   ctor_test_class(T y) { x = y;};\n"
    "   T x;\n"
    "};\n"
    // global scope program variables
    "ctor_test_class<int> global_var0(int(0xbeefbeef));\n"
    "ctor_test_class<uint> global_var1(uint(0xbeefbeefU));\n"
    "ctor_test_class<float> global_var2(float(FLT_MAX));\n"
    "__kernel void test_ctors_executed_multiple_ctors(global_ptr<uint[]> output)\n"
    "{\n"
    "   ulong gid = get_global_id(0);\n"
    "   int result = 0;\n"
    "   if(global_var0.x != int(0xbeefbeef))   result = 1;\n"
    "   if(global_var1.x != uint(0xbeefbeefU)) result = 1;\n"
    "   if(global_var2.x != float(FLT_MAX))    result = 1;\n"
    "   output[gid] = result;\n"
    "}\n"
;
#endif

int test_ctors_execution(cl_device_id device,
                         cl_context context,
                         cl_command_queue queue,
                         int count,
                         std::string kernel_name,
                         const char * kernel_source)
{
    int error = CL_SUCCESS;

    cl_mem output_buffer;
    cl_program program;
    cl_kernel kernel;

    size_t dim = 1;
    size_t work_size[1];
// -----------------------------------------------------------------------------------
// ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
// -----------------------------------------------------------------------------------
// Only OpenCL C++ to SPIR-V compilation
#if defined(DEVELOPMENT) && defined(ONLY_SPIRV_COMPILATION)
    error = create_opencl_kernel(context, &program, &kernel, kernel_source, kernel_name);
    RETURN_ON_ERROR(error)
    return error;
// Use OpenCL C kernels instead of OpenCL C++ kernels (test C++ host code)
#elif defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
    error = create_opencl_kernel(context, &program, &kernel, kernel_source, kernel_name, "", false);
    RETURN_ON_ERROR(error)
// Normal run
#else
    error = create_opencl_kernel(context, &program, &kernel, kernel_source, kernel_name);
    RETURN_ON_ERROR(error)
#endif

    // host vector, size == count, output[0...count-1] == 1
    std::vector<cl_uint> output(count, cl_uint(1));
    output_buffer = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE), sizeof(cl_uint) * output.size(), NULL, &error);
    RETURN_ON_CL_ERROR(error, "clCreateBuffer")

    error = clEnqueueWriteBuffer(queue, output_buffer, CL_TRUE, 0, sizeof(cl_uint) * output.size(), static_cast<void *>(output.data()), 0, NULL, NULL);
    RETURN_ON_CL_ERROR(error, "clEnqueueWriteBuffer")

    error = clSetKernelArg(kernel, 0, sizeof(output_buffer), &output_buffer);
    RETURN_ON_CL_ERROR(error, "clSetKernelArg")

    work_size[0] = output.size();
    error = clEnqueueNDRangeKernel(queue, kernel, dim, NULL, work_size, NULL, 0, NULL, NULL);
    RETURN_ON_CL_ERROR(error, "clEnqueueNDRangeKernel")

    error = clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, sizeof(cl_uint) * output.size(), static_cast<void *>(output.data()), 0, NULL, NULL);
    RETURN_ON_CL_ERROR(error, "clEnqueueReadBuffer")

    size_t sum = std::accumulate(output.begin(), output.end(), size_t(0));
    if(sum != 0)
    {
        error = -1;
        CHECK_ERROR_MSG(error, "Test %s failed.", kernel_name.c_str());
    }

    clReleaseMemObject(output_buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    return error;
}

AUTO_TEST_CASE(test_global_scope_ctors_executed)
(cl_device_id device, cl_context context, cl_command_queue queue, int count)
{
    int error = CL_SUCCESS;
    int local_error = CL_SUCCESS;

    local_error = test_ctors_execution(
        device, context, queue, count,
        "test_ctors_executed", kernel_test_ctors_executed
    );
    CHECK_ERROR(local_error);
    error |= local_error;

    local_error = test_ctors_execution(
        device, context, queue, count,
        "test_ctors_executed_multiple_ctors", kernel_test_ctors_executed_multiple_ctors
    );
    CHECK_ERROR(local_error);
    error |= local_error;

    if(error != CL_SUCCESS)
    {
        return -1;
    }
    return error;
}

// TEST 2
// Verify that constructors are only executed once when multiple kernels from a program are executed.

// How: The first kernel (test_ctors_executed_once_set) is run once. It changes values of program scope
// variables, then the second kernel is run multiple times, each time verifying that global variables
// have correct values (the second kernel should observe the values assigned by the first kernel, not
// by the constructors).

// -----------------------------------------------------------------------------------
// ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
// -----------------------------------------------------------------------------------
#if defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
const char * program_test_ctors_executed_once =
    "__kernel void test_ctors_executed_once_set()\n"
    "{\n"
    "}\n"
    "__kernel void test_ctors_executed_once_read(global uint *output)\n"
    "{\n"
    "   ulong gid = get_global_id(0);\n"
    "   output[gid] = 0;\n"
    "}\n"
;
#else
const char * program_test_ctors_executed_once =
    "#include <opencl_memory>\n"
    "#include <opencl_work_item>\n"
    "using namespace cl;\n"
    // struct template
    "template<class T>\n"
    "struct ctor_test_class {\n"
    // non-trivial ctor
    "   ctor_test_class(T y) { x = y;};\n"
    "   T x;\n"
    "};\n"
    // global scope program variables
    "ctor_test_class<int> global_var0(int(0));\n"
    "ctor_test_class<uint> global_var1(uint(0));\n"

    "__kernel void test_ctors_executed_once_set()\n"
    "{\n"
    "   ulong gid = get_global_id(0);\n"
    "   if(gid == 0) {\n"
    "       global_var0.x = int(0xbeefbeef);\n"
    "       global_var1.x = uint(0xbeefbeefU);\n"
    "   }\n"
    "}\n\n"

    "__kernel void test_ctors_executed_once_read(global_ptr<uint[]> output)\n"
    "{\n"
    "   ulong gid = get_global_id(0);\n"
    "   int result = 0;\n"
    "   if(global_var0.x != int(0xbeefbeef))   result = 1;\n"
    "   if(global_var1.x != uint(0xbeefbeefU)) result = 1;\n"
    "   output[gid] = result;\n"
    "}\n"
;
#endif

AUTO_TEST_CASE(test_global_scope_ctors_executed_once)
(cl_device_id device, cl_context context, cl_command_queue queue, int count)
{
    int error = CL_SUCCESS;

    cl_mem output_buffer;
    cl_program program;
    cl_kernel kernel_set_global_vars;
    cl_kernel kernel_read_global_vars;

    size_t dim = 1;
    size_t work_size[1];
// -----------------------------------------------------------------------------------
// ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
// -----------------------------------------------------------------------------------
// Only OpenCL C++ to SPIR-V compilation
#if defined(DEVELOPMENT) && defined(ONLY_SPIRV_COMPILATION)
    error = create_opencl_kernel(
        context, &program, &kernel_set_global_vars,
        program_test_ctors_executed_once, "test_ctors_executed_once_set"
    );
    RETURN_ON_ERROR(error)
    return error;
// Use OpenCL C kernels instead of OpenCL C++ kernels (test C++ host code)
#elif defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
    error = create_opencl_kernel(
        context, &program, &kernel_set_global_vars,
        program_test_ctors_executed_once, "test_ctors_executed_once_set", "", false
    );
    RETURN_ON_ERROR(error)
    // Get the second kernel
    kernel_read_global_vars = clCreateKernel(program, "test_ctors_executed_once_read", &error);
    RETURN_ON_CL_ERROR(error, "clCreateKernel");
// Normal run
#else
    error = create_opencl_kernel(
        context, &program, &kernel_set_global_vars,
        program_test_ctors_executed_once, "test_ctors_executed_once_set"
    );
    RETURN_ON_ERROR(error)
    // Get the second kernel
    kernel_read_global_vars = clCreateKernel(program, "test_ctors_executed_once_read", &error);
    RETURN_ON_CL_ERROR(error, "clCreateKernel");
#endif

    // Execute kernel_set_global_vars

    work_size[0] = count;
    error = clEnqueueNDRangeKernel(queue, kernel_set_global_vars, dim, NULL, work_size, NULL, 0, NULL, NULL);
    RETURN_ON_CL_ERROR(error, "clEnqueueNDRangeKernel")

    // Execute kernel_read_global_vars 4 times, each time we check if
    // global variables have correct values.

    // host vector, size == count, output[0...count-1] == 1
    std::vector<cl_uint> output(count, cl_uint(1));
    output_buffer = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE), sizeof(cl_uint) * output.size(), NULL, &error);
    RETURN_ON_CL_ERROR(error, "clCreateBuffer")

    for(size_t i = 0; i < 4; i++)
    {
        std::fill(output.begin(), output.end(), cl_uint(1));
        error = clEnqueueWriteBuffer(
            queue, output_buffer, CL_TRUE,
            0, sizeof(cl_uint) * output.size(),
            static_cast<void *>(output.data()),
            0, NULL, NULL
        );
        RETURN_ON_CL_ERROR(error, "clEnqueueWriteBuffer")

        error = clSetKernelArg(kernel_read_global_vars, 0, sizeof(output_buffer), &output_buffer);
        RETURN_ON_CL_ERROR(error, "clSetKernelArg")

        work_size[0] = output.size();
        error = clEnqueueNDRangeKernel(
            queue, kernel_read_global_vars,
            dim, NULL, work_size, NULL,
            0, NULL, NULL
        );
        RETURN_ON_CL_ERROR(error, "clEnqueueNDRangeKernel")

        error = clEnqueueReadBuffer(
            queue, output_buffer, CL_TRUE,
            0, sizeof(cl_uint) * output.size(),
            static_cast<void *>(output.data()),
            0, NULL, NULL
        );
        RETURN_ON_CL_ERROR(error, "clEnqueueReadBuffer")

        size_t sum = std::accumulate(output.begin(), output.end(), size_t(0));
        if(sum != 0)
        {
            error = -1;
            CHECK_ERROR_MSG(error, "Test test_ctors_executed_onces failed.");
        }
    }

    clReleaseMemObject(output_buffer);
    clReleaseKernel(kernel_set_global_vars);
    clReleaseKernel(kernel_read_global_vars);
    clReleaseProgram(program);
    return error;
}

// TEST3
// Verify that when constructor is executed, the ND-range used is (1,1,1).

// -----------------------------------------------------------------------------------
// ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
// -----------------------------------------------------------------------------------
#if defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
const char * program_test_ctors_ndrange =
    "__kernel void test_ctors_ndrange(global int *output)\n"
    "{\n"
    "   ulong gid = get_global_id(0);\n"
    "   output[gid] = 0;\n"
    "}\n"
;
#else
const char * program_test_ctors_ndrange =
    "#include <opencl_memory>\n"
    "#include <opencl_work_item>\n"
    "using namespace cl;\n"
    // struct
    "struct ctor_test_class {\n"
    // non-trivial ctor
    "   ctor_test_class() {\n"
    "       x = get_global_size(0);\n"
    "       y = get_global_size(1);\n"
    "       z = get_global_size(2);\n"
    "   };\n"
    "   ulong x;\n"
    "   ulong y;\n"
    "   ulong z;\n"
    // return true if the ND-range used when ctor was exectured was
    // (1, 1, 1); otherwise - false
    "   bool check() { return (x == 1) && (y == 1) && (z == 1);}"
    "};\n"
    // global scope program variables
    "ctor_test_class global_var0;\n"
    "ctor_test_class global_var1;\n"

    "__kernel void test_ctors_ndrange(global_ptr<uint[]> output)\n"
    "{\n"
    "   ulong gid = get_global_id(0);\n"
    "   int result = 0;\n"
    "   if(!global_var0.check()) result = 1;\n"
    "   if(!global_var1.check()) result = 1;\n"
    "   output[gid] = result;\n"
    "}\n"
;
#endif

AUTO_TEST_CASE(test_global_scope_ctors_ndrange)
(cl_device_id device, cl_context context, cl_command_queue queue, int count)
{
    int error = CL_SUCCESS;

    cl_mem output_buffer;
    cl_program program;
    cl_kernel kernel;

    size_t dim = 1;
    size_t work_size[1];
// -----------------------------------------------------------------------------------
// ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
// -----------------------------------------------------------------------------------
// Only OpenCL C++ to SPIR-V compilation
#if defined(DEVELOPMENT) && defined(ONLY_SPIRV_COMPILATION)
    error = create_opencl_kernel(
        context, &program, &kernel,
        program_test_ctors_ndrange, "test_ctors_ndrange"
    );
    RETURN_ON_ERROR(error)
    return error;
// Use OpenCL C kernels instead of OpenCL C++ kernels (test C++ host code)
#elif defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
    error = create_opencl_kernel(
        context, &program, &kernel,
        program_test_ctors_ndrange, "test_ctors_ndrange", "", false
    );
    RETURN_ON_ERROR(error)
// Normal run
#else
    error = create_opencl_kernel(
        context, &program, &kernel,
        program_test_ctors_ndrange, "test_ctors_ndrange"
    );
    RETURN_ON_ERROR(error)
#endif

    // host vector, size == count, output[0...count-1] == 1
    std::vector<cl_uint> output(count, cl_uint(1));
    output_buffer = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE), sizeof(cl_uint) * output.size(), NULL, &error);
    RETURN_ON_CL_ERROR(error, "clCreateBuffer")

    error = clEnqueueWriteBuffer(
        queue, output_buffer, CL_TRUE,
        0, sizeof(cl_uint) * output.size(),
        static_cast<void *>(output.data()),
        0, NULL, NULL
    );
    RETURN_ON_CL_ERROR(error, "clEnqueueWriteBuffer")

    error = clSetKernelArg(kernel, 0, sizeof(output_buffer), &output_buffer);
    RETURN_ON_CL_ERROR(error, "clSetKernelArg")

    work_size[0] = output.size();
    error = clEnqueueNDRangeKernel(
        queue, kernel,
        dim, NULL, work_size, NULL,
        0, NULL, NULL
    );
    RETURN_ON_CL_ERROR(error, "clEnqueueNDRangeKernel")

    error = clEnqueueReadBuffer(
        queue, output_buffer, CL_TRUE,
        0, sizeof(cl_uint) * output.size(),
        static_cast<void *>(output.data()),
        0, NULL, NULL
    );
    RETURN_ON_CL_ERROR(error, "clEnqueueReadBuffer")

    size_t sum = std::accumulate(output.begin(), output.end(), size_t(0));
    if(sum != 0)
    {
        error = -1;
        CHECK_ERROR_MSG(error, "Test test_ctors_executed_ndrange failed.");
    }

    clReleaseMemObject(output_buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    return error;
}

#endif // TEST_CONFORMANCE_CLCPP_API_TEST_CTORS_HPP
