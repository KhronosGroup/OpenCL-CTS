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
#ifndef TEST_CONFORMANCE_CLCPP_API_TEST_DTORS_HPP
#define TEST_CONFORMANCE_CLCPP_API_TEST_DTORS_HPP

#include <vector>
#include <limits>
#include <algorithm>
#include <numeric>

#include "../common.hpp"

// TEST 1
// Verify that destructor is executed.

// How: destructor of struct dtor_test_class has a side effect: zeroing buffer. If values
// in buffer are not zeros after releasing program, destructor was not executed.

// -----------------------------------------------------------------------------------
// ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
// -----------------------------------------------------------------------------------
#if defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
const char * program_test_dtor_is_executed =
    "__kernel void test_dtor_is_executed(global uint *output)\n"
    "{\n"
    "   ulong gid = get_global_id(0);\n"
    "   output[gid] = 0;\n"
    "}\n"
;
#else
const char * program_test_dtor_is_executed =
    "#include <opencl_memory>\n"
    "#include <opencl_work_item>\n"
    "using namespace cl;\n"
    // struct
    "struct dtor_test_class {\n"
    // non-trivial dtor
    // set all values in buffer to 0
    "   ~dtor_test_class() {\n"
    "       for(ulong i = 0; i < size; i++)\n"
    "           buffer[i] = 0;\n"
    "   };\n"
    "   global_ptr<uint[]> buffer;\n"
    "   ulong size;\n"
    "};\n"
    // global scope program variable
    "dtor_test_class global_var;\n"

    // values in output __MUST BE__ greater than 0 for the test to work
    // correctly
    "__kernel void test_dtor_is_executed(global_ptr<uint[]> output)\n"
    "{\n"
    "   ulong gid = get_global_id(0);\n"
    // set buffer and size in global var
    "   if(gid == 0){\n"
    "       global_var.buffer = output;\n"
    "       global_var.size = get_global_size(0);\n"
    "   }\n"
    "}\n"
;
#endif

AUTO_TEST_CASE(test_global_scope_dtor_is_executed)
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
        program_test_dtor_is_executed, "test_dtor_is_executed"
    );
    RETURN_ON_ERROR(error)
    return error;
// Use OpenCL C kernels instead of OpenCL C++ kernels (test C++ host code)
#elif defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
    error = create_opencl_kernel(
        context, &program, &kernel,
        program_test_dtor_is_executed, "test_dtor_is_executed", "", false
    );
    RETURN_ON_ERROR(error)
// Normal run
#else
    error = create_opencl_kernel(
        context, &program, &kernel,
        program_test_dtor_is_executed, "test_dtor_is_executed"
    );
    RETURN_ON_ERROR(error)
#endif

    // host vector, size == count, output[0...count-1] == 0xbeefbeef (3203383023)
    // values in output __MUST BE__ greater than 0 for the test to work correctly
    std::vector<cl_uint> output(count, cl_uint(0xbeefbeef));
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

    // Release kernel and program
    // Dtor should be called now
    error = clReleaseKernel(kernel);
    RETURN_ON_CL_ERROR(error, "clReleaseKernel")
    error = clReleaseProgram(program);
    RETURN_ON_CL_ERROR(error, "clReleaseProgram")

    // Finish
    error = clFinish(queue);
    RETURN_ON_CL_ERROR(error, "clFinish")

    // Read output buffer
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
        CHECK_ERROR_MSG(error, "Test test_dtor_is_executed failed.");
    }

    clReleaseMemObject(output_buffer);
    return error;
}

// TEST 2
// Verify that multiple destructors, if present, are executed. Order between multiple
// destructors is undefined.
// Verify that each destructor is executed only once.

// How:
// 0) dtor_test_class struct has a global pointer to a buffer, it's set by
// test_dtors_executed_once kernel.
// 1) Destructors have a side effect: each dtor writes to its part of the buffer. If all
// dtors are executed, all values in that buffer should be changed.
// 2) The first time destructors are executed, they set their parts of the buffer to zero.
// Next time to 1, next time to 2 etc. Since dtors should be executed only once, all
// values in that buffer should be equal to zero.

// -----------------------------------------------------------------------------------
// ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
// -----------------------------------------------------------------------------------
#if defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
const char * program_test_dtors_executed_once =
    "__kernel void test_dtors_executed_once(global uint *output)\n"
    "{\n"
    "   ulong gid = get_global_id(0);\n"
    "   output[gid] = 0;\n"
    "}\n"
;
#else
const char * program_test_dtors_executed_once =
    "#include <opencl_memory>\n"
    "#include <opencl_work_item>\n"
    "using namespace cl;\n"
    // struct
    "struct dtor_test_class {\n"
    // non-trivial dtor
    // Set all values in range [start; end - 1] in buffer to counter.
    // If dtor is executed only once (correct), all values in range
    // [start; end - 1] in buffer should be equal to zero after releasing
    // the program
    "   ~dtor_test_class() {\n"
    "       for(ulong i = start; i < end; i++){\n"
    "           buffer[i] = counter;\n"
    "       };\n"
    "       counter++;\n"
    "   };\n"
    "   global_ptr<uint[]> buffer;\n"
    "   ulong start;\n"
    "   ulong end;\n"
    "   ulong counter;\n"
    "};\n"
    // global scope program variables
    "dtor_test_class global_var0;\n"
    "dtor_test_class global_var1;\n"
    "dtor_test_class global_var2;\n"
    "dtor_test_class global_var3;\n"

    // values in output __MUST BE__ greater than 0 for the test to work correctly
    "__kernel void test_dtors_executed_once(global_ptr<uint[]> output)\n"
    "{\n"
    "   ulong gid = get_global_id(0);\n"
    // set buffer and size in global var
    "   if(gid == 0){\n"
    "       ulong end = get_global_size(0) / 4;"
    // global_var0
    "       global_var0.buffer = output;\n"
    "       global_var0.start = 0;\n"
    "       global_var0.end = end;\n"
    "       global_var0.counter = 0;\n"
    // global_var1
    "       global_var1.buffer = output;\n"
    "       global_var1.start = end;\n"
    "       end += get_global_size(0) / 4;\n"
    "       global_var1.end = end;\n"
    "       global_var1.counter = 0;\n"
    // global_var2
    "       global_var2.buffer = output;\n"
    "       global_var2.start = end;\n"
    "       end += get_global_size(0) / 4;\n"
    "       global_var2.end = end;\n"
    "       global_var2.counter = 0;\n"
    // global_var3
    "       global_var3.buffer = output;\n"
    "       global_var3.start = end;\n"
    "       global_var3.end = get_global_size(0);\n"
    "       global_var3.counter = 0;\n"
    "   }\n"
    "}\n"
;
#endif

AUTO_TEST_CASE(test_global_scope_dtors_executed_once)
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
        program_test_dtors_executed_once, "test_dtors_executed_once"
    );
    RETURN_ON_ERROR(error)
    return error;
// Use OpenCL C kernels instead of OpenCL C++ kernels (test C++ host code)
#elif defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
    error = create_opencl_kernel(
        context, &program, &kernel,
        program_test_dtors_executed_once, "test_dtors_executed_once", "", false
    );
    RETURN_ON_ERROR(error)
// Normal run
#else
    error = create_opencl_kernel(
        context, &program, &kernel,
        program_test_dtors_executed_once, "test_dtors_executed_once"
    );
    RETURN_ON_ERROR(error)
#endif

    // host vector, size == count, output[0...count-1] == 0xbeefbeef (3203383023)
    // values in output __MUST BE__ greater than 0 for the test to work correctly
    cl_uint init_value = cl_uint(0xbeefbeef);
    std::vector<cl_uint> output(count, init_value);
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


    // Increments the program reference count. Twice
    error = clRetainProgram(program);
    RETURN_ON_CL_ERROR(error, "clRetainProgram")
    error = clRetainProgram(program);
    RETURN_ON_CL_ERROR(error, "clRetainProgram")

    // Should just decrement the program reference count.
    error = clReleaseProgram(program);
    RETURN_ON_CL_ERROR(error, "clReleaseProgram")
    error = clFinish(queue);
    RETURN_ON_CL_ERROR(error, "clFinish")

    // Should just decrement the program reference count.
    error = clReleaseProgram(program);
    RETURN_ON_CL_ERROR(error, "clReleaseProgram")
    error = clFinish(queue);
    RETURN_ON_CL_ERROR(error, "clFinish")

#ifndef USE_OPENCLC_KERNELS
    // At this point global scope variables should not be destroyed,
    // values in output buffer should not be modified.

    // Read output buffer
    error = clEnqueueReadBuffer(
        queue, output_buffer, CL_TRUE,
        0, sizeof(cl_uint) * output.size(),
        static_cast<void *>(output.data()),
        0, NULL, NULL
    );
    RETURN_ON_CL_ERROR(error, "clEnqueueReadBuffer")
    for(auto& i : output)
    {
        if(i != init_value)
        {
            log_error("ERROR: Test test_global_scope_dtors_executed_once failed.");
            log_error("\tDestructors were executed prematurely.\n");
            RETURN_ON_ERROR(-1)
        }
    }
#endif

    // Release kernel and program, destructors should be called now
    error = clReleaseKernel(kernel);
    RETURN_ON_CL_ERROR(error, "clReleaseKernel")
    error = clReleaseProgram(program);
    RETURN_ON_CL_ERROR(error, "clReleaseProgram")

    // Finish
    error = clFinish(queue);
    RETURN_ON_CL_ERROR(error, "clFinish")

    // Read output buffer
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
        log_error("ERROR: Test test_global_scope_dtors_executed_once failed.");
        // Maybe some dtors were not run?
        for(auto& i : output)
        {
            if(i == init_value)
            {
                log_error("\tSome dtors were not executed.");
                break;
            }
        }
        log_error("\n");
        RETURN_ON_ERROR(-1)
    }

    // Clean
    clReleaseMemObject(output_buffer);
    return error;
}

// TEST3
// Verify that ND-range during destructor execution is set to (1,1,1)

// -----------------------------------------------------------------------------------
// ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
// -----------------------------------------------------------------------------------
#if defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
const char * program_test_dtor_ndrange =
    "__kernel void test_dtor_ndrange(global uint *output)\n"
    "{\n"
    "   ulong gid = get_global_id(0);\n"
    "   output[gid] = 0;\n"
    "}\n"
;
#else
const char * program_test_dtor_ndrange =
    "#include <opencl_memory>\n"
    "#include <opencl_work_item>\n"
    "using namespace cl;\n"
    // struct
    "struct dtor_test_class {\n"
    // non-trivial dtor
    // set all values in buffer to 0 only if ND-range is (1, 1, 1)
    "   ~dtor_test_class() {\n"
    "       if(check()){\n"
    "           for(ulong i = 0; i < size; i++)\n"
    "               buffer[i] = 0;\n"
    "       }\n"
    "   };\n"
    // return true if the ND-range is (1, 1, 1); otherwise - false
    "   bool check() {\n"
    "       return (get_global_size(0) == 1)"
              " && (get_global_size(1) == 1)"
              " && (get_global_size(2) == 1);\n"
    "   }"
    "   ulong size;\n"
    "   global_ptr<uint[]> buffer;\n"
    "};\n"
    // global scope program variable
    "dtor_test_class global_var;\n"

    // values in output __MUST BE__ greater than 0 for the test to work correctly
    "__kernel void test_dtor_ndrange(global_ptr<uint[]> output)\n"
    "{\n"
    "   ulong gid = get_global_id(0);\n"
    // set buffer and size in global var
    "   if(gid == 0){\n"
    "       global_var.buffer = output;\n"
    "       global_var.size = get_global_size(0);\n"
    "   }\n"
    "}\n"
;
#endif

AUTO_TEST_CASE(test_global_scope_dtor_ndrange)
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
        program_test_dtor_ndrange, "test_dtor_ndrange"
    );
    RETURN_ON_ERROR(error)
    return error;
// Use OpenCL C kernels instead of OpenCL C++ kernels (test C++ host code)
#elif defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
    error = create_opencl_kernel(
        context, &program, &kernel,
        program_test_dtor_ndrange, "test_dtor_ndrange", "", false
    );
    RETURN_ON_ERROR(error)
// Normal run
#else
    error = create_opencl_kernel(
        context, &program, &kernel,
        program_test_dtor_ndrange, "test_dtor_ndrange"
    );
    RETURN_ON_ERROR(error)
#endif

    // host vector, size == count, output[0...count-1] == 0xbeefbeef (3203383023)
    // values in output __MUST BE__ greater than 0 for the test to work correctly
    std::vector<cl_uint> output(count, cl_uint(0xbeefbeef));
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

    // Release kernel and program
    // Dtor should be called now
    error = clReleaseKernel(kernel);
    RETURN_ON_CL_ERROR(error, "clReleaseKernel")
    error = clReleaseProgram(program);
    RETURN_ON_CL_ERROR(error, "clReleaseProgram")

    // Finish
    error = clFinish(queue);
    RETURN_ON_CL_ERROR(error, "clFinish")

    // Read output buffer
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
        CHECK_ERROR_MSG(error, "Test test_dtor_ndrange failed.");
    }

    clReleaseMemObject(output_buffer);
    return error;
}

#endif // TEST_CONFORMANCE_CLCPP_API_TEST_DTORS_HPP
