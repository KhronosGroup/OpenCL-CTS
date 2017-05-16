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
#ifndef TEST_CONFORMANCE_CLCPP_API_TEST_SPEC_CONSTS_HPP
#define TEST_CONFORMANCE_CLCPP_API_TEST_SPEC_CONSTS_HPP

#include <vector>
#include <limits>
#include <algorithm>

#include "../common.hpp"

// TEST 1
// Verify that if left unset the specialization constant defaults to the default value set in SPIR-V (zero).

// -----------------------------------------------------------------------------------
// ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
// -----------------------------------------------------------------------------------
#if defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
const char * kernel_test_spec_consts_defaults =
    "__kernel void test_spec_consts_defaults(global int *output)\n"
    "{\n"
    "   ulong gid = get_global_id(0);\n"
    "   output[gid] = 0;\n"
    "}\n"
;
#else
const char * kernel_test_spec_consts_defaults =
    "#include <opencl_memory>\n"
    "#include <opencl_work_item>\n"
    "#include <opencl_spec_constant>\n"
    "using namespace cl;\n"
    "spec_constant<char,  1> spec1(0);\n"
    "spec_constant<uchar, 2> spec2(0);\n"
    "spec_constant<short, 3> spec3(0);\n"
    "spec_constant<ushort,4> spec4(0);\n"
    "spec_constant<int,   5> spec5(0);\n"
    "spec_constant<uint,  6> spec6(0);\n"
    "spec_constant<long,  7> spec7(0);\n"
    "spec_constant<ulong, 8> spec8(0);\n"
    "spec_constant<float, 9> spec9(0.0f);\n"
    "#ifdef cl_khr_fp64\n"
    "spec_constant<double, 10> spec10(0.0);\n"
    "#endif\n"
    "#ifdef cl_khr_fp16\n"
    "spec_constant<half, 11> spec11(0.0h);\n"
    "#endif\n"
    "__kernel void test_spec_consts_defaults(global_ptr<int[]> output)\n"
    "{\n"
    "   ulong gid = get_global_id(0);\n"
    "   int result = 0;\n"
    "   if(get(spec1) != char(0))   result = 1;\n"
    "   if(get(spec2) != uchar(0))  result = 1;\n"
    "   if(get(spec3) != short(0))  result = 1;\n"
    "   if(get(spec4) != ushort(0)) result = 1;\n"
    "   if(get(spec5) != int(0))    result = 1;\n"
    "   if(get(spec6) != uint(0))   result = 1;\n"
    "   if(get(spec7) != long(0))   result = 1;\n"
    "   if(get(spec8) != ulong(0))  result = 1;\n"
    "   if(get(spec9) != float(0))  result = 1;\n"
    "#ifdef cl_khr_fp64\n"
    "   if(get(spec10) != double(0)) result = 1;\n"
    "#endif\n"
    "#ifdef cl_khr_fp16\n"
    "   if(get(spec11) != half(0)) result = 1;\n"
    "#endif\n"
    "   output[gid] = result;\n"
    "}\n"
;
#endif

AUTO_TEST_CASE(test_spec_consts_defaults)
(cl_device_id device, cl_context context, cl_command_queue queue, int count)
{
    int error = CL_SUCCESS;

    cl_mem output_buffer;
    cl_program program;
    cl_kernel kernel;

    size_t dim = 1;
    size_t work_size[1];

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
    error = create_opencl_kernel(context, &program, &kernel, kernel_test_spec_consts_defaults, "test_spec_consts_defaults", options);
    RETURN_ON_ERROR(error)
    return error;
// Use OpenCL C kernels instead of OpenCL C++ kernels (test C++ host code)
#elif defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
    error = create_opencl_kernel(context, &program, &kernel, kernel_test_spec_consts_defaults, "test_spec_consts_defaults", "", false);
    RETURN_ON_ERROR(error)
// Normal run
#else
    // Spec constants are NOT set before clBuildProgram (called in create_opencl_kernel), so
    // they all should default to the default value set in SPIR-V (zero).
    error = create_opencl_kernel(context, &program, &kernel, kernel_test_spec_consts_defaults, "test_spec_consts_defaults", options);
    RETURN_ON_ERROR(error)
#endif

    // host vector, size == 1, output[0] == 1
    std::vector<cl_int> output(1, cl_int(1));
    output_buffer = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE), sizeof(cl_int) * output.size(), NULL, &error);
    RETURN_ON_CL_ERROR(error, "clCreateBuffer")

    error = clEnqueueWriteBuffer(queue, output_buffer, CL_TRUE, 0, sizeof(cl_int) * output.size(), static_cast<void *>(output.data()), 0, NULL, NULL);
    RETURN_ON_CL_ERROR(error, "clEnqueueWriteBuffer")

    error = clSetKernelArg(kernel, 0, sizeof(output_buffer), &output_buffer);
    RETURN_ON_CL_ERROR(error, "clSetKernelArg")

    work_size[0] = output.size();
    error = clEnqueueNDRangeKernel(queue, kernel, dim, NULL, work_size, NULL, 0, NULL, NULL);
    RETURN_ON_CL_ERROR(error, "clEnqueueNDRangeKerne")

    error = clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, sizeof(cl_int) * output.size(), static_cast<void *>(output.data()), 0, NULL, NULL);
    RETURN_ON_CL_ERROR(error, "clEnqueueReadBuffer")

    // if output[0] != 0, then some spec constant(s) did not default to zero.
    if(output[0] != 0)
    {
        RETURN_ON_ERROR_MSG(-1, "Test test_spec_consts_defaults failed, output[0]: %d.", output[0])
    }

    clReleaseMemObject(output_buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    return error;
}

// TEST 2
// Verify that setting an existing specialization constant affects only
// the value of that constant and not of other specialization constants.

// -----------------------------------------------------------------------------------
// ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
// -----------------------------------------------------------------------------------
#if defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
const char * kernel_test_spec_consts_many_constants =
    "__kernel void test_spec_consts_many_constants(global int *output)\n"
    "{\n"
    "   ulong gid = get_global_id(0);\n"
    "   output[gid] = 0;\n"
    "}\n"
;
#else
const char * kernel_test_spec_consts_many_constants =
    "#include <opencl_memory>\n"
    "#include <opencl_work_item>\n"
    "#include <opencl_spec_constant>\n"
    "using namespace cl;\n"
    "spec_constant<int, 1> spec1(0);\n"
    "spec_constant<int, 2> spec2(0);\n"
    "spec_constant<int, 3> spec3(0);\n"
    "__kernel void test_spec_consts_defaults(global_ptr<int[]> output)\n"
    "{\n"
    "   ulong gid = get_global_id(0);\n"
    "   int result = 0;\n"
    "   if(get(spec1) != int(-1024)) result += 1;\n"
    "   if(get(spec2) != int(0))     result += 2;\n"
    "   if(get(spec3) != int(1024))  result += 4;\n"
    "   output[gid] = result;\n"
    "}\n"
;
#endif

AUTO_TEST_CASE(test_spec_consts_many_constants)
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
        kernel_test_spec_consts_many_constants, "test_spec_consts_many_constants"
    );
    RETURN_ON_ERROR(error)
    return error;
// Use OpenCL C kernels instead of OpenCL C++ kernels (test C++ host code)
#elif defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
    error = create_opencl_kernel(
        context, &program, &kernel,
        kernel_test_spec_consts_many_constants, "test_spec_consts_many_constants", "", false
    );
    RETURN_ON_ERROR(error)
// Normal run
#else
    // Create program
    error = create_openclcpp_program(context, &program, 1, &kernel_test_spec_consts_many_constants);
    RETURN_ON_ERROR(error)

    // Set specialization constants

    // clSetProgramSpecializationConstant(
    //     cl_program /* program */, cl_uint /* spec_id */, size_t  /* spec_size */,const void* /* spec_value */
    // )
    cl_int spec1 = -1024;
    cl_int spec3 = 1024;
    // Set spec1
    error = clSetProgramSpecializationConstant(program, cl_uint(1), sizeof(cl_int), static_cast<void*>(&spec1));
    RETURN_ON_CL_ERROR(error, "clSetProgramSpecializationConstant")
    // Specialization constant spec2 should default to zero
    // Set spec3
    error = clSetProgramSpecializationConstant(program, cl_uint(3), sizeof(cl_int), static_cast<void*>(&spec3));
    RETURN_ON_CL_ERROR(error, "clSetProgramSpecializationConstant")

    // Build program and create kernel
    error = build_program_create_kernel_helper(
        context, &program, &kernel, 1, &kernel_test_spec_consts_many_constants, "test_spec_consts_many_constants"
    );
    RETURN_ON_ERROR(error)
#endif

    // host vector, size == 1, output[0] == 1
    std::vector<cl_int> output(1, cl_int(1));
    output_buffer = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE), sizeof(cl_int) * output.size(), NULL, &error);
    RETURN_ON_CL_ERROR(error, "clCreateBuffer")

    error = clEnqueueWriteBuffer(queue, output_buffer, CL_TRUE, 0, sizeof(cl_int) * output.size(), static_cast<void *>(output.data()), 0, NULL, NULL);
    RETURN_ON_CL_ERROR(error, "clEnqueueWriteBuffer")

    error = clSetKernelArg(kernel, 0, sizeof(output_buffer), &output_buffer);
    RETURN_ON_CL_ERROR(error, "clSetKernelArg")

    work_size[0] = output.size();
    error = clEnqueueNDRangeKernel(queue, kernel, dim, NULL, work_size, NULL, 0, NULL, NULL);
    RETURN_ON_CL_ERROR(error, "clEnqueueNDRangeKernel")

    error = clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, sizeof(cl_int) * output.size(), static_cast<void *>(output.data()), 0, NULL, NULL);
    RETURN_ON_CL_ERROR(error, "clEnqueueReadBuffer")

    // if output[0] != 0, then values of spec constants were incorrect
    if(output[0] != 0)
    {
        RETURN_ON_ERROR_MSG(-1, "Test test_spec_consts_many_constants failed, output[0]: %d.", output[0]);
    }

    clReleaseMemObject(output_buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    return error;
}

// TEST 3
// Verify that the API correctly handles the size of a specialization constant by exercising
// the API for specialization constants of different types (int, bool, float, etc.)

// -----------------------------------------------------------------------------------
// ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
// -----------------------------------------------------------------------------------
#if defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
const char * kernel_test_spec_consts_different_types =
    "__kernel void test_spec_consts_different_types(global int *output)\n"
    "{\n"
    "   ulong gid = get_global_id(0);\n"
    "   output[gid] = 0;\n"
    "}\n"
;
#else
const char * kernel_test_spec_consts_different_types =
    "#include <opencl_memory>\n"
    "#include <opencl_work_item>\n"
    "#include <opencl_spec_constant>\n"
    "#include <opencl_limits>\n"
    "using namespace cl;\n"
    "spec_constant<char,  1> spec1(0);\n"
    "spec_constant<uchar, 2> spec2(0);\n"
    "spec_constant<short, 3> spec3(0);\n"
    "spec_constant<ushort,4> spec4(0);\n"
    "spec_constant<int,   5> spec5(0);\n"
    "spec_constant<uint,  6> spec6(0);\n"
    "spec_constant<long,  7> spec7(0);\n"
    "spec_constant<ulong, 8> spec8(0);\n"
    "spec_constant<float, 9> spec9(0.0f);\n"
    "#ifdef cl_khr_fp64\n"
    "spec_constant<double, 10> spec10(0.0);\n"
    "#endif\n"
    "#ifdef cl_khr_fp16\n"
    "spec_constant<half, 11> spec11(0.0h);\n"
    "#endif\n"
    "__kernel void test_spec_consts_different_types(global_ptr<int[]> output)\n"
    "{\n"
    "   ulong gid = get_global_id(0);\n"
    "   int result = 0;\n"
    "   if(get(spec1) != char(CHAR_MAX))    result += 1;\n"
    "   if(get(spec2) != uchar(UCHAR_MAX))  result += 2;\n"
    "   if(get(spec3) != short(SHRT_MAX))   result += 4;\n"
    "   if(get(spec4) != ushort(USHRT_MAX)) result += 8;\n"
    "   if(get(spec5) != int(INT_MAX))      result += 16;\n"
    "   if(get(spec6) != uint(UINT_MAX))    result += 32;\n"
    "   if(get(spec7) != long(LONG_MAX))    result += 64;\n"
    "   if(get(spec8) != ulong(ULONG_MAX))  result += 128;\n"
    "   if(get(spec9) != float(FLT_MAX))    result += 256;\n"
    "#ifdef cl_khr_fp64\n"
    "   if(get(spec10) != double(DBL_MAX)) result += 512;\n"
    "#endif\n"
    "#ifdef cl_khr_fp16\n"
    "   if(get(spec11) != half(HALF_MAX)) result += 1024;\n"
    "#endif\n"
    "   output[gid] = result;\n"
    "}\n"
;
#endif


AUTO_TEST_CASE(test_spec_consts_different_types)
(cl_device_id device, cl_context context, cl_command_queue queue, int count)
{
    int error = CL_SUCCESS;

    cl_mem output_buffer;
    cl_program program;
    cl_kernel kernel;

    size_t dim = 1;
    size_t work_size[1];

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
    error = create_opencl_kernel(context, &program, &kernel, kernel_test_spec_consts_different_types, "test_spec_consts_different_types", options);
    RETURN_ON_ERROR(error)
    return error;
// Use OpenCL C kernels instead of OpenCL C++ kernels (test C++ host code)
#elif defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
    error = create_opencl_kernel(context, &program, &kernel, kernel_test_spec_consts_different_types, "test_spec_consts_different_types", "", false);
    RETURN_ON_ERROR(error)
// Normal run
#else
    // Create program
    error = create_openclcpp_program(context, &program, 1, &kernel_test_spec_consts_different_types, options.c_str());
    RETURN_ON_ERROR(error)

    // Set specialization constants
    cl_uint spec_id = 1;

    cl_char   spec1 = CL_CHAR_MAX;
    cl_uchar  spec2 = CL_UCHAR_MAX;
    cl_short  spec3 = CL_SHRT_MAX;
    cl_ushort spec4 = CL_USHRT_MAX;
    cl_int    spec5 = CL_INT_MAX;
    cl_uint   spec6 = CL_UINT_MAX;
    cl_long   spec7 = CL_LONG_MAX;
    cl_ulong  spec8 = CL_ULONG_MAX;
    cl_float  spec9 = CL_FLT_MAX;
    cl_double spec10 = CL_DBL_MAX;
    cl_half   spec11 = CL_HALF_MAX;

    // Set spec1
    error = clSetProgramSpecializationConstant(program, spec_id++, sizeof(cl_char), static_cast<void*>(&spec1));
    RETURN_ON_CL_ERROR(error, "clSetProgramSpecializationConstant")
    // Set spec2
    error = clSetProgramSpecializationConstant(program, spec_id++, sizeof(cl_uchar), static_cast<void*>(&spec2));
    RETURN_ON_CL_ERROR(error, "clSetProgramSpecializationConstant")
    // Set spec3
    error = clSetProgramSpecializationConstant(program, spec_id++, sizeof(cl_short), static_cast<void*>(&spec3));
    RETURN_ON_CL_ERROR(error, "clSetProgramSpecializationConstant")
    // Set spec4
    error = clSetProgramSpecializationConstant(program, spec_id++, sizeof(cl_ushort), static_cast<void*>(&spec4));
    RETURN_ON_CL_ERROR(error, "clSetProgramSpecializationConstant")
    // Set spec5
    error = clSetProgramSpecializationConstant(program, spec_id++, sizeof(cl_int), static_cast<void*>(&spec5));
    RETURN_ON_CL_ERROR(error, "clSetProgramSpecializationConstant")
    // Set spec6
    error = clSetProgramSpecializationConstant(program, spec_id++, sizeof(cl_uint), static_cast<void*>(&spec6));
    RETURN_ON_CL_ERROR(error, "clSetProgramSpecializationConstant")
    // Set spec7
    error = clSetProgramSpecializationConstant(program, spec_id++, sizeof(cl_long), static_cast<void*>(&spec7));
    RETURN_ON_CL_ERROR(error, "clSetProgramSpecializationConstant")
    // Set spec8
    error = clSetProgramSpecializationConstant(program, spec_id++, sizeof(cl_ulong), static_cast<void*>(&spec8));
    RETURN_ON_CL_ERROR(error, "clSetProgramSpecializationConstant")
    // Set spec9
    error = clSetProgramSpecializationConstant(program, spec_id++, sizeof(cl_float), static_cast<void*>(&spec9));
    RETURN_ON_CL_ERROR(error, "clSetProgramSpecializationConstant")
    // Set spec10
    if(is_extension_available(device, "cl_khr_fp64"))
    {
        error = clSetProgramSpecializationConstant(program, cl_uint(10), sizeof(cl_double), static_cast<void*>(&spec10));
        RETURN_ON_CL_ERROR(error, "clSetProgramSpecializationConstant")
    }
    // Set spec11
    if(is_extension_available(device, "cl_khr_fp16"))
    {
        error = clSetProgramSpecializationConstant(program, cl_uint(11), sizeof(cl_half), static_cast<void*>(&spec11));
        RETURN_ON_CL_ERROR(error, "clSetProgramSpecializationConstant")
    }

    // Build program and create kernel
    error = build_program_create_kernel_helper(
        context, &program, &kernel, 1, &kernel_test_spec_consts_many_constants, "test_spec_consts_many_constants"
    );
    RETURN_ON_ERROR(error)
#endif

    // Copy output to output_buffer, run kernel, copy output_buffer back to output, check result

    // host vector, size == 1, output[0] == 1
    std::vector<cl_int> output(1, cl_int(1));
    output_buffer = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE), sizeof(cl_int) * output.size(), NULL, &error);
    RETURN_ON_CL_ERROR(error, "clCreateBuffer")

    error = clEnqueueWriteBuffer(queue, output_buffer, CL_TRUE, 0, sizeof(cl_int) * output.size(), static_cast<void *>(output.data()), 0, NULL, NULL);
    RETURN_ON_CL_ERROR(error, "clEnqueueWriteBuffer")

    error = clSetKernelArg(kernel, 0, sizeof(output_buffer), &output_buffer);
    RETURN_ON_CL_ERROR(error, "clSetKernelArg")

    work_size[0] = output.size();
    error = clEnqueueNDRangeKernel(queue, kernel, dim, NULL, work_size, NULL, 0, NULL, NULL);
    RETURN_ON_CL_ERROR(error, "clEnqueueNDRangeKernel")

    error = clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, sizeof(cl_int) * output.size(), static_cast<void *>(output.data()), 0, NULL, NULL);
    RETURN_ON_CL_ERROR(error, "clEnqueueReadBuffer")

    // if output[0] != 0, then some spec constants had incorrect values
    if(output[0] != 0)
    {
        RETURN_ON_ERROR_MSG(-1, "Test test_spec_consts_different_types failed, output[0]: %d.", output[0])
    }

    clReleaseMemObject(output_buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    return error;
}

#endif // TEST_CONFORMANCE_CLCPP_API_TEST_SPEC_CONSTS_HPP
