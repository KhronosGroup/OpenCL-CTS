//
// Copyright (c) 2020-2023 The Khronos Group Inc.
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

#include <sstream>

#include "testBase.h"
#include "types.hpp"


template <typename T>
int run_case(cl_device_id deviceID, cl_context context, cl_command_queue queue,
             const char *name, T init_buffer, T spec_constant_value,
             T final_value, bool use_spec_constant)
{
    clProgramWrapper prog;
    cl_int err = get_unbuilt_program_with_il(prog, deviceID, context, name);
    SPIRV_CHECK_ERROR(err, "Failed to create program");

    if (use_spec_constant)
    {
        err = clSetProgramSpecializationConstant(prog, 101, sizeof(T),
                                                 &spec_constant_value);
        SPIRV_CHECK_ERROR(err, "Failed to set specialization constant");
    }

    err = clBuildProgram(prog, 1, &deviceID, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        cl_int outputErr = OutputBuildLog(prog, deviceID);
        SPIRV_CHECK_ERROR(outputErr, "OutputBuildLog failed");
        return -1;
    }

    clKernelWrapper kernel = clCreateKernel(prog, "spec_const_kernel", &err);
    SPIRV_CHECK_ERROR(err, "Failed to create kernel");
    size_t bytes = sizeof(T);
    clMemWrapper output_buffer =
        clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, bytes,
                       &init_buffer, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create output_buffer");

    err = clSetKernelArg(kernel, 0, sizeof(clMemWrapper), &output_buffer);
    SPIRV_CHECK_ERROR(err, "Failed to set kernel argument output_buffer");

    size_t work_size = 1;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &work_size, NULL, 0,
                                 NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to enqueue kernel");

    T device_results = 0;
    err = clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, bytes,
                              &device_results, 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to copy from output_buffer");
    T reference = 0;
    use_spec_constant ? reference = final_value : reference = init_buffer;
    if (device_results != reference)
    {
        std::stringstream sstr;
        sstr << "Values do not match. Expected " << reference << " obtained "
             << device_results;
        log_error("%s\n", sstr.str().c_str());
        err = -1;
    }
    return err;
}

template <typename T>
int test_spec_constant(cl_device_id deviceID, cl_context context,
                       cl_command_queue queue, const char *name, T init_buffer,
                       T spec_constant_value, T final_value)
{
    if (std::string(name).find("double") != std::string::npos)
    {
        if (!is_extension_available(deviceID, "cl_khr_fp64"))
        {
            log_info("Extension cl_khr_fp64 not supported; skipping double "
                     "tests.\n");
            return TEST_SKIPPED_ITSELF;
        }
    }
    if (std::string(name).find("half") != std::string::npos)
    {
        if (!is_extension_available(deviceID, "cl_khr_fp16"))
        {
            log_info("Extension cl_khr_fp16 not supported; skipping half "
                     "tests.\n");
            return TEST_SKIPPED_ITSELF;
        }
    }
    cl_int err = CL_SUCCESS;
    err = run_case<T>(deviceID, context, queue, name, init_buffer,
                      spec_constant_value, final_value, false);
    err |= run_case<T>(deviceID, context, queue, name, init_buffer,
                       spec_constant_value, final_value, true);

    if (err == CL_SUCCESS)
    {
        return TEST_PASS;
    }
    else
    {
        return TEST_FAIL;
    }
}


#define TEST_SPEC_CONSTANT(NAME, type, init_buffer, spec_constant_value)       \
    REGISTER_TEST_VERSION(op_spec_constant_##NAME##_simple, Version(2, 2))     \
    {                                                                          \
        type init_value = init_buffer;                                         \
        type final_value = init_value + spec_constant_value;                   \
        return test_spec_constant(                                             \
            device, context, queue, "op_spec_constant_" #NAME "_simple",       \
            init_value, (type)spec_constant_value, final_value);               \
    }

// type name, type, value init, spec constant value
TEST_SPEC_CONSTANT(uint, cl_uint, 25, 43)
TEST_SPEC_CONSTANT(uchar, cl_uchar, 19, 4)
TEST_SPEC_CONSTANT(ushort, cl_ushort, 6000, 3000)
TEST_SPEC_CONSTANT(ulong, cl_ulong, 9223372036854775000UL, 200)
TEST_SPEC_CONSTANT(float, cl_float, 1.5, -3.7)
TEST_SPEC_CONSTANT(half, cl_half, 1, 2)
TEST_SPEC_CONSTANT(double, cl_double, 14534.53453, 1.53453)

// Boolean tests
// documentation: 'If a specialization constant is a boolean
// constant, spec_value should be a pointer to a cl_uchar value'

REGISTER_TEST_VERSION(op_spec_constant_true_simple, Version(2, 2))
{
    // 1-st ndrange init_value is expected value (no change)
    // 2-nd ndrange sets spec const to 'false' so value = value + 1
    cl_uchar value = (cl_uchar)7;
    cl_uchar init_value = value;
    cl_uchar final_value = value + 1;
    return test_spec_constant<cl_uchar>(device, context, queue,
                                        "op_spec_constant_true_simple",
                                        init_value, 0, final_value);
}

REGISTER_TEST_VERSION(op_spec_constant_false_simple, Version(2, 2))
{
    // 1-st ndrange init_value is expected value (no change)
    // 2-nd ndrange sets spec const to 'true' so value = value + 1
    cl_uchar value = (cl_uchar)7;
    cl_uchar init_value = value;
    cl_uchar final_value = value + 1;
    return test_spec_constant<cl_uchar>(device, context, queue,
                                        "op_spec_constant_false_simple",
                                        init_value, 1, final_value);
}

REGISTER_TEST_VERSION(op_spec_constant_compile_link, Version(2, 2))
{
    // Compile the first object file.
    // This object file has a spec constant with id 101.
    // Set the spec constant value on this object file several times - the value
    // set before compiling should be used, and all values set after compiling
    // should be ignored.

    clProgramWrapper obj1;
    cl_int err = get_unbuilt_program_with_il(
        obj1, device, context, "op_spec_constant_compile_link_obj");
    SPIRV_CHECK_ERROR(err, "Failed to create obj program");

    const cl_uint oValue = 1;
    err =
        clSetProgramSpecializationConstant(obj1, 101, sizeof(oValue), &oValue);
    SPIRV_CHECK_ERROR(err, "Failed to set obj spec constant before compiling");

    err = clCompileProgram(obj1, 1, &device, nullptr, 0, nullptr, nullptr,
                           nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        OutputBuildLog(obj1, device);
    }
    SPIRV_CHECK_ERROR(err, "Failed to compile obj1 program");

    const cl_uint bogus0 = 999; // should be ignored
    err =
        clSetProgramSpecializationConstant(obj1, 101, sizeof(bogus0), &bogus0);
    SPIRV_CHECK_ERROR(err, "Failed to set obj spec constant after compiling");

    // Compile the second object file.
    // This object file also has a spec constant with id 101.

    clProgramWrapper obj2;
    err = get_unbuilt_program_with_il(obj2, device, context,
                                      "op_spec_constant_compile_link_main");
    SPIRV_CHECK_ERROR(err, "Failed to create main program");

    const cl_uint mValue = 2;
    err =
        clSetProgramSpecializationConstant(obj2, 101, sizeof(mValue), &mValue);
    SPIRV_CHECK_ERROR(err, "Failed to set main spec constant before compiling");

    err = clCompileProgram(obj2, 1, &device, nullptr, 0, nullptr, nullptr,
                           nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        OutputBuildLog(obj2, device);
    }
    SPIRV_CHECK_ERROR(err, "Failed to compile obj2 program");

    // Link the two object files together and create the test kernel.

    const cl_program progs[] = { obj1, obj2 };
    clProgramWrapper prog = clLinkProgram(context, 1, &device, nullptr, 2,
                                          progs, nullptr, nullptr, &err);
    if (err != CL_SUCCESS && prog != nullptr)
    {
        OutputBuildLog(prog, device);
    }
    SPIRV_CHECK_ERROR(err, "Failed to link program");

    const cl_uint bogus1 = 99999; // should be ignored
    err =
        clSetProgramSpecializationConstant(obj1, 101, sizeof(bogus1), &bogus1);
    SPIRV_CHECK_ERROR(err, "Failed to set obj spec constant after linking");

    clKernelWrapper kernel = clCreateKernel(prog, "spec_const_kernel", &err);
    SPIRV_CHECK_ERROR(err, "Failed to create kernel");

    // Execute the test kernel and get the results.

    clMemWrapper output_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                                sizeof(cl_uint), nullptr, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create output_buffer");

    err = clSetKernelArg(kernel, 0, sizeof(clMemWrapper), &output_buffer);
    SPIRV_CHECK_ERROR(err, "Failed to set kernel argument output_buffer");

    size_t work_size = 1;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &work_size, nullptr,
                                 0, nullptr, nullptr);
    SPIRV_CHECK_ERROR(err, "Failed to enqueue kernel");

    cl_uint result = 0;
    err = clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, sizeof(cl_uint),
                              &result, 0, nullptr, nullptr);
    SPIRV_CHECK_ERROR(err, "Failed to read result from output_buffer");

    // The expected value is the sum of the spec constant values set before
    // compiling.

    const cl_uint expected = oValue + mValue;
    if (result != expected)
    {
        log_error("Result mismatch: expected %u, got %u\n", expected, result);
        return TEST_FAIL;
    }

    return TEST_PASS;
}

static int build_or_compile(cl_program prog, cl_device_id device, bool build)
{
    cl_int err = CL_SUCCESS;

    if (build)
    {
        err = clBuildProgram(prog, 0, nullptr, nullptr, nullptr, nullptr);
    }
    else
    {
        err = clCompileProgram(prog, 0, nullptr, nullptr, 0, nullptr, nullptr,
                               nullptr, nullptr);
    }
    if (err != CL_SUCCESS)
    {
        OutputBuildLog(prog, device);
    }
    return err;
}

static int spec_constant_compile_twice_helper(cl_device_id device,
                                              cl_context context,
                                              cl_command_queue queue,
                                              bool build)
{
    clProgramWrapper prog, linked;
    cl_int err = get_unbuilt_program_with_il(prog, device, context,
                                             "op_spec_constant_compile_twice");
    SPIRV_CHECK_ERROR(err, "Failed to create program");

    const cl_uint sValue0 = 1;
    err = clSetProgramSpecializationConstant(prog, 101, sizeof(cl_uint),
                                             &sValue0);
    SPIRV_CHECK_ERROR(err, "Failed to set initial specialization constant");

    err = build_or_compile(prog, device, build);
    SPIRV_CHECK_ERROR(err, "Failed to build or compile initial program");

    const cl_uint sValue1 = 2;
    err = clSetProgramSpecializationConstant(prog, 101, sizeof(cl_uint),
                                             &sValue1);
    SPIRV_CHECK_ERROR(err, "Failed to set updated specialization constant");

    err = build_or_compile(prog, device, build);
    SPIRV_CHECK_ERROR(err, "Failed to build or compile updated program");

    if (build == false)
    {
        linked = clLinkProgram(context, 1, &device, nullptr, 1, &prog, nullptr,
                               nullptr, &err);
        if (err != CL_SUCCESS && linked != nullptr)
        {
            OutputBuildLog(linked, device);
        }
        SPIRV_CHECK_ERROR(err, "Failed to link updated program");
    }

    clKernelWrapper kernel =
        clCreateKernel(build ? prog : linked, "spec_const_kernel", &err);
    SPIRV_CHECK_ERROR(err, "Failed to create kernel");

    // Execute the test kernel and get the results.

    clMemWrapper output_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                                sizeof(cl_uint), nullptr, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create output_buffer");

    err = clSetKernelArg(kernel, 0, sizeof(clMemWrapper), &output_buffer);
    SPIRV_CHECK_ERROR(err, "Failed to set kernel argument output_buffer");

    size_t work_size = 1;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &work_size, nullptr,
                                 0, nullptr, nullptr);
    SPIRV_CHECK_ERROR(err, "Failed to enqueue kernel");

    cl_uint result = 0;
    err = clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, sizeof(cl_uint),
                              &result, 0, nullptr, nullptr);
    SPIRV_CHECK_ERROR(err, "Failed to read result from output_buffer");

    // The expected value is the second spec constant value, not the first.

    const cl_uint expected = sValue1;
    if (result != expected)
    {
        log_error("Result mismatch: expected %u, got %u\n", expected, result);
        return TEST_FAIL;
    }

    return TEST_PASS;
}

REGISTER_TEST_VERSION(op_spec_constant_build_twice, Version(2, 2))
{
    return spec_constant_compile_twice_helper(device, context, queue, true);
}

REGISTER_TEST_VERSION(op_spec_constant_compile_twice, Version(2, 2))
{
    return spec_constant_compile_twice_helper(device, context, queue, false);
}
