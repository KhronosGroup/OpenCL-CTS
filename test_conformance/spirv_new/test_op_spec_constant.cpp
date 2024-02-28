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

#include "testBase.h"
#include "types.hpp"


template <typename T>
int run_case(cl_device_id deviceID, cl_context context, cl_command_queue queue,
             const char *name, T init_buffer, T spec_constant_value,
             T final_value, bool use_spec_constant)
{
    clProgramWrapper prog;
    cl_int err = CL_SUCCESS;
    if (use_spec_constant)
    {
        spec_const new_spec_const =
            spec_const(101, sizeof(T), &spec_constant_value);

        err =
            get_program_with_il(prog, deviceID, context, name, new_spec_const);
    }
    else
    {
        err = get_program_with_il(prog, deviceID, context, name);
    }
    SPIRV_CHECK_ERROR(err, "Failed to build program");

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
        log_error("Values do not match. Expected %d obtained %d\n", reference,
                  device_results);
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
    TEST_SPIRV_FUNC_VERSION(op_spec_constant_##NAME##_simple, Version(2, 2))   \
    {                                                                          \
        type init_value = init_buffer;                                         \
        type final_value = init_value + spec_constant_value;                   \
        return test_spec_constant(                                             \
            deviceID, context, queue, "op_spec_constant_" #NAME "_simple",     \
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
// documenation: 'If a specialization constant is a boolean
// constant, spec_value should be a pointer to a cl_uchar value'

TEST_SPIRV_FUNC_VERSION(op_spec_constant_true_simple, Version(2, 2))
{
    // 1-st ndrange init_value is expected value (no change)
    // 2-nd ndrange sets spec const to 'false' so value = value + 1
    cl_uchar value = (cl_uchar)7;
    cl_uchar init_value = value;
    cl_uchar final_value = value + 1;
    return test_spec_constant<cl_uchar>(deviceID, context, queue,
                                        "op_spec_constant_true_simple",
                                        init_value, 0, final_value);
}

TEST_SPIRV_FUNC_VERSION(op_spec_constant_false_simple, Version(2, 2))
{
    // 1-st ndrange init_value is expected value (no change)
    // 2-nd ndrange sets spec const to 'true' so value = value + 1
    cl_uchar value = (cl_uchar)7;
    cl_uchar init_value = value;
    cl_uchar final_value = value + 1;
    return test_spec_constant<cl_uchar>(deviceID, context, queue,
                                        "op_spec_constant_false_simple",
                                        init_value, 1, final_value);
}
