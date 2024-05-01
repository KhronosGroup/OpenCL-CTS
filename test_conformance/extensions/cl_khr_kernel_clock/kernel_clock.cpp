// Copyright (c) 2024 The Khronos Group Inc.
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
#include "procs.h"
#include "harness/typeWrappers.h"

// write 1 to the output if the clock did not increase
static const char *kernel_sources[2] = {
    {
        "__kernel void SampleClock(__global uint* buf)\n"
        "{\n"
        "   ulong time1, time2;\n"
        "   time1 = clock_read_%s();\n"
        "   time2 = clock_read_%s();\n"
        "   if(time1 > time2)\n"
        "   {\n"
        "       buf[0] = 1;\n"
        "    }\n"
        "}\n",
    },
    {
        "__kernel void SampleClock(__global uint* buf)\n"
        "{\n"
        "   uint2 time1, time2;\n"
        "   time1 = clock_read_hilo_%s();\n"
        "   time2 = clock_read_hilo_%s();\n"
        "   if (time1.hi > time2.hi || (time1.hi == time2.hi && time1.lo > "
        "time2.lo))\n"
        "   {\n"
        "        buf[0] = 1;\n"
        "   }\n"
        "}\n",
    }
};

class KernelClockTest {

public:
    KernelClockTest(cl_device_id device, cl_context context,
                    cl_command_queue queue,
                    cl_device_kernel_clock_capabilities_khr capability)
        : device(device), context(context), queue(queue), capability(capability)
    {}

    bool Skip()
    {
        cl_device_kernel_clock_capabilities_khr capabilities;
        cl_int error =
            clGetDeviceInfo(device, CL_DEVICE_KERNEL_CLOCK_CAPABILITIES_KHR,
                            sizeof(cl_device_kernel_clock_capabilities_khr),
                            &capabilities, NULL);
        test_error(error,
                   "Unable to query "
                   "CL_DEVICE_KERNEL_CLOCK_CAPABILITIES_KHR");

        // Skip if capability is not supported
        return capability != (capabilities & capability);
    }

    cl_int RunTest()
    {
        size_t global_size = 1;
        cl_uint buf = 0;
        char kernel_src[512];
        const char *ptr;
        cl_int error;

        // 2 built-ins for each scope
        for (size_t i = 0; i < 2; i++)
        {
            buf = 0;
            clProgramWrapper program;
            clKernelWrapper kernel;
            clMemWrapper out_mem;

            if (i == 0 && !gHasLong)
            {
                log_info("The device does not support ulong. Testing hilo "
                         "built-ins only\n");
                continue;
            }

            switch (capability)
            {
                case CL_DEVICE_KERNEL_CLOCK_SCOPE_DEVICE_KHR: {
                    sprintf(kernel_src, kernel_sources[i], "device", "device");
                    break;
                }
                case CL_DEVICE_KERNEL_CLOCK_SCOPE_WORK_GROUP_KHR: {
                    sprintf(kernel_src, kernel_sources[i], "workgroup",
                            "workgroup");
                    break;
                }
                case CL_DEVICE_KERNEL_CLOCK_SCOPE_SUB_GROUP_KHR: {
                    sprintf(kernel_src, kernel_sources[i], "subgroup",
                            "subgroup");
                    break;
                }
            }

            ptr = kernel_src;

            error = create_single_kernel_helper_create_program(
                context, &program, 1, &ptr);
            test_error(error, "Failed to create program with source");

            error =
                clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
            test_error(error, "Failed to build program");

            out_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                     sizeof(cl_uint), nullptr, &error);
            test_error(error, "clCreateBuffer failed");

            kernel = clCreateKernel(program, "SampleClock", &error);
            test_error(error, "Failed to create kernel");

            error = clSetKernelArg(kernel, 0, sizeof(out_mem), &out_mem);
            test_error(error, "clSetKernelArg failed");

            error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size,
                                           NULL, 0, NULL, NULL);
            test_error(error, "clNDRangeKernel failed");

            error = clEnqueueReadBuffer(queue, out_mem, CL_BLOCKING, 0,
                                        sizeof(cl_uint), &buf, 0, NULL, NULL);
            test_error(error, "clEnqueueReadBuffer failed");

            if (buf == 1)
            {
                log_error(
                    "Sampling the clock returned bad values, time1 > time2.\n");
                return TEST_FAIL;
            }
        }

        return CL_SUCCESS;
    }

private:
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_device_kernel_clock_capabilities_khr capability;
};

int MakeAndRunTest(cl_device_id device, cl_context context,
                   cl_command_queue queue,
                   cl_device_kernel_clock_capabilities_khr capability)
{
    if (!is_extension_available(device, "cl_khr_kernel_clock"))
    {
        log_info(
            "The device does not support the cl_khr_kernel_clock extension.\n");
        return TEST_SKIPPED_ITSELF;
    }

    KernelClockTest test_fixture =
        KernelClockTest(device, context, queue, capability);

    if (test_fixture.Skip())
    {
        return TEST_SKIPPED_ITSELF;
    }

    cl_int error = test_fixture.RunTest();
    test_error_ret(error, "Test Failed", TEST_FAIL);

    return TEST_PASS;
}

int test_device_scope(cl_device_id device, cl_context context,
                      cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest(device, context, queue,
                          CL_DEVICE_KERNEL_CLOCK_SCOPE_DEVICE_KHR);
}

int test_workgroup_scope(cl_device_id device, cl_context context,
                         cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest(device, context, queue,
                          CL_DEVICE_KERNEL_CLOCK_SCOPE_WORK_GROUP_KHR);
}

int test_subgroup_scope(cl_device_id device, cl_context context,
                        cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest(device, context, queue,
                          CL_DEVICE_KERNEL_CLOCK_SCOPE_SUB_GROUP_KHR);
}
