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
#if defined(__APPLE__)
#include <OpenCL/opencl.h>
#include <mach/mach_time.h>
#else
#include <CL/cl.h>
#include <malloc.h>
#endif
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "harness/kernelHelpers.h"

///////////////////////////////////////////////////////////////////////////////
// CL error checking.

#if defined(_MSC_VER)
#define CL_EXIT_ERROR(cmd, ...)                                                \
    {                                                                          \
        if ((cmd) != CL_SUCCESS)                                               \
        {                                                                      \
            log_error("CL ERROR: %s %u: ", __FILE__, __LINE__);                \
            log_error(##__VA_ARGS__);                                          \
            log_error("\n");                                                   \
            return -1;                                                         \
        }                                                                      \
    }
#else
#define CL_EXIT_ERROR(cmd, format, ...)                                        \
    {                                                                          \
        if ((cmd) != CL_SUCCESS)                                               \
        {                                                                      \
            log_error("CL ERROR: %s %u: ", __FILE__, __LINE__);                \
            log_error(format, ##__VA_ARGS__);                                  \
            log_error("\n");                                                   \
            return -1;                                                         \
        }                                                                      \
    }
#endif

#define CL_EXIT_BUILD_ERROR(cmd, program, format, ...)                         \
    {                                                                          \
        if ((cmd) != CL_SUCCESS)                                               \
        {                                                                      \
            cl_uint num_devices_;                                              \
            clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES,                  \
                             sizeof(num_devices_), &num_devices_, NULL);       \
            cl_device_id *device_list;                                         \
            device_list =                                                      \
                (cl_device_id *)malloc(num_devices_ * sizeof(cl_device_id));   \
            clGetProgramInfo(program, CL_PROGRAM_DEVICES,                      \
                             num_devices_ * sizeof(cl_device_id), device_list, \
                             NULL);                                            \
            for (unsigned i = 0; i < num_devices_; ++i)                        \
            {                                                                  \
                size_t len;                                                    \
                char buffer[2048];                                             \
                clGetProgramBuildInfo(program, device_list[i],                 \
                                      CL_PROGRAM_BUILD_LOG, sizeof(buffer),    \
                                      buffer, &len);                           \
                log_error("DEVICE %u CL BUILD ERROR: %s(%u): ", i, __FILE__,   \
                          __LINE__);                                           \
                log_error(format, ##__VA_ARGS__);                              \
                log_error("\n");                                               \
            }                                                                  \
            free(device_list);                                                 \
            return -1;                                                         \
        }                                                                      \
    }

const char *src[] = { "__kernel void simple_task(__global float* output) {\n"
                      "  output[0] += 1;\n"
                      "}\n" };

enum
{
    MaxDevices = 8
};

int test_userevents(cl_device_id deviceID, cl_context context,
                    cl_command_queue queue, int num_elements)
{

    cl_int err;

    cl_event u1 = clCreateUserEvent(context, &err);
    CL_EXIT_ERROR(err, "clCreateUserEvent failed");

    // Test event properties.
    cl_int s;
    size_t sizeofs;
    CL_EXIT_ERROR(clGetEventInfo(u1, CL_EVENT_COMMAND_EXECUTION_STATUS,
                                 sizeof s, &s, &sizeofs),
                  "clGetEventInfo failed");
    CL_EXIT_ERROR((sizeof s == sizeofs) ? CL_SUCCESS : -1,
                  "clGetEventInfo returned wrong size for "
                  "CL_EVENT_COMMAND_EXECUTION_STATUS");
    CL_EXIT_ERROR((s == CL_SUBMITTED) ? CL_SUCCESS : -1,
                  "clGetEventInfo returned wrong value for "
                  "CL_EVENT_COMMAND_EXECUTION_STATUS");

    cl_command_type t;
    size_t sizeoft;
    CL_EXIT_ERROR(
        clGetEventInfo(u1, CL_EVENT_COMMAND_TYPE, sizeof t, &t, &sizeoft),
        "clGetEventInfo failed");
    CL_EXIT_ERROR(
        (sizeof t == sizeoft) ? CL_SUCCESS : -1,
        "clGetEventInfo returned wrong size for CL_EVENT_COMMAND_TYPE");
    CL_EXIT_ERROR(
        (t == CL_COMMAND_USER) ? CL_SUCCESS : -1,
        "clGetEventInfo returned wrong value for CL_EVENT_COMMAND_TYPE");

    cl_command_queue q;
    size_t sizeofq;
    CL_EXIT_ERROR(
        clGetEventInfo(u1, CL_EVENT_COMMAND_QUEUE, sizeof q, &q, &sizeofq),
        "clGetEventInfo failed");
    CL_EXIT_ERROR(
        (sizeof q == sizeofq) ? CL_SUCCESS : -1,
        "clGetEventInfo returned wrong size for CL_EVENT_COMMAND_QUEUE");
    CL_EXIT_ERROR(
        (q == NULL) ? CL_SUCCESS : -1,
        "clGetEventInfo returned wrong value for CL_EVENT_COMMAND_QUEUE");

    cl_context c;
    size_t sizeofc;
    CL_EXIT_ERROR(clGetEventInfo(u1, CL_EVENT_CONTEXT, sizeof c, &c, &sizeofc),
                  "clGetEventInfo failed");
    CL_EXIT_ERROR((sizeof c == sizeofc) ? CL_SUCCESS : -1,
                  "clGetEventInfo returned wrong size for CL_EVENT_CONTEXT");
    CL_EXIT_ERROR((c == context) ? CL_SUCCESS : -1,
                  "clGetEventInfo returned wrong value for CL_EVENT_CONTEXT");

    cl_ulong p;
    err = clGetEventProfilingInfo(u1, CL_PROFILING_COMMAND_QUEUED, sizeof p, &p,
                                  0);
    CL_EXIT_ERROR((err != CL_SUCCESS) ? CL_SUCCESS : -1,
                  "clGetEventProfilingInfo returned wrong error.");

    // Test semantics.
    cl_program program;
    err = create_single_kernel_helper_create_program(context, &program, 1, src);
    CL_EXIT_ERROR(err, "clCreateProgramWithSource failed");

    CL_EXIT_BUILD_ERROR(clBuildProgram(program, 0, NULL, "", NULL, NULL),
                        program, "Building program from inline src:\t%s",
                        src[0]);

    cl_kernel k0 = clCreateKernel(program, "simple_task", &err);
    CL_EXIT_ERROR(err, "clCreateKernel failed");

    float buffer[1];
    cl_mem output = clCreateBuffer(context, CL_MEM_USE_HOST_PTR, sizeof buffer,
                                   buffer, &err);
    CL_EXIT_ERROR(err, "clCreateBuffer failed.");

    CL_EXIT_ERROR(clSetKernelArg(k0, 0, sizeof(output), &output),
                  "clSetKernelArg failed");


    // Successful case.
    // //////////////////////////////////////////////////////////////////////////////////////
    {
        cl_event e[4];
        cl_uint N = sizeof e / sizeof(cl_event);

        log_info("Enqueuing tasks\n");
        for (cl_uint i = 0; i != N; ++i)
            CL_EXIT_ERROR(clEnqueueTask(queue, k0, 1, &u1, &e[i]),
                          "clEnqueueTaskFailed");

        log_info("Checking task status before setting user event status\n");
        for (cl_uint i = 0; i != N; ++i)
        {
            CL_EXIT_ERROR(clGetEventInfo(e[i],
                                         CL_EVENT_COMMAND_EXECUTION_STATUS,
                                         sizeof s, &s, 0),
                          "clGetEventInfo failed");
            CL_EXIT_ERROR(
                (s >= CL_SUBMITTED) ? CL_SUCCESS : -1,
                "clGetEventInfo %u returned wrong status before user event", i);
        }

        log_info("Setting user event status to complete\n");
        CL_EXIT_ERROR(clSetUserEventStatus(u1, CL_COMPLETE),
                      "clSetUserEventStatus failed");

        log_info("Waiting for tasks to finish executing\n");
        CL_EXIT_ERROR(clWaitForEvents(1, &e[N - 1]), "clWaitForEvent failed");

        log_info("Checking task status after setting user event status\n");
        for (cl_uint i = 0; i != N; ++i)
        {
            CL_EXIT_ERROR(clGetEventInfo(e[i],
                                         CL_EVENT_COMMAND_EXECUTION_STATUS,
                                         sizeof s, &s, 0),
                          "clGetEventInfo failed");
            CL_EXIT_ERROR((s != CL_QUEUED) ? CL_SUCCESS : -1,
                          "clGetEventInfo %u returned wrong status %04x after "
                          "successful user event",
                          i, s);
        }

        CL_EXIT_ERROR(clReleaseEvent(u1), "clReleaseEvent failed");

        for (cl_uint i = 0; i != N; ++i)
            CL_EXIT_ERROR(clReleaseEvent(e[i]), "clReleaseEvent failed");

        log_info("Successful user event case passed.\n");
    }

    // Test unsuccessful user event case.
    // ///////////////////////////////////////////////////////////////////
    {
        cl_event u2 = clCreateUserEvent(context, &err);
        CL_EXIT_ERROR(err, "clCreateUserEvent failed");

        cl_event e[4];
        cl_uint N = sizeof e / sizeof(cl_event);

        log_info("Enqueuing tasks\n");
        for (cl_uint i = 0; i != N; ++i)
            CL_EXIT_ERROR(clEnqueueTask(queue, k0, 1, &u2, &e[i]),
                          "clEnqueueTaskFailed");

        log_info("Checking task status before setting user event status\n");
        for (cl_uint i = 0; i != N; ++i)
        {
            CL_EXIT_ERROR(clGetEventInfo(e[i],
                                         CL_EVENT_COMMAND_EXECUTION_STATUS,
                                         sizeof s, &s, 0),
                          "clGetEventInfo failed");
            CL_EXIT_ERROR(
                (s == CL_QUEUED || s == CL_SUBMITTED) ? CL_SUCCESS : -1,
                "clGetEventInfo %u returned wrong status %d before user event",
                i, (int)s);
        }

        log_info("Setting user event status to unsuccessful result\n");
        CL_EXIT_ERROR(clSetUserEventStatus(u2, -1),
                      "clSetUserEventStatus failed");

        log_info("Waiting for tasks to finish executing\n");
        CL_EXIT_ERROR((clWaitForEvents(N, &e[0]) != CL_SUCCESS) ? CL_SUCCESS
                                                                : -1,
                      "clWaitForEvent succeeded when it should have failed");

        log_info("Checking task status after setting user event status\n");
        for (cl_uint i = 0; i != N; ++i)
        {
            CL_EXIT_ERROR(clGetEventInfo(e[i],
                                         CL_EVENT_COMMAND_EXECUTION_STATUS,
                                         sizeof s, &s, 0),
                          "clGetEventInfo failed");
            CL_EXIT_ERROR((s != CL_QUEUED) ? CL_SUCCESS : -1,
                          "clGetEventInfo %u returned wrong status %04x after "
                          "unsuccessful user event",
                          i, s);
        }

        CL_EXIT_ERROR(clReleaseEvent(u2), "clReleaseEvent failed");

        for (cl_uint i = 0; i != N; ++i)
            CL_EXIT_ERROR(clReleaseEvent(e[i]), "clReleaseEvent failed");

        log_info("Unsuccessful user event case passed.\n");
    }

    clReleaseKernel(k0);
    clReleaseProgram(program);
    clReleaseMemObject(output);

    return 0;
}
