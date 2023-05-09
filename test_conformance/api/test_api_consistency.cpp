//
// Copyright (c) 2020 The Khronos Group Inc.
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
#include "harness/testHarness.h"
#include "harness/deviceInfo.h"

static const char* test_kernel = R"CLC(
__kernel void test(__global int* dst) {
    dst[0] = 0;
}
)CLC";

int test_consistency_svm(cl_device_id deviceID, cl_context context,
                         cl_command_queue queue, int num_elements)
{
    // clGetDeviceInfo, passing CL_DEVICE_SVM_CAPABILITIES:
    // May return 0, indicating that device does not support Shared Virtual
    // Memory.
    cl_int error;

    const size_t allocSize = 16;
    clMemWrapper mem;
    clProgramWrapper program;
    clKernelWrapper kernel;

    cl_device_svm_capabilities svmCaps = 0;
    error = clGetDeviceInfo(deviceID, CL_DEVICE_SVM_CAPABILITIES,
                            sizeof(svmCaps), &svmCaps, NULL);
    test_error(error, "Unable to query CL_DEVICE_SVM_CAPABILITIES");

    if (svmCaps == 0)
    {
        // Test setup:

        mem =
            clCreateBuffer(context, CL_MEM_READ_WRITE, allocSize, NULL, &error);
        test_error(error, "Unable to create test buffer");

        error = create_single_kernel_helper(context, &program, &kernel, 1,
                                            &test_kernel, "test");
        test_error(error, "Unable to create test kernel");

        // clGetMemObjectInfo, passing CL_MEM_USES_SVM_POINTER
        // Returns CL_FALSE if no devices in the context associated with
        // memobj support Shared Virtual Memory.
        cl_bool usesSVMPointer;
        error =
            clGetMemObjectInfo(mem, CL_MEM_USES_SVM_POINTER,
                               sizeof(usesSVMPointer), &usesSVMPointer, NULL);
        test_error(error, "Unable to query CL_MEM_USES_SVM_POINTER");
        test_assert_error(usesSVMPointer == CL_FALSE,
                          "CL_DEVICE_SVM_CAPABILITIES returned 0 but "
                          "CL_MEM_USES_SVM_POINTER did not return CL_FALSE");

        // Check that the SVM APIs can be called.

        // Returns NULL if no devices in context support Shared Virtual Memory.
        void* ptr0 = clSVMAlloc(context, CL_MEM_READ_WRITE, allocSize, 0);
        void* ptr1 = clSVMAlloc(context, CL_MEM_READ_WRITE, allocSize, 0);
        test_assert_error(ptr0 == NULL && ptr1 == NULL,
                          "CL_DEVICE_SVM_CAPABILITIES returned 0 but "
                          "clSVMAlloc returned a non-NULL value");

        // clEnqueueSVMFree, clEnqueueSVMMemcpy, clEnqueueSVMMemFill,
        // clEnqueueSVMMap, clEnqueueSVMUnmap, clEnqueueSVMMigrateMem Returns
        // CL_INVALID_OPERATION if the device associated with command_queue does
        // not support Shared Virtual Memory.

        // These calls purposefully pass bogus pointers to the functions to
        // better test that they are a NOP when SVM is not supported.
        void* bogus0 = (void*)0xDEADBEEF;
        void* bogus1 = (void*)0xDEADDEAD;
        cl_uint pattern = 0xAAAAAAAA;
        error = clEnqueueSVMMemFill(queue, bogus0, &pattern, sizeof(pattern),
                                    allocSize, 0, NULL, NULL);
        test_failure_error(
            error, CL_INVALID_OPERATION,
            "CL_DEVICE_SVM_CAPABILITIES returned 0 but clEnqueueSVMMemFill did "
            "not return CL_INVALID_OPERATION");

        error = clEnqueueSVMMemcpy(queue, CL_TRUE, bogus1, bogus0, allocSize, 0,
                                   NULL, NULL);
        test_failure_error(
            error, CL_INVALID_OPERATION,
            "CL_DEVICE_SVM_CAPABILITIES returned 0 but "
            "clEnqueueSVMMemcpy did not return CL_INVALID_OPERATION");

        error = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_READ, bogus1, allocSize,
                                0, NULL, NULL);
        test_failure_error(
            error, CL_INVALID_OPERATION,
            "CL_DEVICE_SVM_CAPABILITIES returned 0 but "
            "clEnqueueSVMMap did not return CL_INVALID_OPERATION");

        error = clEnqueueSVMUnmap(queue, bogus1, 0, NULL, NULL);
        test_failure_error(
            error, CL_INVALID_OPERATION,
            "CL_DEVICE_SVM_CAPABILITIES returned 0 but "
            "clEnqueueSVMUnmap did not return CL_INVALID_OPERATION");

        error = clEnqueueSVMMigrateMem(queue, 1, (const void**)&bogus1, NULL, 0,
                                       0, NULL, NULL);
        test_failure_error(
            error, CL_INVALID_OPERATION,
            "CL_DEVICE_SVM_CAPABILITIES returned 0 but "
            "clEnqueueSVMMigrateMem did not return CL_INVALID_OPERATION");

        // If the enqueue calls above did not return errors, a clFinish would be
        // needed here to ensure the SVM operations are complete before freeing
        // the SVM pointers.

        clSVMFree(context, bogus0);
        error = clEnqueueSVMFree(queue, 1, &bogus0, NULL, NULL, 0, NULL, NULL);
        test_failure_error(
            error, CL_INVALID_OPERATION,
            "CL_DEVICE_SVM_CAPABILITIES returned 0 but "
            "clEnqueueSVMFree did not return CL_INVALID_OPERATION");

        // If the enqueue calls above did not return errors, a clFinish should
        // be included here to ensure the enqueued SVM free is complete.

        // clSetKernelArgSVMPointer, clSetKernelExecInfo
        // Returns CL_INVALID_OPERATION if no devices in the context associated
        // with kernel support Shared Virtual Memory.

        error = clSetKernelArgSVMPointer(kernel, 0, NULL);
        test_failure_error(
            error, CL_INVALID_OPERATION,
            "CL_DEVICE_SVM_CAPABILITIES returned 0 but "
            "clSetKernelArgSVMPointer did not return CL_INVALID_OPERATION");

        error =
            clSetKernelExecInfo(kernel, CL_KERNEL_EXEC_INFO_SVM_PTRS, 0, NULL);
        test_failure_error(
            error, CL_INVALID_OPERATION,
            "CL_DEVICE_SVM_CAPABILITIES returned 0 but "
            "clSetKernelExecInfo did not return CL_INVALID_OPERATION");
    }

    return TEST_PASS;
}

static int check_atomic_capabilities(cl_device_atomic_capabilities atomicCaps,
                                     cl_device_atomic_capabilities requiredCaps)
{
    if ((atomicCaps & requiredCaps) != requiredCaps)
    {
        log_error("Atomic capabilities %llx is missing support for at least "
                  "one required capability %llx!\n",
                  atomicCaps, requiredCaps);
        return TEST_FAIL;
    }

    if ((atomicCaps & CL_DEVICE_ATOMIC_SCOPE_ALL_DEVICES) != 0
        && (atomicCaps & CL_DEVICE_ATOMIC_SCOPE_DEVICE) == 0)
    {
        log_error("Support for CL_DEVICE_ATOMIC_SCOPE_ALL_DEVICES requires "
                  "support for CL_DEVICE_ATOMIC_SCOPE_DEVICE!\n");
        return TEST_FAIL;
    }

    if ((atomicCaps & CL_DEVICE_ATOMIC_SCOPE_DEVICE) != 0
        && (atomicCaps & CL_DEVICE_ATOMIC_SCOPE_WORK_GROUP) == 0)
    {
        log_error("Support for CL_DEVICE_ATOMIC_SCOPE_DEVICE requires "
                  "support for CL_DEVICE_ATOMIC_SCOPE_WORK_GROUP!\n");
        return TEST_FAIL;
    }

    if ((atomicCaps & CL_DEVICE_ATOMIC_ORDER_SEQ_CST) != 0
        && (atomicCaps & CL_DEVICE_ATOMIC_ORDER_ACQ_REL) == 0)
    {
        log_error("Support for CL_DEVICE_ATOMIC_ORDER_SEQ_CST requires "
                  "support for CL_DEVICE_ATOMIC_ORDER_ACQ_REL!\n");
        return TEST_FAIL;
    }

    if ((atomicCaps & CL_DEVICE_ATOMIC_ORDER_ACQ_REL) != 0
        && (atomicCaps & CL_DEVICE_ATOMIC_ORDER_RELAXED) == 0)
    {
        log_error("Support for CL_DEVICE_ATOMIC_ORDER_ACQ_REL requires "
                  "support for CL_DEVICE_ATOMIC_ORDER_RELAXED!\n");
        return TEST_FAIL;
    }

    return TEST_PASS;
}

int test_consistency_memory_model(cl_device_id deviceID, cl_context context,
                                  cl_command_queue queue, int num_elements)
{
    cl_int error;
    cl_device_atomic_capabilities atomicCaps = 0;

    error = clGetDeviceInfo(deviceID, CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES,
                            sizeof(atomicCaps), &atomicCaps, NULL);
    test_error(error, "Unable to query CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES");

    error = check_atomic_capabilities(atomicCaps,
                                      CL_DEVICE_ATOMIC_ORDER_RELAXED
                                          | CL_DEVICE_ATOMIC_SCOPE_WORK_GROUP);
    if (error == TEST_FAIL)
    {
        log_error("Checks failed for CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES\n");
        return error;
    }

    error = clGetDeviceInfo(deviceID, CL_DEVICE_ATOMIC_FENCE_CAPABILITIES,
                            sizeof(atomicCaps), &atomicCaps, NULL);
    test_error(error, "Unable to query CL_DEVICE_ATOMIC_FENCE_CAPABILITIES");

    error = check_atomic_capabilities(atomicCaps,
                                      CL_DEVICE_ATOMIC_ORDER_RELAXED
                                          | CL_DEVICE_ATOMIC_ORDER_ACQ_REL
                                          | CL_DEVICE_ATOMIC_SCOPE_WORK_GROUP);
    if (error == TEST_FAIL)
    {
        log_error("Checks failed for CL_DEVICE_ATOMIC_FENCE_CAPABILITIES\n");
        return error;
    }

    return TEST_PASS;
}

int test_consistency_device_enqueue(cl_device_id deviceID, cl_context context,
                                    cl_command_queue queue, int num_elements)
{
    // clGetDeviceInfo, passing CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES
    // May return 0, indicating that device does not support Device-Side Enqueue
    // and On-Device Queues.
    cl_int error;

    cl_device_device_enqueue_capabilities dseCaps = 0;
    error = clGetDeviceInfo(deviceID, CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES,
                            sizeof(dseCaps), &dseCaps, NULL);
    test_error(error, "Unable to query CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES");

    if (dseCaps == 0)
    {
        // clGetDeviceInfo, passing CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES
        // Returns 0 if device does not support Device-Side Enqueue and
        // On-Device Queues.

        cl_command_queue_properties devQueueProps = 0;
        error = clGetDeviceInfo(deviceID, CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES,
                                sizeof(devQueueProps), &devQueueProps, NULL);
        test_error(error,
                   "Unable to query CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES");
        test_assert_error(
            devQueueProps == 0,
            "CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES returned 0 but "
            "CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES returned a non-zero value");

        // clGetDeviceInfo, passing
        // CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE,
        // CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE,
        // CL_DEVICE_MAX_ON_DEVICE_QUEUES, or
        // CL_DEVICE_MAX_ON_DEVICE_EVENTS
        // Returns 0 if device does not support Device-Side Enqueue and
        // On-Device Queues.

        cl_uint u = 0;

        error =
            clGetDeviceInfo(deviceID, CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE,
                            sizeof(u), &u, NULL);
        test_error(error,
                   "Unable to query CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE");
        test_assert_error(u == 0,
                          "CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES returned 0 "
                          "but CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE "
                          "returned a non-zero value");

        error = clGetDeviceInfo(deviceID, CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE,
                                sizeof(u), &u, NULL);
        test_error(error, "Unable to query CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE");
        test_assert_error(
            u == 0,
            "CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES returned 0 but "
            "CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE returned a non-zero value");

        error = clGetDeviceInfo(deviceID, CL_DEVICE_MAX_ON_DEVICE_QUEUES,
                                sizeof(u), &u, NULL);
        test_error(error, "Unable to query CL_DEVICE_MAX_ON_DEVICE_QUEUES");
        test_assert_error(
            u == 0,
            "CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES returned 0 but "
            "CL_DEVICE_MAX_ON_DEVICE_QUEUES returned a non-zero value");

        error = clGetDeviceInfo(deviceID, CL_DEVICE_MAX_ON_DEVICE_EVENTS,
                                sizeof(u), &u, NULL);
        test_error(error, "Unable to query CL_DEVICE_MAX_ON_DEVICE_EVENTS");
        test_assert_error(
            u == 0,
            "CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES returned 0 but "
            "CL_DEVICE_MAX_ON_DEVICE_EVENTS returned a non-zero value");

        // clGetCommandQueueInfo, passing CL_QUEUE_SIZE
        // Returns CL_INVALID_COMMAND_QUEUE since command_queue cannot be a
        // valid device command-queue.

        error =
            clGetCommandQueueInfo(queue, CL_QUEUE_SIZE, sizeof(u), &u, NULL);
        test_failure_error(
            error, CL_INVALID_COMMAND_QUEUE,
            "CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES returned 0 but "
            "CL_QUEUE_SIZE did not return CL_INVALID_COMMAND_QUEUE");

        cl_command_queue q = NULL;
        error = clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE_DEFAULT, sizeof(q),
                                      &q, NULL);
        test_error(error, "Unable to query CL_QUEUE_DEVICE_DEFAULT");
        test_assert_error(
            q == NULL,
            "CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES returned 0 but "
            "CL_QUEUE_DEVICE_DEFAULT returned a non-NULL value");

        // clSetDefaultDeviceCommandQueue
        // Returns CL_INVALID_OPERATION if device does not support On-Device
        // Queues.
        error = clSetDefaultDeviceCommandQueue(context, deviceID, NULL);
        test_failure_error(error, CL_INVALID_OPERATION,
                           "CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES returned 0 "
                           "but clSetDefaultDeviceCommandQueue did not return "
                           "CL_INVALID_OPERATION");
    }
    else
    {
        if ((dseCaps & CL_DEVICE_QUEUE_REPLACEABLE_DEFAULT) == 0)
        {
            // clSetDefaultDeviceCommandQueue
            // Returns CL_INVALID_OPERATION if device does not support a
            // replaceable default On-Device Queue.
            error = clSetDefaultDeviceCommandQueue(context, deviceID, NULL);
            test_failure_error(
                error, CL_INVALID_OPERATION,
                "CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES did not "
                "include CL_DEVICE_QUEUE_REPLACEABLE_DEFAULT but "
                "clSetDefaultDeviceCommandQueue did not return "
                "CL_INVALID_OPERATION");
        }

        // If CL_DEVICE_QUEUE_REPLACEABLE_DEFAULT is set,
        // CL_DEVICE_QUEUE_SUPPORTED must also be set.
        if ((dseCaps & CL_DEVICE_QUEUE_REPLACEABLE_DEFAULT) != 0
            && (dseCaps & CL_DEVICE_QUEUE_SUPPORTED) == 0)
        {
            log_error("DEVICE_QUEUE_REPLACEABLE_DEFAULT is set but "
                      "DEVICE_QUEUE_SUPPORTED is not set\n");
            return TEST_FAIL;
        }

        // Devices that set CL_DEVICE_QUEUE_SUPPORTED must also return CL_TRUE
        // for CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT.
        if ((dseCaps & CL_DEVICE_QUEUE_SUPPORTED) != 0)
        {
            cl_bool b;
            error = clGetDeviceInfo(deviceID,
                                    CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT,
                                    sizeof(b), &b, NULL);
            test_error(
                error,
                "Unable to query CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT");
            test_assert_error(
                b == CL_TRUE,
                "DEVICE_QUEUE_SUPPORTED is set but "
                "CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT returned CL_FALSE");
        }
    }

    return TEST_PASS;
}

int test_consistency_pipes(cl_device_id deviceID, cl_context context,
                           cl_command_queue queue, int num_elements)
{
    // clGetDeviceInfo, passing CL_DEVICE_PIPE_SUPPORT
    // May return CL_FALSE, indicating that device does not support Pipes.
    cl_int error;

    cl_bool pipeSupport = CL_FALSE;
    error = clGetDeviceInfo(deviceID, CL_DEVICE_PIPE_SUPPORT,
                            sizeof(pipeSupport), &pipeSupport, NULL);
    test_error(error, "Unable to query CL_DEVICE_PIPE_SUPPORT");

    if (pipeSupport == CL_FALSE)
    {
        // clGetDeviceInfo, passing
        // CL_DEVICE_MAX_PIPE_ARGS,
        // CL_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS, or
        // CL_DEVICE_PIPE_MAX_PACKET_SIZE
        // Returns 0 if device does not support Pipes.

        cl_uint u = 0;

        error = clGetDeviceInfo(deviceID, CL_DEVICE_MAX_PIPE_ARGS, sizeof(u),
                                &u, NULL);
        test_error(error, "Unable to query CL_DEVICE_MAX_PIPE_ARGS");
        test_assert_error(u == 0,
                          "CL_DEVICE_PIPE_SUPPORT returned CL_FALSE, but "
                          "CL_DEVICE_MAX_PIPE_ARGS returned a non-zero value");

        error =
            clGetDeviceInfo(deviceID, CL_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS,
                            sizeof(u), &u, NULL);
        test_error(error,
                   "Unable to query CL_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS");
        test_assert_error(u == 0,
                          "CL_DEVICE_PIPE_SUPPORT returned CL_FALSE, but "
                          "CL_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS returned "
                          "a non-zero value");

        error = clGetDeviceInfo(deviceID, CL_DEVICE_PIPE_MAX_PACKET_SIZE,
                                sizeof(u), &u, NULL);
        test_error(error, "Unable to query CL_DEVICE_PIPE_MAX_PACKET_SIZE");
        test_assert_error(
            u == 0,
            "CL_DEVICE_PIPE_SUPPORT returned CL_FALSE, but "
            "CL_DEVICE_PIPE_MAX_PACKET_SIZE returned a non-zero value");

        // clCreatePipe
        // Returns CL_INVALID_OPERATION if no devices in context support Pipes.
        clMemWrapper mem =
            clCreatePipe(context, CL_MEM_HOST_NO_ACCESS, 4, 4, NULL, &error);
        test_failure_error(error, CL_INVALID_OPERATION,
                           "CL_DEVICE_PIPE_SUPPORT returned CL_FALSE but "
                           "clCreatePipe did not return CL_INVALID_OPERATION");

        // clGetPipeInfo
        // Returns CL_INVALID_MEM_OBJECT since pipe cannot be a valid pipe
        // object.
        clMemWrapper not_a_pipe =
            clCreateBuffer(context, CL_MEM_READ_WRITE, 4, NULL, &error);
        test_error(error, "Unable to create non-pipe buffer");

        error =
            clGetPipeInfo(not_a_pipe, CL_PIPE_PACKET_SIZE, sizeof(u), &u, NULL);
        test_failure_error(
            error, CL_INVALID_MEM_OBJECT,
            "CL_DEVICE_PIPE_SUPPORT returned CL_FALSE but clGetPipeInfo did "
            "not return CL_INVALID_MEM_OBJECT");
    }
    else
    {
        // Devices that support pipes must also return CL_TRUE
        // for CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT.
        cl_bool b;
        error =
            clGetDeviceInfo(deviceID, CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT,
                            sizeof(b), &b, NULL);
        test_error(error,
                   "Unable to query CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT");
        test_assert_error(
            b == CL_TRUE,
            "CL_DEVICE_PIPE_SUPPORT returned CL_TRUE but "
            "CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT returned CL_FALSE");
    }

    return TEST_PASS;
}

int test_consistency_progvar(cl_device_id deviceID, cl_context context,
                             cl_command_queue queue, int num_elements)
{
    // clGetDeviceInfo, passing CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE
    // May return 0, indicating that device does not support Program Scope
    // Global Variables.
    cl_int error;

    clProgramWrapper program;
    clKernelWrapper kernel;

    size_t maxGlobalVariableSize = 0;
    error = clGetDeviceInfo(deviceID, CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE,
                            sizeof(maxGlobalVariableSize),
                            &maxGlobalVariableSize, NULL);
    test_error(error, "Unable to query CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE");

    if (maxGlobalVariableSize == 0)
    {
        // Test setup:

        error = create_single_kernel_helper(context, &program, &kernel, 1,
                                            &test_kernel, "test");
        test_error(error, "Unable to create test kernel");

        size_t sz = SIZE_MAX;

        // clGetDeviceInfo, passing
        // CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE
        // Returns 0 if device does not support Program Scope Global Variables.

        error = clGetDeviceInfo(deviceID,
                                CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE,
                                sizeof(sz), &sz, NULL);
        test_error(
            error,
            "Unable to query CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE");
        test_assert_error(
            sz == 0,
            "CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE returned 0 but "
            "CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE returned a "
            "non-zero value");

        // clGetProgramBuildInfo, passing
        // CL_PROGRAM_BUILD_GLOBAL_VARIABLE_TOTAL_SIZE
        // Returns 0 if device does not support Program Scope Global Variables.

        error = clGetProgramBuildInfo(
            program, deviceID, CL_PROGRAM_BUILD_GLOBAL_VARIABLE_TOTAL_SIZE,
            sizeof(sz), &sz, NULL);
        test_error(
            error,
            "Unable to query CL_PROGRAM_BUILD_GLOBAL_VARIABLE_TOTAL_SIZE");
        test_assert_error(sz == 0,
                          "CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE returned 0 "
                          "but CL_PROGRAM_BUILD_GLOBAL_VARIABLE_TOTAL_SIZE "
                          "returned a non-zero value");
    }

    return TEST_PASS;
}

int test_consistency_non_uniform_work_group(cl_device_id deviceID,
                                            cl_context context,
                                            cl_command_queue queue,
                                            int num_elements)
{
    // clGetDeviceInfo, passing CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT:
    // May return CL_FALSE, indicating that device does not support Non-Uniform
    // Work Groups.
    cl_int error;

    const size_t allocSize = 16;
    clMemWrapper mem;
    clProgramWrapper program;
    clKernelWrapper kernel;

    cl_bool nonUniformWorkGroupSupport = CL_FALSE;
    error = clGetDeviceInfo(deviceID, CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT,
                            sizeof(nonUniformWorkGroupSupport),
                            &nonUniformWorkGroupSupport, NULL);
    test_error(error,
               "Unable to query CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT");

    if (nonUniformWorkGroupSupport == CL_FALSE)
    {
        // Test setup:

        mem =
            clCreateBuffer(context, CL_MEM_READ_WRITE, allocSize, NULL, &error);
        test_error(error, "Unable to create test buffer");

        error = create_single_kernel_helper(context, &program, &kernel, 1,
                                            &test_kernel, "test");
        test_error(error, "Unable to create test kernel");

        error = clSetKernelArg(kernel, 0, sizeof(mem), &mem);

        // clEnqueueNDRangeKernel
        // Behaves as though Non-Uniform Work Groups were not enabled for
        // kernel, if the device associated with command_queue does not support
        // Non-Uniform Work Groups.

        size_t global_work_size[] = { 3, 3, 3 };
        size_t local_work_size[] = { 2, 2, 2 };

        // First, check that a NULL local work size succeeds.
        error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_work_size,
                                       NULL, 0, NULL, NULL);
        test_error(error,
                   "Unable to enqueue kernel with a NULL local work size");

        error = clFinish(queue);
        test_error(error, "Error calling clFinish after NULL local work size");

        // 1D non-uniform work group:
        error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_work_size,
                                       local_work_size, 0, NULL, NULL);
        test_failure_error(
            error, CL_INVALID_WORK_GROUP_SIZE,
            "CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT returned CL_FALSE but 1D "
            "clEnqueueNDRangeKernel did not return CL_INVALID_WORK_GROUP_SIZE");

        // 2D non-uniform work group:
        global_work_size[0] = local_work_size[0];
        error = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size,
                                       local_work_size, 0, NULL, NULL);
        test_failure_error(
            error, CL_INVALID_WORK_GROUP_SIZE,
            "CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT returned CL_FALSE but 2D "
            "clEnqueueNDRangeKernel did not return CL_INVALID_WORK_GROUP_SIZE");

        // 3D non-uniform work group:
        global_work_size[1] = local_work_size[1];
        error = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size,
                                       local_work_size, 0, NULL, NULL);
        test_failure_error(
            error, CL_INVALID_WORK_GROUP_SIZE,
            "CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT returned CL_FALSE but 3D "
            "clEnqueueNDRangeKernel did not return CL_INVALID_WORK_GROUP_SIZE");
    }

    return TEST_PASS;
}

int test_consistency_read_write_images(cl_device_id deviceID,
                                       cl_context context,
                                       cl_command_queue queue, int num_elements)
{
    // clGetDeviceInfo, passing
    // CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS May return 0,
    // indicating that device does not support Read-Write Images.
    cl_int error;

    cl_uint maxReadWriteImageArgs = 0;
    error = clGetDeviceInfo(deviceID, CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS,
                            sizeof(maxReadWriteImageArgs),
                            &maxReadWriteImageArgs, NULL);
    test_error(error,
               "Unable to query "
               "CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS");

    // clGetSupportedImageFormats, passing
    // CL_MEM_KERNEL_READ_AND_WRITE
    // Returns an empty set (such as num_image_formats equal to 0), indicating
    // that no image formats are supported for reading and writing in the same
    // kernel, if no devices in context support Read-Write Images.

    cl_uint totalReadWriteImageFormats = 0;

    const cl_mem_object_type image_types[] = {
        CL_MEM_OBJECT_IMAGE1D,       CL_MEM_OBJECT_IMAGE1D_BUFFER,
        CL_MEM_OBJECT_IMAGE2D,       CL_MEM_OBJECT_IMAGE3D,
        CL_MEM_OBJECT_IMAGE1D_ARRAY, CL_MEM_OBJECT_IMAGE2D_ARRAY,
    };
    for (size_t i = 0; i < ARRAY_SIZE(image_types); i++)
    {
        cl_uint numImageFormats = 0;
        error = clGetSupportedImageFormats(
            context, CL_MEM_KERNEL_READ_AND_WRITE, image_types[i], 0, NULL,
            &numImageFormats);
        test_error(error,
                   "Unable to query number of CL_MEM_KERNEL_READ_AND_WRITE "
                   "image formats");

        totalReadWriteImageFormats += numImageFormats;
    }

    if (maxReadWriteImageArgs == 0)
    {
        test_assert_error(
            totalReadWriteImageFormats == 0,
            "CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS returned 0 "
            "but clGetSupportedImageFormats(CL_MEM_KERNEL_READ_AND_WRITE) "
            "returned a non-empty set");
    }
    else
    {
        test_assert_error(
            totalReadWriteImageFormats != 0,
            "CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS is non-zero "
            "but clGetSupportedImageFormats(CL_MEM_KERNEL_READ_AND_WRITE) "
            "returned an empty set");
    }

    return TEST_PASS;
}

int test_consistency_2d_image_from_buffer(cl_device_id deviceID,
                                          cl_context context,
                                          cl_command_queue queue,
                                          int num_elements)
{
    // clGetDeviceInfo, passing CL_DEVICE_IMAGE_PITCH_ALIGNMENT or
    // CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT
    // May return 0, indicating that device does not support Creating a 2D Image
    // from a Buffer.
    cl_int error;

    const cl_image_format imageFormat = { CL_RGBA, CL_UNORM_INT8 };
    const size_t imageDim = 2;
    const size_t elementSize = 4;
    const size_t bufferSize = imageDim * imageDim * elementSize;

    clMemWrapper buffer;
    clMemWrapper image;

    cl_uint imagePitchAlignment = 0;
    error = clGetDeviceInfo(deviceID, CL_DEVICE_IMAGE_PITCH_ALIGNMENT,
                            sizeof(imagePitchAlignment), &imagePitchAlignment,
                            NULL);
    test_error(error,
               "Unable to query "
               "CL_DEVICE_IMAGE_PITCH_ALIGNMENT");

    cl_uint imageBaseAddressAlignment = 0;
    error = clGetDeviceInfo(deviceID, CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT,
                            sizeof(imageBaseAddressAlignment),
                            &imageBaseAddressAlignment, NULL);
    test_error(error,
               "Unable to query "
               "CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT");

    bool supports_cl_khr_image2d_from_buffer =
        is_extension_available(deviceID, "cl_khr_image2d_from_buffer");

    if (imagePitchAlignment == 0 || imageBaseAddressAlignment == 0)
    {
        // This probably means that Creating a 2D Image from a Buffer is not
        // supported.

        // Test setup:
        buffer =
            clCreateBuffer(context, CL_MEM_READ_ONLY, bufferSize, NULL, &error);
        test_error(error, "Unable to create test buffer");

        // Check that both queries return zero:
        test_assert_error(
            imagePitchAlignment == 0,
            "CL_DEVICE_IMAGE_PITCH_ALIGNMENT returned a non-zero value but "
            "CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT returned 0");
        test_assert_error(
            imageBaseAddressAlignment == 0,
            "CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT returned a non-zero value "
            "but CL_DEVICE_IMAGE_PITCH_ALIGNMENT returned 0");

        // clGetDeviceInfo, passing CL_DEVICE_EXTENSIONS
        // Will not describe support for the cl_khr_image2d_from_buffer
        // extension if device does not support Creating a 2D Image from a
        // Buffer.
        test_assert_error(supports_cl_khr_image2d_from_buffer == false,
                          "Device does not support Creating a 2D Image from a "
                          "Buffer but does support cl_khr_image2d_from_buffer");

        // clCreateImage or clCreateImageWithProperties, passing image_type
        // equal to CL_MEM_OBJECT_IMAGE2D and mem_object not equal to
        // NULL
        // Returns CL_INVALID_OPERATION if no devices in context support
        // Creating a 2D Image from a Buffer.

        cl_image_desc imageDesc = { 0 };
        imageDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
        imageDesc.image_width = imageDim;
        imageDesc.image_height = imageDim;
        imageDesc.mem_object = buffer;

        image = clCreateImage(context, CL_MEM_READ_ONLY, &imageFormat,
                              &imageDesc, NULL, &error);
        test_failure_error(
            error, CL_INVALID_OPERATION,
            "Device does not support Creating a 2D Image from a "
            "Buffer but clCreateImage did not return CL_INVALID_OPERATION");

        image =
            clCreateImageWithProperties(context, NULL, CL_MEM_READ_ONLY,
                                        &imageFormat, &imageDesc, NULL, &error);
        test_failure_error(error, CL_INVALID_OPERATION,
                           "Device does not support Creating a 2D Image from a "
                           "Buffer but clCreateImageWithProperties did not "
                           "return CL_INVALID_OPERATION");
    }
    else
    {
        test_assert_error(supports_cl_khr_image2d_from_buffer,
                          "Device supports Creating a 2D Image from a Buffer "
                          "but does not support cl_khr_image2d_from_buffer");
    }

    return TEST_PASS;
}

// Nothing needed for sRGB Images:
// All of the sRGB Image Channel Orders (such as CL_​sRGBA) are optional for
// devices supporting OpenCL 3.0.

int test_consistency_depth_images(cl_device_id deviceID, cl_context context,
                                  cl_command_queue queue, int num_elements)
{
    // The CL_DEPTH Image Channel Order is optional for devices supporting
    // OpenCL 3.0.
    cl_int error;

    cl_uint totalDepthImageFormats = 0;

    const cl_mem_flags mem_flags[] = {
        CL_MEM_WRITE_ONLY,
        CL_MEM_READ_WRITE,
        CL_MEM_KERNEL_READ_AND_WRITE,
    };
    for (size_t i = 0; i < ARRAY_SIZE(mem_flags); i++)
    {
        cl_uint numImageFormats = 0;
        error = clGetSupportedImageFormats(context, mem_flags[i],
                                           CL_MEM_OBJECT_IMAGE2D, 0, NULL,
                                           &numImageFormats);
        test_error(
            error,
            "Unable to query number of CL_MEM_OBJECT_IMAGE2D image formats");

        std::vector<cl_image_format> imageFormats(numImageFormats);
        error = clGetSupportedImageFormats(
            context, mem_flags[i], CL_MEM_OBJECT_IMAGE2D, imageFormats.size(),
            imageFormats.data(), NULL);
        test_error(error,
                   "Unable to query CL_MEM_OBJECT_IMAGE2D image formats");
        for (auto& imageFormat : imageFormats)
        {
            if (imageFormat.image_channel_order == CL_DEPTH)
            {
                totalDepthImageFormats++;
            }
        }
    }

    bool supports_cl_khr_depth_images =
        is_extension_available(deviceID, "cl_khr_depth_images");

    if (totalDepthImageFormats == 0)
    {
        test_assert_error(supports_cl_khr_depth_images == false,
                          "Device does not support Depth Images but does "
                          "support cl_khr_depth_images");
    }
    else
    {
        test_assert_error(supports_cl_khr_depth_images,
                          "Device supports Depth Images but does not support "
                          "cl_khr_depth_images");
    }

    return TEST_PASS;
}

int test_consistency_device_and_host_timer(cl_device_id deviceID,
                                           cl_context context,
                                           cl_command_queue queue,
                                           int num_elements)
{
    // clGetPlatformInfo, passing CL_PLATFORM_HOST_TIMER_RESOLUTION
    // May return 0, indicating that platform does not support Device and Host
    // Timer Synchronization.
    cl_int error;

    cl_platform_id platform = NULL;
    error = clGetDeviceInfo(deviceID, CL_DEVICE_PLATFORM, sizeof(platform),
                            &platform, NULL);
    test_error(error, "Unable to query CL_DEVICE_PLATFORM");

    cl_ulong hostTimerResolution = 0;
    error = clGetPlatformInfo(platform, CL_PLATFORM_HOST_TIMER_RESOLUTION,
                              sizeof(hostTimerResolution), &hostTimerResolution,
                              NULL);
    test_error(error, "Unable to query CL_PLATFORM_HOST_TIMER_RESOLUTION");

    if (hostTimerResolution == 0)
    {
        // clGetDeviceAndHostTimer, clGetHostTimer
        // Returns CL_INVALID_OPERATION if the platform associated with device
        // does not support Device and Host Timer Synchronization.

        cl_ulong dt = 0;
        cl_ulong ht = 0;

        error = clGetDeviceAndHostTimer(deviceID, &dt, &ht);
        test_failure_error(
            error, CL_INVALID_OPERATION,
            "CL_PLATFORM_HOST_TIMER_RESOLUTION returned 0 but "
            "clGetDeviceAndHostTimer did not return CL_INVALID_OPERATION");

        error = clGetHostTimer(deviceID, &ht);
        test_failure_error(
            error, CL_INVALID_OPERATION,
            "CL_PLATFORM_HOST_TIMER_RESOLUTION returned 0 but "
            "clGetHostTimer did not return CL_INVALID_OPERATION");
    }

    return TEST_PASS;
}

int test_consistency_il_programs(cl_device_id deviceID, cl_context context,
                                 cl_command_queue queue, int num_elements)
{
    // clGetDeviceInfo, passing CL_DEVICE_IL_VERSION or
    // CL_DEVICE_ILS_WITH_VERSION
    // May return an empty string and empty array, indicating that device does
    // not support Intermediate Language Programs.
    cl_int error;

    clProgramWrapper program;
    clKernelWrapper kernel;

    // Even if the device does not support Intermediate Language Programs the
    // size of the string query should not be zero.
    size_t sz = SIZE_MAX;
    error = clGetDeviceInfo(deviceID, CL_DEVICE_IL_VERSION, 0, NULL, &sz);
    test_error(error, "Unable to query CL_DEVICE_IL_VERSION");
    test_assert_error(sz != 0,
                      "CL_DEVICE_IL_VERSION should return a non-zero size");

    std::string ilVersion = get_device_il_version_string(deviceID);

    error = clGetDeviceInfo(deviceID, CL_DEVICE_ILS_WITH_VERSION, 0, NULL, &sz);
    test_error(error, "Unable to query CL_DEVICE_ILS_WITH_VERSION");

    if (ilVersion == "" || sz == 0)
    {
        // This probably means that Intermediate Language Programs are not
        // supported.

        // Check that both queries are consistent:
        test_assert_error(
            ilVersion == "",
            "CL_DEVICE_IL_VERSION returned a non-empty string but "
            "CL_DEVICE_ILS_WITH_VERSION returned no supported ILs");

        test_assert_error(sz == 0,
                          "CL_DEVICE_ILS_WITH_VERSION returned supported ILs "
                          "but CL_DEVICE_IL_VERSION returned an empty string");

        bool supports_cl_khr_il_program =
            is_extension_available(deviceID, "cl_khr_il_program");
        test_assert_error(supports_cl_khr_il_program == false,
                          "Device does not support IL Programs but does "
                          "support cl_khr_il_program");

        // Test setup:

        error = create_single_kernel_helper(context, &program, &kernel, 1,
                                            &test_kernel, "test");
        test_error(error, "Unable to create test kernel");

        // clGetProgramInfo, passing CL_PROGRAM_IL
        // Returns an empty buffer (such as param_value_size_ret equal to 0) if
        // no devices in the context associated with program support
        // Intermediate Language Programs.

        error = clGetProgramInfo(program, CL_PROGRAM_IL, 0, NULL, &sz);
        test_error(error, "Unable to query CL_PROGRAM_IL");
        test_assert_error(sz == 0,
                          "Device does not support IL Programs but "
                          "CL_PROGRAM_IL returned a non-zero size");

        // clCreateProgramWithIL
        // Returns CL_INVALID_OPERATION if no devices in context support
        // Intermediate Language Programs.

        cl_uint bogus = 0xDEADBEEF;
        clProgramWrapper ilProgram =
            clCreateProgramWithIL(context, &bogus, sizeof(bogus), &error);
        test_failure_error(
            error, CL_INVALID_OPERATION,
            "Device does not support IL Programs but clCreateProgramWithIL did "
            "not return CL_INVALID_OPERATION");

        // clSetProgramSpecializationConstant
        // Returns CL_INVALID_OPERATION if no devices associated with program
        // support Intermediate Language Programs.

        cl_uint specConst = 42;
        error = clSetProgramSpecializationConstant(
            program, 0, sizeof(specConst), &specConst);
        test_failure_error(error, CL_INVALID_OPERATION,
                           "Device does not support IL Programs but "
                           "clSetProgramSpecializationConstant did not return "
                           "CL_INVALID_OPERATION");
    }

    return TEST_PASS;
}

int test_consistency_subgroups(cl_device_id deviceID, cl_context context,
                               cl_command_queue queue, int num_elements)
{
    // clGetDeviceInfo, passing CL_DEVICE_MAX_NUM_SUB_GROUPS
    // May return 0, indicating that device does not support Subgroups.
    cl_int error;

    clProgramWrapper program;
    clKernelWrapper kernel;

    cl_uint maxNumSubGroups = 0;
    error = clGetDeviceInfo(deviceID, CL_DEVICE_MAX_NUM_SUB_GROUPS,
                            sizeof(maxNumSubGroups), &maxNumSubGroups, NULL);
    test_error(error, "Unable to query CL_DEVICE_MAX_NUM_SUB_GROUPS");

    if (maxNumSubGroups == 0)
    {
        // Test setup:

        error = create_single_kernel_helper(context, &program, &kernel, 1,
                                            &test_kernel, "test");
        test_error(error, "Unable to create test kernel");

        // clGetDeviceInfo, passing
        // CL_DEVICE_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS
        // Returns CL_FALSE if device does not support Subgroups.

        cl_bool ifp = CL_FALSE;
        error = clGetDeviceInfo(
            deviceID, CL_DEVICE_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS,
            sizeof(ifp), &ifp, NULL);
        test_error(
            error,
            "Unable to query CL_DEVICE_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS");
        test_assert_error(ifp == CL_FALSE,
                          "Device does not support Subgroups but "
                          "CL_DEVICE_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS "
                          "did not return CL_FALSE");

        // clGetDeviceInfo, passing CL_DEVICE_EXTENSIONS
        // Will not describe support for the cl_khr_subgroups extension if
        // device does not support Subgroups.

        bool supports_cl_khr_subgroups =
            is_extension_available(deviceID, "cl_khr_subgroups");
        test_assert_error(supports_cl_khr_subgroups == false,
                          "Device does not support Subgroups but does "
                          "support cl_khr_subgroups");

        // clGetKernelSubGroupInfo
        // Returns CL_INVALID_OPERATION if device does not support Subgroups.

        size_t sz = SIZE_MAX;
        error = clGetKernelSubGroupInfo(kernel, deviceID,
                                        CL_KERNEL_MAX_NUM_SUB_GROUPS, 0, NULL,
                                        sizeof(sz), &sz, NULL);
        test_failure_error(
            error, CL_INVALID_OPERATION,
            "Device does not support Subgroups but clGetKernelSubGroupInfo did "
            "not return CL_INVALID_OPERATION");
    }

    return TEST_PASS;
}

static void CL_CALLBACK program_callback(cl_program, void*) {}

int test_consistency_prog_ctor_dtor(cl_device_id deviceID, cl_context context,
                                    cl_command_queue queue, int num_elements)
{
    cl_int error;

    clProgramWrapper program;
    clKernelWrapper kernel;

    // Test setup:

    error = create_single_kernel_helper(context, &program, &kernel, 1,
                                        &test_kernel, "test");
    test_error(error, "Unable to create test kernel");

    // clGetProgramInfo, passing CL_PROGRAM_SCOPE_GLOBAL_CTORS_PRESENT or
    // CL_PROGRAM_SCOPE_GLOBAL_DTORS_PRESENT
    // Returns CL_FALSE if no devices in the context associated with program
    // support Program Initialization and Clean-Up Kernels.

    cl_bool b = CL_FALSE;

    error = clGetProgramInfo(program, CL_PROGRAM_SCOPE_GLOBAL_CTORS_PRESENT,
                             sizeof(b), &b, NULL);
    test_error(error, "Unable to query CL_PROGRAM_SCOPE_GLOBAL_CTORS_PRESENT");
    test_assert_error(
        b == CL_FALSE,
        "CL_PROGRAM_SCOPE_GLOBAL_CTORS_PRESENT did not return CL_FALSE");

    error = clGetProgramInfo(program, CL_PROGRAM_SCOPE_GLOBAL_DTORS_PRESENT,
                             sizeof(b), &b, NULL);
    test_error(error, "Unable to query CL_PROGRAM_SCOPE_GLOBAL_DTORS_PRESENT");
    test_assert_error(
        b == CL_FALSE,
        "CL_PROGRAM_SCOPE_GLOBAL_DTORS_PRESENT did not return CL_FALSE");

    // clSetProgramReleaseCallback
    // Returns CL_INVALID_OPERATION if no devices in the context associated with
    // program support Program Initialization and Clean-Up Kernels.

    error = clSetProgramReleaseCallback(program, program_callback, NULL);
    test_failure_error(
        error, CL_INVALID_OPERATION,
        "clSetProgramReleaseCallback did not return CL_INVALID_OPERATION");

    return TEST_PASS;
}

int test_consistency_3d_image_writes(cl_device_id deviceID, cl_context context,
                                     cl_command_queue queue, int num_elements)
{
    // clGetSupportedImageFormats, passing CL_MEM_OBJECT_IMAGE3D and one of
    // CL_MEM_WRITE_ONLY, CL_MEM_READ_WRITE, or CL_MEM_KERNEL_READ_AND_WRITE
    // Returns an empty set (such as num_image_formats equal to 0),
    // indicating that no image formats are supported for writing to 3D
    // image objects, if no devices in context support Writing to 3D Image
    // Objects.
    cl_int error;

    cl_uint total3DImageWriteFormats = 0;

    const cl_mem_flags mem_flags[] = {
        CL_MEM_WRITE_ONLY,
        CL_MEM_READ_WRITE,
        CL_MEM_KERNEL_READ_AND_WRITE,
    };
    for (size_t i = 0; i < ARRAY_SIZE(mem_flags); i++)
    {
        cl_uint numImageFormats = 0;
        error = clGetSupportedImageFormats(context, mem_flags[i],
                                           CL_MEM_OBJECT_IMAGE3D, 0, NULL,
                                           &numImageFormats);
        test_error(
            error,
            "Unable to query number of CL_MEM_OBJECT_IMAGE3D image formats");

        total3DImageWriteFormats += numImageFormats;
    }

    bool supports_cl_khr_3d_image_writes =
        is_extension_available(deviceID, "cl_khr_3d_image_writes");

    if (total3DImageWriteFormats == 0)
    {
        // clGetDeviceInfo, passing CL_DEVICE_EXTENSIONS
        // Will not describe support for the cl_khr_3d_image_writes extension if
        // device does not support Writing to 3D Image Objects.
        test_assert_error(supports_cl_khr_3d_image_writes == false,
                          "Device does not support Writing to 3D Image Objects "
                          "but does support cl_khr_3d_image_writes");
    }
    else
    {
        test_assert_error(supports_cl_khr_3d_image_writes,
                          "Device supports Writing to 3D Image Objects but "
                          "does not support cl_khr_3d_image_writes");
    }

    return TEST_PASS;
}
