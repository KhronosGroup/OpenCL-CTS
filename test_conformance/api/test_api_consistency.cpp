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

const char* test_kernel = R"CLC(
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
    int error;

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

        cl_uint pattern = 0xAAAAAAAA;
        error = clEnqueueSVMMemFill(queue, ptr0, &pattern, sizeof(pattern),
                                    allocSize, 0, NULL, NULL);
        test_failure_error(
            error, CL_INVALID_OPERATION,
            "CL_DEVICE_SVM_CAPABILITIES returned 0 but clEnqueueSVMMemFill did "
            "not return CL_INVALID_OPERATION");

        error = clEnqueueSVMMemcpy(queue, CL_TRUE, ptr1, ptr0, allocSize, 0,
                                   NULL, NULL);
        test_failure_error(
            error, CL_INVALID_OPERATION,
            "CL_DEVICE_SVM_CAPABILITIES returned 0 but "
            "clEnqueueSVMMemcpy did not return CL_INVALID_OPERATION");

        error = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_READ, ptr1, allocSize, 0,
                                NULL, NULL);
        test_failure_error(
            error, CL_INVALID_OPERATION,
            "CL_DEVICE_SVM_CAPABILITIES returned 0 but "
            "clEnqueueSVMMap did not return CL_INVALID_OPERATION");

        error = clEnqueueSVMUnmap(queue, ptr1, 0, NULL, NULL);
        test_failure_error(
            error, CL_INVALID_OPERATION,
            "CL_DEVICE_SVM_CAPABILITIES returned 0 but "
            "clEnqueueSVMUnmap did not return CL_INVALID_OPERATION");

        // If the enqueue calls above did not return errors, a clFinish would be
        // needed here to ensure the SVM operations are complete before freeing
        // the SVM pointers.

        // These calls to free SVM purposefully passes a bogus pointer to the
        // free function to better test that it they are a NOP when SVM is not
        // supported.
        void* bogus = (void*)0xDEADBEEF;
        clSVMFree(context, bogus);
        error = clEnqueueSVMFree(queue, 1, &bogus, NULL, NULL, 0, NULL, NULL);
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
        log_info("Check: ATOMIC_SCOPE_ALL_DEVICES is supported, but "
                 "ATOMIC_SCOPE_DEVICE is not?\n");
    }

    if ((atomicCaps & CL_DEVICE_ATOMIC_SCOPE_DEVICE) != 0
        && (atomicCaps & CL_DEVICE_ATOMIC_SCOPE_WORK_GROUP) == 0)
    {
        log_info("Check: ATOMIC_SCOPE_DEVICE is supported, but "
                 "ATOMIC_SCOPE_WORK_GROUP is not?\n");
    }

    return TEST_PASS;
}

int test_consistency_memory_model(cl_device_id deviceID, cl_context context,
                                  cl_command_queue queue, int num_elements)
{
    int error;
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
    int error;

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

        // clGetCommandQueueInfo, passing CL_QUEUE_SIZE or
        // CL_QUEUE_DEVICE_DEFAULT
        // Returns 0 or NULL if the device associated with command_queue does
        // not support On-Device Queues.

        error =
            clGetCommandQueueInfo(queue, CL_QUEUE_SIZE, sizeof(u), &u, NULL);
        // TODO: is this a valid query?  See:
        // https://github.com/KhronosGroup/OpenCL-Docs/issues/402
        // test_error(error, "Unable to query CL_QUEUE_SIZE");
        if (error == CL_SUCCESS && u != 0)
        {
            log_error("CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES returned 0 but "
                      "CL_QUEUE_SIZE returned a non-zero value\n");
            return TEST_FAIL;
        }

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
    int error;

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
        error = clGetPipeInfo(mem, CL_PIPE_PACKET_SIZE, sizeof(u), &u, NULL);
        test_failure_error(
            error, CL_INVALID_MEM_OBJECT,
            "CL_DEVICE_PIPE_SUPPORT returned CL_FALSE but "
            "clGetPipeInfo did not return CL_INVALID_MEM_OBJECT");
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
    int error;

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

        size_t sz = 0;

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
    int error;

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

    if (true || nonUniformWorkGroupSupport == CL_FALSE)
    {
        // Test setup:

        mem =
            clCreateBuffer(context, CL_MEM_READ_WRITE, allocSize, NULL, &error);
        test_error(error, "Unable to create test buffer");

        error = create_single_kernel_helper_with_build_options(
            context, &program, &kernel, 1, &test_kernel, "test",
            "-cl-std=CL3.0");
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
