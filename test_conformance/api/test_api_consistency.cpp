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

        // Check that the SVM APIs can be called.
        // It's OK if they return an error.
        void* ptr0;
        ptr0 = clSVMAlloc(context, CL_MEM_READ_WRITE, allocSize, 0);

        void* ptr1;
        ptr1 = clSVMAlloc(context, CL_MEM_READ_WRITE, allocSize, 0);

        cl_uint pattern = 0xAAAAAAAA;
        clEnqueueSVMMemFill(queue, ptr0, &pattern, sizeof(pattern), allocSize,
                            0, NULL, NULL);
        clEnqueueSVMMemcpy(queue, CL_TRUE, ptr1, ptr0, allocSize, 0, NULL,
                           NULL);

        clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_READ, ptr1, allocSize, 0, NULL,
                        NULL);
        clEnqueueSVMUnmap(queue, ptr1, 0, NULL, NULL);

        error = clFinish(queue);
        test_error(error, "Error calling clFinish after SVM operations");

        clSVMFree(context, ptr0);
        clEnqueueSVMFree(queue, 1, &ptr1, NULL, NULL, 0, NULL, NULL);

        error = clFinish(queue);
        test_error(error, "Error calling clFinish after SVM free");
    }

    return TEST_PASS;
}

static int check_atomic_capabilities(cl_device_atomic_capabilities atomicCaps, cl_device_atomic_capabilities requiredCaps)
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
        // It isn't an error to support ALL_DEVICES atomics but not DEVICE
        // atomics, but it is strange.
        log_info("Check: ATOMIC_SCOPE_ALL_DEVICES is supported, but "
                 "ATOMIC_SCOPE_DEVICE is not?\n");
    }

    if ((atomicCaps & CL_DEVICE_ATOMIC_SCOPE_DEVICE) != 0
        && (atomicCaps & CL_DEVICE_ATOMIC_SCOPE_WORK_GROUP) == 0)
    {
        // It isn't an error to support DEVICE atomics but not WORK_GROUP
        // atomics, but it is strange.
        log_info("Check: ATOMIC_SCOPE_DEVICE is supported, but "
                 "ATOMIC_SCOPE_WORK_GROPU is not?\n");
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
        if (devQueueProps != 0)
        {
            // It isn't an error to return nonzero device queue properties, but
            // it is strange.
            log_info("Check: DEVICE_ENQUEUE_CAPABILITIES is zero, but "
                     "QUEUE_ON_DEVICE_PROPERTIES is nonzero?\n");
        }

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
        if (u != 0)
        {
            // It isn't an error to return a nonzero preferred device queue
            // size, but it is strange.
            log_info("Check: DEVICE_ENQUEUE_CAPABILITIES is zero, but "
                     "​QUEUE_ON_DEVICE_PREFERRED_SIZE is nonzero?\n");
        }

        error = clGetDeviceInfo(deviceID, CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE,
                                sizeof(u), &u, NULL);
        test_error(error, "Unable to query CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE");
        if (u != 0)
        {
            // It isn't an error to return a nonzero preferred device queue
            // size, but it is strange.
            log_info("Check: DEVICE_ENQUEUE_CAPABILITIES is zero, but "
                     "QUEUE_ON_DEVICE_MAX_SIZE is nonzero?\n");
        }

        error = clGetDeviceInfo(deviceID, CL_DEVICE_MAX_ON_DEVICE_QUEUES,
                                sizeof(u), &u, NULL);
        test_error(error, "Unable to query CL_DEVICE_MAX_ON_DEVICE_QUEUES");
        if (u != 0)
        {
            // It isn't an error to return a nonzero maximum number of on-device
            // queues, but it is strange.
            log_info("Check: DEVICE_ENQUEUE_CAPABILITIES is zero, but "
                     "MAX_ON_DEVICE_QUEUES is nonzero?\n");
        }

        error = clGetDeviceInfo(deviceID, CL_DEVICE_MAX_ON_DEVICE_EVENTS,
                                sizeof(u), &u, NULL);
        test_error(error, "Unable to query CL_DEVICE_MAX_ON_DEVICE_EVENTS");
        if (u != 0)
        {
            // It isn't an error to return a nonzero maximum number of on-device
            // events, but it is strange.
            log_info("Check: DEVICE_ENQUEUE_CAPABILITIES is zero, but "
                     "MAX_ON_DEVICE_EVENTS is nonzero?\n");
        }

        // clGetCommandQueueInfo, passing CL_QUEUE_SIZE or
        // CL_QUEUE_DEVICE_DEFAULT Returns 0 or NULL if the device associated
        // with command_queue does not support On-Device Queues.

        error =
            clGetCommandQueueInfo(queue, CL_QUEUE_SIZE, sizeof(u), &u, NULL);
        test_error(error, "Unable to query CL_QUEUE_SIZE");

        cl_command_queue q = NULL;
        error = clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE_DEFAULT, sizeof(q),
                                      &q, NULL);
        test_error(error, "Unable to query CL_QUEUE_DEVICE_DEFAULT");

        // Check that clSetDefaultDeviceCommandQueue can be called.
        // It's OK if it returns an error, and in fact it should, since NULL is
        // not a valid device queue.
        clSetDefaultDeviceCommandQueue(context, deviceID, NULL);
    }
    else
    {
        if ((dseCaps & CL_DEVICE_QUEUE_REPLACEABLE_DEFAULT) == 0)
        {
            // Check that clSetDefaultDeviceCommandQueue can be called.
            // It's OK if it returns an error, and in fact it should, since NULL
            // is not a valid device queue.
            clSetDefaultDeviceCommandQueue(context, deviceID, NULL);
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
            if (b != CL_TRUE)
            {
                log_error("DEVICE_QUEUE_SUPPORTED is set but "
                          "CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT returned "
                          "CL_FALSE\n");
                return TEST_FAIL;
            }
        }
    }

    return TEST_PASS;
}
