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
    // clGetDeviceInfo, passing CL_​DEVICE_​SVM_​CAPABILITIES:
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
        // Returns CL_​FALSE if no devices in the context associated with
        // memobj support Shared Virtual Memory.
        cl_bool usesSVMPointer;
        error =
            clGetMemObjectInfo(mem, CL_MEM_USES_SVM_POINTER,
                               sizeof(usesSVMPointer), &usesSVMPointer, NULL);
        test_error(error,
                   "Unable to query CL_​MEM_​USES_​SVM_​POINTER");

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
