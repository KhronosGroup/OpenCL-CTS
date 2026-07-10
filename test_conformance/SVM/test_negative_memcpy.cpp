//
// Copyright (c) 2025 The Khronos Group Inc.
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
#include "common.h"

REGISTER_TEST(svm_negative_memcpy_overlap)
{
    cl_int err = CL_SUCCESS;

    cl_device_svm_capabilities caps = 0;
    err = clGetDeviceInfo(device, CL_DEVICE_SVM_CAPABILITIES, sizeof(caps),
                          &caps, NULL);

    if (err != CL_SUCCESS || caps == 0)
    {
        log_error("svm_negative_memcpy_overlap: SVM is not supported on this "
                  "device\n");
        return TEST_SKIP;
    }

    clContextWrapper contextWrapper = NULL;
    clCommandQueueWrapper queues[MAXQ];
    cl_uint num_devices = 0;
    err = create_cl_objects(device, NULL, &contextWrapper, NULL, &queues[0],
                            &num_devices, 0);
    test_error(err, "svm_negative_memcpy_overlap: create_cl_objects failed");

    queue = queues[0];

    const size_t dataSize = 32;

    clSVMWrapper data_buf =
        clSVMWrapper(contextWrapper, dataSize * 2, CL_MEM_READ_WRITE);

    test_assert_error(data_buf() != nullptr, "clSVMAlloc failed");

    // 1. A case where the end of the source memory partially overlaps the
    // beginning of the destination memory.
    {
        char *src = (char *)data_buf();
        char *dst = src + dataSize / 2;

        err = clEnqueueSVMMemcpy(queue, CL_TRUE, dst, src, dataSize, 0, NULL,
                                 NULL);

        if (err != CL_MEM_COPY_OVERLAP)
        {
            log_error(
                "svm_negative_memcpy_overlap: invalid result for "
                "clEnqueueSVMMemcpy, expected CL_MEM_COPY_OVERLAP, got %s\n",
                IGetErrorString(err));
            return TEST_FAIL;
        }
    }

    // 2. A case where the beginning of the source memory partially overlaps the
    // end of the destination memory.
    {
        char *dst = (char *)data_buf();
        char *src = dst + dataSize / 2;

        err = clEnqueueSVMMemcpy(queue, CL_TRUE, dst, src, dataSize, 0, NULL,
                                 NULL);

        if (err != CL_MEM_COPY_OVERLAP)
        {
            log_error(
                "svm_negative_memcpy_overlap: invalid result for "
                "clEnqueueSVMMemcpy, expected CL_MEM_COPY_OVERLAP, got %s\n",
                IGetErrorString(err));
            return TEST_FAIL;
        }
    }

    // 3. A case where the source memory completely overlaps the destination
    // memory.
    {
        char *src = (char *)data_buf();
        char *dst = (char *)data_buf();

        err = clEnqueueSVMMemcpy(queue, CL_TRUE, dst, src, dataSize, 0, NULL,
                                 NULL);

        if (err != CL_MEM_COPY_OVERLAP)
        {
            log_error(
                "svm_negative_memcpy_overlap: invalid result for "
                "clEnqueueSVMMemcpy, expected CL_MEM_COPY_OVERLAP, got %s\n",
                IGetErrorString(err));
            return TEST_FAIL;
        }
    }

    return TEST_PASS;
}
