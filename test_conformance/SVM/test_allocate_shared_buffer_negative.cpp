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
#include "common.h"

const cl_mem_flags svm_flag_set[] = {
    CL_MEM_READ_WRITE,
    CL_MEM_WRITE_ONLY,
    CL_MEM_READ_ONLY,
    CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER,
    CL_MEM_WRITE_ONLY | CL_MEM_SVM_FINE_GRAIN_BUFFER,
    CL_MEM_READ_ONLY | CL_MEM_SVM_FINE_GRAIN_BUFFER,
    CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS,
    CL_MEM_WRITE_ONLY | CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS,
    CL_MEM_READ_ONLY | CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS,
    0
};
const char* svm_flag_set_names[] = {
    "CL_MEM_READ_WRITE",
    "CL_MEM_WRITE_ONLY",
    "CL_MEM_READ_ONLY",
    "CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER",
    "CL_MEM_WRITE_ONLY | CL_MEM_SVM_FINE_GRAIN_BUFFER",
    "CL_MEM_READ_ONLY | CL_MEM_SVM_FINE_GRAIN_BUFFER",
    "CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS",
    "CL_MEM_WRITE_ONLY | CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS",
    "CL_MEM_READ_ONLY | CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS",
    "0"
};


int test_svm_allocate_shared_buffer_negative(cl_device_id deviceID,
                                             cl_context context2,
                                             cl_command_queue queue,
                                             int num_elements)
{
    clContextWrapper context = NULL;
    clProgramWrapper program = NULL;
    cl_uint num_devices = 0;
    cl_int err = CL_SUCCESS;
    clCommandQueueWrapper queues[MAXQ];

    cl_device_svm_capabilities caps;
    err = clGetDeviceInfo(deviceID, CL_DEVICE_SVM_CAPABILITIES,
                          sizeof(cl_device_svm_capabilities), &caps, NULL);
    test_error(err, "clGetDeviceInfo failed for CL_DEVICE_SVM_CAPABILITIES");

    // under construction...
    err = create_cl_objects(deviceID, NULL, &context, &program, &queues[0],
                            &num_devices, CL_DEVICE_SVM_COARSE_GRAIN_BUFFER);
    if (err) return err;

    size_t size = 1024;

    // iteration over flag combos
    int num_flags = sizeof(svm_flag_set) / sizeof(cl_mem_flags);
    for (int i = 0; i < num_flags; i++)
    {
        if (((svm_flag_set[i] & CL_MEM_SVM_FINE_GRAIN_BUFFER) != 0
             && (caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) == 0)
            || ((svm_flag_set[i] & CL_MEM_SVM_ATOMICS) != 0
                && (caps & CL_DEVICE_SVM_ATOMICS) == 0))
        {
            log_info("Skipping clSVMalloc with flags: %s\n",
                     svm_flag_set_names[i]);
            continue;
        }

        log_info("Testing clSVMalloc with flags: %s\n", svm_flag_set_names[i]);
        cl_char* pBufData1 =
            (cl_char*)clSVMAlloc(context, svm_flag_set[i], size, 0);
        if (pBufData1 == NULL)
        {
            log_error("SVMalloc returned NULL");
            return -1;
        }

        {
            clMemWrapper buf1 = clCreateBuffer(context, CL_MEM_USE_HOST_PTR,
                                               2 * size, pBufData1, &err);
            test_failure_error(err, CL_INVALID_BUFFER_SIZE,
                               "clCreateBuffer did not return expected error"
                               "CL_INVALID_BUFFER_SIZE");
        }

        clSVMFree(context, pBufData1);
    }

    return TEST_PASS;
}
