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

const cl_mem_flags flag_set[] = {
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
const char* flag_set_names[] = {
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


REGISTER_TEST(svm_allocate_shared_buffer)
{
    clContextWrapper contextWrapper = NULL;
    clProgramWrapper program = NULL;
    cl_uint num_devices = 0;
    cl_int err = CL_SUCCESS;
    clCommandQueueWrapper queues[MAXQ];

    cl_device_svm_capabilities caps;
    err = clGetDeviceInfo(device, CL_DEVICE_SVM_CAPABILITIES,
                          sizeof(cl_device_svm_capabilities), &caps, NULL);
    test_error(err, "clGetDeviceInfo failed for CL_DEVICE_SVM_CAPABILITIES");

    // under construction...
    err = create_cl_objects(device, NULL, &contextWrapper, &program, &queues[0],
                            &num_devices, CL_DEVICE_SVM_COARSE_GRAIN_BUFFER);
    context = contextWrapper;
    if (err) return -1;

    size_t size = 1024;

    // iteration over flag combos
    int num_flags = sizeof(flag_set) / sizeof(cl_mem_flags);
    for (int i = 0; i < num_flags; i++)
    {
        if (((flag_set[i] & CL_MEM_SVM_FINE_GRAIN_BUFFER) != 0
             && (caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) == 0)
            || ((flag_set[i] & CL_MEM_SVM_ATOMICS) != 0
                && (caps & CL_DEVICE_SVM_ATOMICS) == 0))
        {
            log_info("Skipping clSVMalloc with flags: %s\n", flag_set_names[i]);
            continue;
        }

        log_info("Testing clSVMalloc with flags: %s\n", flag_set_names[i]);
        cl_char *pBufData1 =
            (cl_char *)clSVMAlloc(context, flag_set[i], size, 0);
        if (pBufData1 == NULL)
        {
            log_error("SVMalloc returned NULL");
            return -1;
        }

        {
            clMemWrapper buf = clCreateBuffer(context, CL_MEM_USE_HOST_PTR,
                                              size, pBufData1, &err);
            test_error(err, "clCreateBuffer failed");

            cl_char *pBufData2 = NULL;
            cl_uint flags = CL_MAP_READ | CL_MAP_READ;
            if (flag_set[i] & CL_MEM_HOST_READ_ONLY) flags ^= CL_MAP_WRITE;
            if (flag_set[i] & CL_MEM_HOST_WRITE_ONLY) flags ^= CL_MAP_READ;

            if (!(flag_set[i] & CL_MEM_HOST_NO_ACCESS))
            {
                pBufData2 = (cl_char *)clEnqueueMapBuffer(
                    queues[0], buf, CL_TRUE, flags, 0, size, 0, NULL, NULL,
                    &err);
                test_error(err, "clEnqueueMapBuffer failed");

                if (pBufData2 != pBufData1 || NULL == pBufData1)
                {
                    log_error("SVM pointer returned by clEnqueueMapBuffer "
                              "doesn't match pointer returned by clSVMalloc");
                    return -1;
                }
                err = clEnqueueUnmapMemObject(queues[0], buf, pBufData2, 0,
                                              NULL, NULL);
                test_error(err, "clEnqueueUnmapMemObject failed");
                err = clFinish(queues[0]);
                test_error(err, "clFinish failed");
            }
        }

        clSVMFree(context, pBufData1);
    }

    return 0;
}
