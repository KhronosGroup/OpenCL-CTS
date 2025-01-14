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

const char *SVMPointerPassing_test_kernel[] = {
  "__kernel void verify_char(__global uchar* pChar, volatile __global uint* num_correct, uchar expected)\n"
  "{\n"
  "    if(0 == get_global_id(0))\n"
  "    {\n"
  "        *num_correct = 0;\n"
  "        if(*pChar == expected)\n"
  "        {\n"
  "                    *num_correct=1;\n"
  "        }\n"
  "    }\n"
  "}\n"
};


// Test that arbitrarily aligned char pointers into shared buffers can be passed directly to a kernel.
// This iterates through a buffer passing a pointer to each location to the kernel.
// The buffer is initialized to known values at each location.
// The kernel checks that it finds the expected value at each location.
// TODO: possibly make this work across all base types (including typeN?), also check ptr arithmetic ++,--.
REGISTER_TEST(svm_pointer_passing)
{
    clContextWrapper contextWrapper = NULL;
    clProgramWrapper program = NULL;
    cl_uint num_devices = 0;
    cl_int error = CL_SUCCESS;
    clCommandQueueWrapper queues[MAXQ];

    error = create_cl_objects(device, &SVMPointerPassing_test_kernel[0],
                              &contextWrapper, &program, &queues[0],
                              &num_devices, CL_DEVICE_SVM_COARSE_GRAIN_BUFFER);
    context = contextWrapper;
    if (error) return -1;

    clKernelWrapper kernel_verify_char =
        clCreateKernel(program, "verify_char", &error);
    test_error(error, "clCreateKernel failed");

    size_t bufSize = 256;
    cl_uchar *pbuf_svm_alloc = (cl_uchar *)clSVMAlloc(
        context, CL_MEM_READ_WRITE, sizeof(cl_uchar) * bufSize, 0);

    cl_int *pNumCorrect = NULL;
    pNumCorrect =
        (cl_int *)clSVMAlloc(context, CL_MEM_READ_WRITE, sizeof(cl_int), 0);

    {
        clMemWrapper buf =
            clCreateBuffer(context, CL_MEM_USE_HOST_PTR,
                           sizeof(cl_uchar) * bufSize, pbuf_svm_alloc, &error);
        test_error(error, "clCreateBuffer failed.");

        clMemWrapper num_correct = clCreateBuffer(
            context, CL_MEM_USE_HOST_PTR, sizeof(cl_int), pNumCorrect, &error);
        test_error(error, "clCreateBuffer failed.");

        error = clSetKernelArg(kernel_verify_char, 1, sizeof(void *),
                               (void *)&num_correct);
        test_error(error, "clSetKernelArg failed");

        // put values into buf so that we can expect to see these values in the
        // kernel when we pass a pointer to them.
        cl_command_queue cmdq = queues[0];
        cl_uchar *pbuf_map_buffer = (cl_uchar *)clEnqueueMapBuffer(
            cmdq, buf, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0,
            sizeof(cl_uchar) * bufSize, 0, NULL, NULL, &error);
        test_error2(error, pbuf_map_buffer, "clEnqueueMapBuffer failed");
        for (int i = 0; i < (int)bufSize; i++)
        {
            pbuf_map_buffer[i] = (cl_uchar)i;
        }
        error =
            clEnqueueUnmapMemObject(cmdq, buf, pbuf_map_buffer, 0, NULL, NULL);
        test_error(error, "clEnqueueUnmapMemObject failed.");

        for (cl_uint ii = 0; ii < num_devices;
             ++ii) // iterate over all devices in the platform.
        {
            cmdq = queues[ii];
            for (int i = 0; i < (int)bufSize; i++)
            {
                cl_uchar *pChar = &pbuf_svm_alloc[i];
                error = clSetKernelArgSVMPointer(
                    kernel_verify_char, 0,
                    pChar); // pass a pointer to a location within the buffer
                test_error(error, "clSetKernelArg failed");
                error = clSetKernelArg(kernel_verify_char, 2, sizeof(cl_uchar),
                                       (void *)&i); // pass the expected value
                                                    // at the above location.
                test_error(error, "clSetKernelArg failed");
                error =
                    clEnqueueNDRangeKernel(cmdq, kernel_verify_char, 1, NULL,
                                           &bufSize, NULL, 0, NULL, NULL);
                test_error(error, "clEnqueueNDRangeKernel failed");

                pNumCorrect = (cl_int *)clEnqueueMapBuffer(
                    cmdq, num_correct, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0,
                    sizeof(cl_int), 0, NULL, NULL, &error);
                test_error2(error, pNumCorrect, "clEnqueueMapBuffer failed");
                cl_int correct_count = *pNumCorrect;
                error = clEnqueueUnmapMemObject(cmdq, num_correct, pNumCorrect,
                                                0, NULL, NULL);
                test_error(error, "clEnqueueUnmapMemObject failed.");

                if (correct_count != 1)
                {
                    log_error("Passing pointer directly to kernel for byte #%d "
                              "failed on device %d\n",
                              i, ii);
                    return -1;
                }
            }

            error = clFinish(cmdq);
            test_error(error, "clFinish failed");
        }
    }


    clSVMFree(context, pbuf_svm_alloc);
    clSVMFree(context, pNumCorrect);

    return 0;
}
