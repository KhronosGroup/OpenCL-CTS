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

typedef struct {
  cl_int *pA;
  cl_int *pB;
  cl_int *pC;
} BufPtrs;

const char *set_kernel_exec_info_svm_ptrs_kernel[] = {
  "struct BufPtrs;\n"
  "\n"
  "typedef struct {\n"
  "    __global int *pA;\n"
  "    __global int *pB;\n"
  "    __global int *pC;\n"
  "} BufPtrs;\n"
  "\n"
  "__kernel void set_kernel_exec_info_test(__global BufPtrs* pBufs)\n"
  "{\n"
  "    size_t i;\n"
  "   i = get_global_id(0);\n"
  "    pBufs->pA[i]++;\n"
  "    pBufs->pB[i]++;\n"
  "    pBufs->pC[i]++;\n"
  "}\n"
};

// Test that clSetKernelExecInfo works correctly with CL_KERNEL_EXEC_INFO_SVM_PTRS flag.
//
int test_svm_set_kernel_exec_info_svm_ptrs(cl_device_id deviceID, cl_context context2, cl_command_queue queue, int num_elements)
{
  clContextWrapper    c = NULL;
  clProgramWrapper    program = NULL;
  cl_uint     num_devices = 0;
  cl_int      error = CL_SUCCESS;
  clCommandQueueWrapper queues[MAXQ];

  //error = create_cl_objects(deviceID, &set_kernel_exec_info_svm_ptrs_kernel[0], &context, &program, &q, &num_devices, CL_DEVICE_SVM_FINE_GRAIN);
  error = create_cl_objects(deviceID, &set_kernel_exec_info_svm_ptrs_kernel[0], &c, &program, &queues[0], &num_devices, CL_DEVICE_SVM_COARSE_GRAIN_BUFFER);
  if(error == 1) return 0; // no devices capable of requested SVM level, so don't execute but count test as passing.
  if(error < 0) return -1; // fail test.


  clKernelWrapper k = clCreateKernel(program, "set_kernel_exec_info_test", &error);
  test_error(error, "clCreateKernel failed");

  size_t size = num_elements*sizeof(int);
  //int* pA = (int*) clSVMalloc(c, CL_MEM_READ_WRITE | CL_DEVICE_SVM_FINE_GRAIN_SYSTEM, sizeof(int)*num_elements, 0);
  //int* pB = (int*) clSVMalloc(c, CL_MEM_READ_WRITE | CL_DEVICE_SVM_FINE_GRAIN_SYSTEM, sizeof(int)*num_elements, 0);
  //int* pC = (int*) clSVMalloc(c, CL_MEM_READ_WRITE | CL_DEVICE_SVM_FINE_GRAIN_SYSTEM, sizeof(int)*num_elements, 0);
  int* pA = (int*) clSVMAlloc(c, CL_MEM_READ_WRITE, size, 0);
  int* pB = (int*) clSVMAlloc(c, CL_MEM_READ_WRITE, size, 0);
  int* pC = (int*) clSVMAlloc(c, CL_MEM_READ_WRITE, size, 0);
  BufPtrs* pBuf = (BufPtrs*) clSVMAlloc(c, CL_MEM_READ_WRITE, sizeof(BufPtrs), 0);

  bool failed = false;
  {
    clMemWrapper ba,bb,bc,bBuf;
    ba = clCreateBuffer(c, CL_MEM_USE_HOST_PTR, size, pA, &error);
    test_error(error, "clCreateBuffer failed");
    bb = clCreateBuffer(c, CL_MEM_USE_HOST_PTR, size, pB, &error);
    test_error(error, "clCreateBuffer failed");
    bc = clCreateBuffer(c, CL_MEM_USE_HOST_PTR, size, pC, &error);
    test_error(error, "clCreateBuffer failed");
    bBuf = clCreateBuffer(c, CL_MEM_USE_HOST_PTR, sizeof(BufPtrs), pBuf, &error);
    test_error(error, "clCreateBuffer failed");

    clEnqueueMapBuffer(queues[0], ba, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, size, 0, NULL, NULL, &error);
    test_error(error, "clEnqueueMapBuffer failed");
    clEnqueueMapBuffer(queues[0], bb, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, size, 0, NULL, NULL, &error);
    test_error(error, "clEnqueueMapBuffer failed");
    clEnqueueMapBuffer(queues[0], bc, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, size, 0, NULL, NULL, &error);
    test_error(error, "clEnqueueMapBuffer failed");
    clEnqueueMapBuffer(queues[0], bBuf, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(BufPtrs), 0, NULL, NULL, &error);
    test_error(error, "clEnqueueMapBuffer failed");

    for(int i = 0; i < num_elements; i++) pA[i] = pB[i] = pC[i] = 0;

    pBuf->pA = pA;
    pBuf->pB = pB;
    pBuf->pC = pC;

    error = clEnqueueUnmapMemObject(queues[0], ba, pA, 0, NULL, NULL);
    test_error(error, " clEnqueueUnmapMemObject failed.");
    error = clEnqueueUnmapMemObject(queues[0], bb, pB, 0, NULL, NULL);
    test_error(error, " clEnqueueUnmapMemObject failed.");
    error = clEnqueueUnmapMemObject(queues[0], bc, pC, 0, NULL, NULL);
    test_error(error, " clEnqueueUnmapMemObject failed.");
    error = clEnqueueUnmapMemObject(queues[0], bBuf, pBuf, 0, NULL, NULL);
    test_error(error, " clEnqueueUnmapMemObject failed.");


    error = clSetKernelArgSVMPointer(k, 0, pBuf); 
    test_error(error, "clSetKernelArg failed");

    error = clSetKernelExecInfo(k, CL_KERNEL_EXEC_INFO_SVM_PTRS, sizeof(BufPtrs), pBuf);
    test_error(error, "clSetKernelExecInfo failed");

    size_t range =  num_elements;
    error = clEnqueueNDRangeKernel(queues[0], k, 1, NULL, &range, NULL, 0, NULL, NULL);
    test_error(error,"clEnqueueNDRangeKernel failed");

    error = clFinish(queues[0]);
    test_error(error, "clFinish failed.");

    clEnqueueMapBuffer(queues[0], ba, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, size, 0, NULL, NULL, &error);
    test_error(error, "clEnqueueMapBuffer failed");
    clEnqueueMapBuffer(queues[0], bb, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, size, 0, NULL, NULL, &error);
    test_error(error, "clEnqueueMapBuffer failed");
    clEnqueueMapBuffer(queues[0], bc, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, size, 0, NULL, NULL, &error);
    test_error(error, "clEnqueueMapBuffer failed");

    for(int i = 0; i < num_elements; i++) 
    {
      if(pA[i] + pB[i] + pC[i] != 3)
        failed = true;
    }

    error = clEnqueueUnmapMemObject(queues[0], ba, pA, 0, NULL, NULL);
    test_error(error, " clEnqueueUnmapMemObject failed.");
    error = clEnqueueUnmapMemObject(queues[0], bb, pB, 0, NULL, NULL);
    test_error(error, " clEnqueueUnmapMemObject failed.");
    error = clEnqueueUnmapMemObject(queues[0], bc, pC, 0, NULL, NULL);
    test_error(error, " clEnqueueUnmapMemObject failed.");
  }

  error = clFinish(queues[0]);
  test_error(error, " clFinish failed.");

  clSVMFree(c, pA);
  clSVMFree(c, pB);
  clSVMFree(c, pC);
  clSVMFree(c, pBuf);

  if(failed) return -1;

  return 0;
}
