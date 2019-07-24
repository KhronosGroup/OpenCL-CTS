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

static char hash_table_kernel[] =
  "#if 0\n"
  "#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable\n"
  "#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable\n"
  "#endif\n"
  "typedef struct BinNode {\n"
  " int value;\n"
  " atomic_uintptr_t pNext;\n"
  "} BinNode;\n"

  "__kernel void build_hash_table(__global uint* input, __global BinNode* pNodes, volatile __global atomic_uint* pNumNodes, uint numBins)\n"
  "{\n"
  " __global BinNode *pNew = &pNodes[ atomic_fetch_add_explicit(pNumNodes, 1, memory_order_relaxed, memory_scope_all_svm_devices) ];\n"
  " uint i = get_global_id(0);\n"
  " uint b = input[i] % numBins;\n"
  " pNew->value = input[i];\n"
  " uintptr_t next = atomic_load_explicit(&(pNodes[b].pNext), memory_order_seq_cst, memory_scope_all_svm_devices);\n"
  " do\n"
  " {\n"
  "   atomic_store_explicit(&(pNew->pNext), next, memory_order_seq_cst, memory_scope_all_svm_devices);\n" // always inserting at head of list
  " } while(!atomic_compare_exchange_strong_explicit(&(pNodes[b].pNext), &next, (uintptr_t)pNew, memory_order_seq_cst, memory_order_relaxed, memory_scope_all_svm_devices));\n"
  "}\n";

typedef struct BinNode{
  cl_uint value;
  struct BinNode* pNext;
} BinNode;

void build_hash_table_on_host(cl_context c, cl_uint* input, size_t inputSize, BinNode* pNodes, cl_int volatile *pNumNodes, cl_uint numBins)
{
  for(cl_uint i = 0; i < inputSize; i++)
  {
    BinNode *pNew = &pNodes[ AtomicFetchAddExplicit(pNumNodes, 1, memory_order_relaxed) ];
    cl_uint b = input[i] % numBins;
    pNew->value = input[i];

    BinNode *next = pNodes[b].pNext;
    do {
        pNew->pNext = next;  // always inserting at head of list
    } while(!AtomicCompareExchangeStrongExplicit(&(pNodes[b].pNext), &next, pNew, memory_order_relaxed, memory_order_seq_cst));
  }
}


int launch_kernels_and_verify(clContextWrapper &context, clCommandQueueWrapper* queues, clKernelWrapper &kernel, cl_uint num_devices, cl_uint numBins, size_t num_pixels)
{
  int err = CL_SUCCESS;
  cl_uint *pInputImage = (cl_uint*) clSVMAlloc(context, CL_MEM_READ_ONLY  | CL_MEM_SVM_FINE_GRAIN_BUFFER, sizeof(cl_uint) * num_pixels, 0);
  BinNode *pNodes      = (BinNode*) clSVMAlloc(context, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS, sizeof(BinNode) * (num_pixels * (num_devices + 1) + numBins), 0);
  cl_int *pNumNodes       = (cl_int*)  clSVMAlloc(context, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS, sizeof(cl_int), 0);

  *pNumNodes = numBins;  // using the first numBins nodes to hold the list heads.
  for(cl_uint i=0;i<numBins;i++) {
    pNodes[i].pNext = NULL;
  };

  for(cl_uint i=0; i < num_pixels; i++) pInputImage[i] = i;

  err |= clSetKernelArgSVMPointer(kernel, 0, pInputImage);
  err |= clSetKernelArgSVMPointer(kernel, 1, pNodes);
  err |= clSetKernelArgSVMPointer(kernel, 2, pNumNodes);
  err |= clSetKernelArg(kernel, 3, sizeof(cl_uint), (void*) &numBins);

  test_error(err, "clSetKernelArg failed");

  cl_event done;
  // get all the devices going simultaneously, each device (and the host) will insert all the pixels.
  for(cl_uint d=0; d<num_devices; d++)
  {
    err = clEnqueueNDRangeKernel(queues[d], kernel, 1, NULL, &num_pixels, 0, 0, NULL, &done);
    test_error(err,"clEnqueueNDRangeKernel failed");
  }
  for(cl_uint d=0; d<num_devices; d++) clFlush(queues[d]);

  // wait until we see some activity from a device (try to run host side simultaneously).
  while(numBins == AtomicLoadExplicit(pNumNodes, memory_order_relaxed));

  build_hash_table_on_host(context, pInputImage, num_pixels, pNodes, pNumNodes, numBins);

  for(cl_uint d=0; d<num_devices; d++) clFinish(queues[d]);

  cl_uint num_items = 0;
  // check correctness of each bin in the hash table.
  for(cl_uint i = 0; i < numBins; i++)
  {
    BinNode *pNode = pNodes[i].pNext;
    while(pNode)
    {
      if((pNode->value % numBins) != i)
      {
        log_error("Something went wrong, item is in wrong hash bucket\n");
        break;
      }
      num_items++;
      pNode = pNode->pNext;
    }
  }

  clSVMFree(context, pInputImage);
  clSVMFree(context, pNodes);
  clSVMFree(context, pNumNodes);
  // each device and the host inserted all of the pixels, check that none are missing.
  if(num_items != num_pixels * (num_devices + 1) )
  {
    log_error("The hash table is not correct, num items %d, expected num items: %d\n", num_items, num_pixels * (num_devices + 1));
    return -1; // test did not pass
  }
  return 0;
}

// This tests for memory consistency across devices and the host.
// Each device and the host simultaneously insert values into a single hash table.
// Each bin in the hash table is a linked list.  Each bin is protected against simultaneous
// update using a lock free technique.  The correctness of the list is verfied on the host.
// This test requires the new OpenCL 2.0 atomic operations that implement the new seq_cst memory ordering.
int test_svm_fine_grain_memory_consistency(cl_device_id deviceID, cl_context c, cl_command_queue queue, int num_elements)
{
  clContextWrapper context;
  clProgramWrapper program;
  clKernelWrapper kernel;
  clCommandQueueWrapper queues[MAXQ];

  cl_uint     num_devices = 0;
  cl_int      err = CL_SUCCESS;

  if (sizeof(void *) == 8 && (!is_extension_available(deviceID, "cl_khr_int64_base_atomics") || !is_extension_available(deviceID, "cl_khr_int64_extended_atomics")))
  {
      log_info("WARNING: test skipped. 'cl_khr_int64_base_atomics' and 'cl_khr_int64_extended_atomics' extensions are not supported\n");
      return 0;
  }

  // Make pragmas visible for 64-bit addresses
  hash_table_kernel[4] = sizeof(void *) == 8 ? '1' : '0';

  char *source[] = { hash_table_kernel };

  err = create_cl_objects(deviceID, (const char**)source, &context, &program, &queues[0], &num_devices, CL_DEVICE_SVM_FINE_GRAIN_BUFFER | CL_DEVICE_SVM_ATOMICS);
  if(err == 1) return 0; // no devices capable of requested SVM level, so don't execute but count test as passing.
  if(err < 0) return -1; // fail test.

  kernel = clCreateKernel(program, "build_hash_table", &err);
  test_error(err, "clCreateKernel failed");
  size_t num_pixels = num_elements;

  int result;
  cl_uint numBins = 1;  // all work groups in all devices and the host code will hammer on this one lock.
  result = launch_kernels_and_verify(context, queues, kernel, num_devices, numBins, num_pixels);
  if(result == -1) return result;

  numBins = 2;  // 2 locks within in same cache line will get hit from different devices and host.
  result = launch_kernels_and_verify(context, queues, kernel, num_devices, numBins, num_pixels);
  if(result == -1) return result;

  numBins = 29; // locks span a few cache lines.
  result = launch_kernels_and_verify(context, queues, kernel, num_devices, numBins, num_pixels);
  if(result == -1) return result;

  return result;
}
