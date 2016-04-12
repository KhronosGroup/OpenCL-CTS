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




cl_int create_linked_lists_on_device_no_map(int ci, cl_command_queue cmdq, size_t* pAllocator, cl_kernel kernel_create_lists, size_t numLists  )
{
  cl_int error = CL_SUCCESS;
  log_info("SVM: creating linked list on device: %d ", ci);

  // reset allocator index
  *pAllocator = numLists;   // the first numLists elements of the nodes array are already allocated (they hold the head of each list).
  error = clEnqueueNDRangeKernel(cmdq, kernel_create_lists, 1, NULL, &numLists, NULL, 0, NULL, NULL);
  test_error(error, "clEnqueueNDRange failed.");
  error = clFinish(cmdq);
  test_error(error, "clFinish failed.");
  return error;
}

cl_int verify_linked_lists_on_device_no_map(int vi, cl_command_queue cmdq,cl_int* pNumCorrect, cl_kernel kernel_verify_lists, cl_int ListLength, size_t numLists  )
{
  cl_int error = CL_SUCCESS;

  log_info(" and verifying on device: %d ", vi);

  *pNumCorrect = 0;     // reset numCorrect to zero

  error = clEnqueueNDRangeKernel(cmdq, kernel_verify_lists, 1, NULL, &numLists, NULL, 0, NULL, NULL);
  test_error(error,"clEnqueueNDRangeKernel failed");
  clFinish(cmdq);
  test_error(error,"clFinish failed");

  cl_int correct_count = *pNumCorrect;
  if(correct_count != ListLength * (cl_uint)numLists)
  {
    error = -1;
    log_info("Failed\n");
  }
  else
    log_info("Passed\n");

  return error;
}

// This tests that all devices and the host share a common address space; using only the fine-grain with buffers mode.
// This is done by creating a linked list on a device and then verifying the correctness of the list
// on another device or the host.  This basic test is performed for all combinations of devices and the host that exist within
// the platform.  The test passes only if every combination passes.
int test_svm_shared_address_space_fine_grain_buffers(cl_device_id deviceID, cl_context context2, cl_command_queue queue, int num_elements)
{
  clContextWrapper    context = NULL;
  clProgramWrapper    program = NULL;
  cl_uint     num_devices = 0;
  cl_int      error = CL_SUCCESS;
  clCommandQueueWrapper queues[MAXQ];

  error = create_cl_objects(deviceID, &linked_list_create_and_verify_kernels[0], &context, &program, &queues[0], &num_devices, CL_DEVICE_SVM_FINE_GRAIN_BUFFER);
  if(error == 1) return 0; // no devices capable of requested SVM level, so don't execute but count test as passing.
  if(error < 0) return -1; // fail test.

  size_t numLists =  num_elements;
  cl_int ListLength = 32;

  clKernelWrapper kernel_create_lists = clCreateKernel(program, "create_linked_lists", &error);
  test_error(error, "clCreateKernel failed");

  clKernelWrapper kernel_verify_lists = clCreateKernel(program, "verify_linked_lists", &error);
  test_error(error, "clCreateKernel failed");

  // this buffer holds the linked list nodes.
  Node* pNodes = (Node*) clSVMAlloc(context, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER, sizeof(Node)*ListLength*numLists, 0);

  // this buffer holds an index into the nodes buffer, it is used for node allocation
  size_t *pAllocator = (size_t*) clSVMAlloc(context, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER, sizeof(size_t), 0);

  // this buffer holds the count of correct nodes, which is computed by the verify kernel.
  cl_int *pNumCorrect = (cl_int*) clSVMAlloc(context, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER, sizeof(cl_int), 0);

  error |= clSetKernelArgSVMPointer(kernel_create_lists, 0, pNodes);
  error |= clSetKernelArgSVMPointer(kernel_create_lists, 1, pAllocator);
  error |= clSetKernelArg(kernel_create_lists, 2, sizeof(cl_int),   (void *) &ListLength);

  error |= clSetKernelArgSVMPointer(kernel_verify_lists, 0, pNodes);
  error |= clSetKernelArgSVMPointer(kernel_verify_lists, 1, pNumCorrect);
  error |= clSetKernelArg(kernel_verify_lists, 2, sizeof(cl_int),   (void *) &ListLength);
  test_error(error, "clSetKernelArg failed");

  // Create linked list on one device and verify on another device (or the host).
  // Do this for all possible combinations of devices and host within the platform.
  for (int ci=0; ci<(int)num_devices+1; ci++)  // ci is CreationIndex, index of device/q to create linked list on
  {
    for (int vi=0; vi<(int)num_devices+1; vi++)  // vi is VerificationIndex, index of device/q to verify linked list on
    {
      if(ci == num_devices) // last device index represents the host, note the num_device+1 above.
      {
        log_info("SVM: creating linked list on host ");
        create_linked_lists(pNodes, numLists, ListLength);
      }
      else
      {
        error = create_linked_lists_on_device_no_map(ci, queues[ci], pAllocator, kernel_create_lists, numLists);
        if(error) return -1;
      }

      if(vi == num_devices)
      {
        error = verify_linked_lists(pNodes, numLists, ListLength);
        if(error) return -1;
      }
      else
      {
        error = verify_linked_lists_on_device_no_map(vi, queues[vi], pNumCorrect, kernel_verify_lists, ListLength, numLists);
        if(error) return -1;
      }
    }
  }

  clSVMFree(context, pNodes);
  clSVMFree(context, pAllocator);
  clSVMFree(context, pNumCorrect);

  return 0;
}
