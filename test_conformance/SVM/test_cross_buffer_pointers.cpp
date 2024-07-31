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

// create linked lists that use nodes from two different buffers.
const char *SVMCrossBufferPointers_test_kernel[] = {
  "\n"
  "typedef struct Node {\n"
  "    int global_id;\n"
  "    int position_in_list;\n"
  "    __global struct Node* pNext;\n"
  "} Node;\n"
  "\n"
  "__global Node* allocate_node(__global Node* pNodes1, __global Node* pNodes2, volatile __global int* allocation_index, size_t i)\n"
  "{\n"
  // mix things up, adjacent work items will allocate from different buffers
  "    if(i & 0x1)\n"
  "        return &pNodes1[atomic_inc(allocation_index)];\n"
  "    else\n"
  "        return &pNodes2[atomic_inc(allocation_index)];\n"
  "}\n"
  "\n"
  // The allocation_index parameter must be initialized on the host to N work-items
  // The first N nodes in pNodes will be the heads of the lists.
  "__kernel void create_linked_lists(__global Node* pNodes, __global Node* pNodes2, volatile __global int* allocation_index, int list_length)\n"
  "{\n"
  "    size_t i = get_global_id(0);\n"
  "    __global Node *pNode = &pNodes[i];\n"
  "\n"
  "    pNode->global_id = i;\n"
  "    pNode->position_in_list = 0;\n"
  "\n"
  "    __global Node *pNew;\n"
  "    for(int j=1; j < list_length; j++)\n"
  "    {\n"
  "        pNew = allocate_node(pNodes, pNodes2, allocation_index, i);\n"
  "        pNew->global_id = i;\n"
  "        pNew->position_in_list = j;\n"
  "        pNode->pNext = pNew;  // link new node onto end of list\n"
  "        pNode = pNew;   // move to end of list\n"
  "    }\n"
  "}\n"
  "\n"
  "__kernel void verify_linked_lists(__global Node* pNodes, __global Node* pNodes2, volatile __global uint* num_correct, int list_length)\n"
  "{\n"
  "    size_t i = get_global_id(0);\n"
  "    __global Node *pNode = &pNodes[i];\n"
  "\n"
  "    for(int j=0; j < list_length; j++)\n"
  "    {\n"
  "        if( pNode->global_id == i && pNode->position_in_list == j)\n"
  "        {\n"
  "            atomic_inc(num_correct);\n"
  "        }\n"
  "        else {\n"
  "            break;\n"
  "        }\n"
  "        pNode = pNode->pNext;\n"
  "    }\n"
  "}\n"
};


// Creates linked list using host code.
cl_int create_linked_lists_on_host(cl_command_queue cmdq, cl_mem nodes, cl_mem nodes2, cl_int ListLength, size_t numLists )
{
  cl_int error = CL_SUCCESS;

  log_info("SVM: creating linked list on host ");

  Node *pNodes = (Node*) clEnqueueMapBuffer(cmdq, nodes, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(Node)*ListLength*numLists, 0, NULL,NULL, &error);
  test_error2(error, pNodes, "clEnqueueMapBuffer failed");

  Node *pNodes2 = (Node*) clEnqueueMapBuffer(cmdq, nodes2, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(Node)*ListLength*numLists, 0, NULL,NULL, &error);
  test_error2(error, pNodes2, "clEnqueueMapBuffer failed");

  create_linked_lists(pNodes, numLists, ListLength);

  error = clEnqueueUnmapMemObject(cmdq, nodes, pNodes, 0,NULL,NULL);
  test_error(error, "clEnqueueUnmapMemObject failed");
  error = clEnqueueUnmapMemObject(cmdq, nodes2, pNodes2, 0,NULL,NULL);
  test_error(error, "clEnqueueUnmapMemObject failed");
  error = clFinish(cmdq);
  test_error(error, "clFinish failed");
  return error;
}

// Verify correctness of the linked list using host code.
cl_int verify_linked_lists_on_host(int ci, cl_command_queue cmdq, cl_mem nodes, cl_mem nodes2, cl_int ListLength, size_t numLists )
{
  cl_int error = CL_SUCCESS;

  //log_info(" and verifying on host ");

  Node *pNodes = (Node*) clEnqueueMapBuffer(cmdq, nodes, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(Node)*ListLength * numLists, 0, NULL,NULL, &error);
  test_error2(error, pNodes, "clEnqueueMapBuffer failed");
  Node *pNodes2 = (Node*) clEnqueueMapBuffer(cmdq, nodes2, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(Node)*ListLength * numLists, 0, NULL,NULL, &error);
  test_error2(error, pNodes, "clEnqueueMapBuffer failed");

  error = verify_linked_lists(pNodes, numLists, ListLength);
  if(error) return -1;

  error = clEnqueueUnmapMemObject(cmdq, nodes, pNodes, 0,NULL,NULL);
  test_error(error, "clEnqueueUnmapMemObject failed");
  error = clEnqueueUnmapMemObject(cmdq, nodes2, pNodes2, 0,NULL,NULL);
  test_error(error, "clEnqueueUnmapMemObject failed");
  error = clFinish(cmdq);
  test_error(error, "clFinish failed");
  return error;
}

// This tests that shared buffers are able to contain pointers that point to other shared buffers.
// This tests that all devices and the host share a common address space; using only the coarse-grain features.
// This is done by creating a linked list on a device and then verifying the correctness of the list
// on another device or the host.
// The linked list nodes are allocated from two different buffers this is done to ensure that cross buffer pointers work correctly.
// This basic test is performed for all combinations of devices and the host.
int test_svm_cross_buffer_pointers_coarse_grain(cl_device_id deviceID, cl_context context2, cl_command_queue queue, int num_elements)
{
  clContextWrapper    context = NULL;
  clProgramWrapper    program = NULL;
  cl_uint     num_devices = 0;
  cl_int      error = CL_SUCCESS;
  clCommandQueueWrapper queues[MAXQ];

  error = create_cl_objects(deviceID, &SVMCrossBufferPointers_test_kernel[0], &context, &program, &queues[0], &num_devices, CL_DEVICE_SVM_COARSE_GRAIN_BUFFER);
  if(error) return -1;

  size_t numLists =  num_elements;
  cl_int ListLength = 32;

  clKernelWrapper kernel_create_lists = clCreateKernel(program, "create_linked_lists", &error);
  test_error(error, "clCreateKernel failed");

  clKernelWrapper kernel_verify_lists = clCreateKernel(program, "verify_linked_lists", &error);
  test_error(error, "clCreateKernel failed");

  // this buffer holds some of the linked list nodes.
  Node* pNodes = (Node*) clSVMAlloc(context, CL_MEM_READ_WRITE, sizeof(Node)*ListLength*numLists, 0);

  // this buffer holds some of the linked list nodes.
  Node* pNodes2 = (Node*) clSVMAlloc(context, CL_MEM_READ_WRITE, sizeof(Node)*ListLength*numLists, 0);

  {
    clMemWrapper nodes = clCreateBuffer(context, CL_MEM_USE_HOST_PTR, sizeof(Node)*ListLength*numLists, pNodes, &error);
    test_error(error, "clCreateBuffer failed.");

    clMemWrapper nodes2 = clCreateBuffer(context, CL_MEM_USE_HOST_PTR, sizeof(Node)*ListLength*numLists, pNodes2, &error);
    test_error(error, "clCreateBuffer failed.");

    // this buffer holds the index into the nodes buffer that is used for node allocation
    clMemWrapper allocator = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                            sizeof(size_t), NULL, &error);
    test_error(error, "clCreateBuffer failed.");

    // this buffer holds the count of correct nodes which is computed by the verify kernel.
    clMemWrapper num_correct = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int), NULL, &error);
    test_error(error, "clCreateBuffer failed.");

    error |= clSetKernelArg(kernel_create_lists, 0, sizeof(void*), (void *) &nodes);
    //error |= clSetKernelArgSVMPointer(kernel_create_lists, 0, (void *) pNodes);
    error |= clSetKernelArg(kernel_create_lists, 1, sizeof(void*), (void *) &nodes2);
    error |= clSetKernelArg(kernel_create_lists, 2, sizeof(void*), (void *) &allocator);
    error |= clSetKernelArg(kernel_create_lists, 3, sizeof(cl_int),   (void *) &ListLength);

    error |= clSetKernelArg(kernel_verify_lists, 0, sizeof(void*), (void *) &nodes);
    error |= clSetKernelArg(kernel_verify_lists, 1, sizeof(void*), (void *) &nodes2);
    error |= clSetKernelArg(kernel_verify_lists, 2, sizeof(void*), (void *) &num_correct);
    error |= clSetKernelArg(kernel_verify_lists, 3, sizeof(cl_int),   (void *) &ListLength);
    test_error(error, "clSetKernelArg failed");

    // Create linked list on one device and verify on another device (or the host).
    // Do this for all possible combinations of devices and host within the platform.
    for (int ci=0; ci<(int)num_devices+1; ci++)  // ci is CreationIndex, index of device/q to create linked list on
    {
      for (int vi=0; vi<(int)num_devices+1; vi++)  // vi is VerificationIndex, index of device/q to verify linked list on
      {
        if(ci == num_devices) // last device index represents the host, note the num_device+1 above.
        {
          error = create_linked_lists_on_host(queues[0], nodes, nodes2, ListLength, numLists);
          if(error) return -1;
        }
        else
        {
          error = create_linked_lists_on_device(ci, queues[ci], allocator, kernel_create_lists, numLists);
          if(error) return -1;
        }

        if(vi == num_devices)
        {
          error = verify_linked_lists_on_host(vi, queues[0], nodes, nodes2, ListLength, numLists);
          if(error) return -1;
        }
        else
        {
          error = verify_linked_lists_on_device(vi, queues[vi], num_correct, kernel_verify_lists, ListLength, numLists);
          if(error) return -1;
        }
      } // inner loop, vi
    } // outer loop, ci
  }

  clSVMFree(context, pNodes2);
  clSVMFree(context, pNodes);

  return 0;
}
