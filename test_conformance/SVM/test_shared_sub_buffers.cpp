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

const char *shared_sub_buffers_test_kernel[] = {
  "typedef struct Node {\n"
  "    int global_id;\n"
  "    int position_in_list;\n"
  "    __global struct Node* pNext;\n"
  "} Node;\n"

  // create linked lists that use nodes from 2 different buffers
  "__global Node* allocate_node(__global Node* pNodes1, __global Node* pNodes2, volatile __global int* allocation_index, size_t i)\n"
  "{\n"
  // mix things up, adjacent work items will allocate from different buffers
  "    if(i & 0x1)\n"
  "        return &pNodes1[atomic_inc(allocation_index)];\n"
  "    else\n"
  "        return &pNodes2[atomic_inc(allocation_index)];\n"
  "}\n"

  // The allocation_index parameter must be initialized on the host to N work-items
  // The first N nodes in pNodes will be the heads of the lists.
  // This tests passing 4 different sub-buffers that come from two parent buffers.
  // Note that we have arguments that appear to be unused, but they are required so that system knows to get all the sub-buffers on to the device
  "__kernel void create_linked_lists(__global Node* pNodes_sub1, __global Node* pNodes2_sub1, __global Node* pNodes_sub2, __global Node* pNodes2_sub2, volatile __global int* allocation_index, int list_length) \n"
  "{\n"
  "    size_t i = get_global_id(0);\n"
  "    __global Node *pNode = &pNodes_sub1[i];\n"
  "    pNode->global_id = i;\n"
  "    pNode->position_in_list = 0;\n"
  "    __global Node *pNew;\n"
  "    for(int j=1; j < list_length; j++) {\n"
  "        pNew = allocate_node(pNodes_sub1, pNodes2_sub1, allocation_index, i);\n"
  "        pNew->global_id = i;\n"
  "        pNew->position_in_list = j;\n"
  "        pNode->pNext = pNew;  // link new node onto end of list\n"
  "        pNode = pNew;   // move to end of list\n"
  "    }\n"
  "}\n"
  // Note that we have arguments that appear to be unused, but they are required so that system knows to get all the sub-buffers on to the device
  "__kernel void verify_linked_lists(__global Node* pNodes_sub1, __global Node* pNodes2_sub1, __global Node* pNodes_sub2, __global Node* pNodes2_sub2, volatile __global uint* num_correct, int list_length)\n"
  "{\n"
  "    size_t i = get_global_id(0);\n"
  "    __global Node *pNode = &pNodes_sub1[i];\n"
  "    for(int j=0; j < list_length; j++) {\n"
  "        if( pNode->global_id == i && pNode->position_in_list == j)\n"
  "            atomic_inc(num_correct);\n"
  "        else \n"
  "            break;\n"
  "        pNode = pNode->pNext;\n"
  "    }\n"
  "}\n"
};


// Creates linked list using host code.
cl_int create_linked_lists_on_host_sb(cl_command_queue cmdq, cl_mem nodes, cl_mem nodes2, cl_int ListLength, size_t numLists )
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
cl_int verify_linked_lists_on_host_sb(int ci, cl_command_queue cmdq, cl_mem nodes, cl_mem nodes2, cl_int ListLength, size_t numLists )
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


// This tests that shared sub-buffers can be created and that they inherit the flags from the parent buffer when no flags are specified.
// This tests that passing only the sub-buffers to a kernel works.
// The test is derived from the cross-buffer pointers test which
// tests that shared buffers are able to contain pointers that point to other shared buffers.
// This tests that all devices and the host share a common address space; using only the coarse-grain features.
// This is done by creating a linked list on a device and then verifying the correctness of the list
// on another device or the host.
// The linked list nodes are allocated from two different buffers this is done to ensure that cross buffer pointers work correctly.
// This basic test is performed for all combinations of devices and the host.
REGISTER_TEST(svm_shared_sub_buffers)
{
    clContextWrapper contextWrapper = NULL;
    clProgramWrapper program = NULL;
    cl_uint num_devices = 0;
    cl_int error = CL_SUCCESS;
    clCommandQueueWrapper queues[MAXQ];

    error = create_cl_objects(device, &shared_sub_buffers_test_kernel[0],
                              &contextWrapper, &program, &queues[0],
                              &num_devices, CL_DEVICE_SVM_COARSE_GRAIN_BUFFER);
    context = contextWrapper;
    if (error) return -1;

    size_t numLists = num_elements;
    if (numLists & 0x1)
        numLists++; // force even size, so we can easily create two sub-buffers
                    // of same size.

    cl_int ListLength = 32;

    clKernelWrapper kernel_create_lists =
        clCreateKernel(program, "create_linked_lists", &error);
    test_error(error, "clCreateKernel failed");

    clKernelWrapper kernel_verify_lists =
        clCreateKernel(program, "verify_linked_lists", &error);
    test_error(error, "clCreateKernel failed");

    size_t nodes_bufsize = sizeof(Node) * ListLength * numLists;
    Node *pNodes =
        (Node *)clSVMAlloc(context, CL_MEM_READ_WRITE, nodes_bufsize, 0);
    Node *pNodes2 =
        (Node *)clSVMAlloc(context, CL_MEM_READ_WRITE, nodes_bufsize, 0);

    {
        // this buffer holds some of the linked list nodes.
        clMemWrapper nodes = clCreateBuffer(context, CL_MEM_USE_HOST_PTR,
                                            nodes_bufsize, pNodes, &error);
        test_error(error, "clCreateBuffer failed.");

        cl_buffer_region r;
        r.origin = 0;
        r.size = nodes_bufsize / 2;
        // this should inherit the flag settings from nodes buffer
        clMemWrapper nodes_sb1 = clCreateSubBuffer(
            nodes, 0, CL_BUFFER_CREATE_TYPE_REGION, (void *)&r, &error);
        test_error(error, "clCreateSubBuffer");
        r.origin = nodes_bufsize / 2;
        clMemWrapper nodes_sb2 = clCreateSubBuffer(
            nodes, 0, CL_BUFFER_CREATE_TYPE_REGION, (void *)&r, &error);
        test_error(error, "clCreateSubBuffer");


        // this buffer holds some of the linked list nodes.
        clMemWrapper nodes2 = clCreateBuffer(
            context, CL_MEM_USE_HOST_PTR, sizeof(Node) * ListLength * numLists,
            pNodes2, &error);
        test_error(error, "clCreateBuffer failed.");
        r.origin = 0;
        r.size = nodes_bufsize / 2;
        // this should inherit the flag settings from nodes buffer
        clMemWrapper nodes2_sb1 = clCreateSubBuffer(
            nodes2, 0, CL_BUFFER_CREATE_TYPE_REGION, (void *)&r, &error);
        test_error(error, "clCreateSubBuffer");
        r.origin = nodes_bufsize / 2;
        clMemWrapper nodes2_sb2 = clCreateSubBuffer(
            nodes2, 0, CL_BUFFER_CREATE_TYPE_REGION, (void *)&r, &error);
        test_error(error, "clCreateSubBuffer");


        // this buffer holds the index into the nodes buffer that is used for
        // node allocation
        clMemWrapper allocator = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                                sizeof(size_t), NULL, &error);
        test_error(error, "clCreateBuffer failed.");

        // this buffer holds the count of correct nodes which is computed by the
        // verify kernel.
        clMemWrapper num_correct = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                                  sizeof(cl_int), NULL, &error);
        test_error(error, "clCreateBuffer failed.");

        error |= clSetKernelArg(kernel_create_lists, 0, sizeof(void *),
                                (void *)&nodes_sb1);
        error |= clSetKernelArg(kernel_create_lists, 1, sizeof(void *),
                                (void *)&nodes2_sb1);
        error |= clSetKernelArg(kernel_create_lists, 2, sizeof(void *),
                                (void *)&nodes_sb2);
        error |= clSetKernelArg(kernel_create_lists, 3, sizeof(void *),
                                (void *)&nodes2_sb2);
        error |= clSetKernelArg(kernel_create_lists, 4, sizeof(void *),
                                (void *)&allocator);
        error |= clSetKernelArg(kernel_create_lists, 5, sizeof(cl_int),
                                (void *)&ListLength);

        error |= clSetKernelArg(kernel_verify_lists, 0, sizeof(void *),
                                (void *)&nodes_sb1);
        error |= clSetKernelArg(kernel_verify_lists, 1, sizeof(void *),
                                (void *)&nodes2_sb1);
        error |= clSetKernelArg(kernel_verify_lists, 2, sizeof(void *),
                                (void *)&nodes_sb2);
        error |= clSetKernelArg(kernel_verify_lists, 3, sizeof(void *),
                                (void *)&nodes2_sb2);
        error |= clSetKernelArg(kernel_verify_lists, 4, sizeof(void *),
                                (void *)&num_correct);
        error |= clSetKernelArg(kernel_verify_lists, 5, sizeof(cl_int),
                                (void *)&ListLength);
        test_error(error, "clSetKernelArg failed");

        // Create linked list on one device and verify on another device (or the
        // host). Do this for all possible combinations of devices and host
        // within the platform.
        for (int ci = 0; ci < (int)num_devices + 1;
             ci++) // ci is CreationIndex, index of device/q to create linked
                   // list on
        {
            for (int vi = 0; vi < (int)num_devices + 1;
                 vi++) // vi is VerificationIndex, index of device/q to verify
                       // linked list on
            {
                if (ci == num_devices) // last device index represents the host,
                                       // note the num_device+1 above.
                {
                    error = create_linked_lists_on_host_sb(
                        queues[0], nodes, nodes2, ListLength, numLists);
                    if (error) return -1;
                }
                else
                {
                    error = create_linked_lists_on_device(
                        ci, queues[ci], allocator, kernel_create_lists,
                        numLists);
                    if (error) return -1;
                }

                if (vi == num_devices)
                {
                    error = verify_linked_lists_on_host_sb(
                        vi, queues[0], nodes, nodes2, ListLength, numLists);
                    if (error) return -1;
                }
                else
                {
                    error = verify_linked_lists_on_device(
                        vi, queues[vi], num_correct, kernel_verify_lists,
                        ListLength, numLists);
                    if (error) return -1;
                }
            } // inner loop, vi
        } // outer loop, ci
    }
    clSVMFree(context, pNodes2);
    clSVMFree(context, pNodes);

    return 0;
}
