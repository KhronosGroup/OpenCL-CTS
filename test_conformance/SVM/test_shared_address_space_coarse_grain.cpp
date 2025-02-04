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

// Creates linked list using host code
cl_int create_linked_lists_on_host(cl_command_queue cmdq, cl_mem nodes, Node *pNodes2, cl_int ListLength, size_t numLists, cl_bool useNewAPI )
{
  cl_int error = CL_SUCCESS;

  log_info("SVM: creating linked list on host ");

  Node *pNodes;
  if (useNewAPI == CL_FALSE)
  {
    pNodes = (Node*) clEnqueueMapBuffer(cmdq, nodes, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(Node)*ListLength*numLists, 0, NULL,NULL, &error);
    test_error2(error, pNodes, "clEnqMapBuffer failed");
  }
  else
  {
    pNodes = pNodes2;
    error = clEnqueueSVMMap(cmdq, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, pNodes2, sizeof(Node)*ListLength*numLists, 0, NULL,NULL);
    test_error2(error, pNodes, "clEnqueueSVMMap failed");
  }

  create_linked_lists(pNodes, numLists, ListLength);

  if (useNewAPI == CL_FALSE)
  {
    error = clEnqueueUnmapMemObject(cmdq, nodes, pNodes, 0,NULL,NULL);
    test_error(error, "clEnqueueUnmapMemObject failed.");
  }
  else
  {
    error = clEnqueueSVMUnmap(cmdq, pNodes2, 0, NULL, NULL);
    test_error(error, "clEnqueueSVMUnmap failed.");
  }

  error = clFinish(cmdq);
  test_error(error, "clFinish failed.");
  return error;
}

// Purpose: uses host code to verify correctness of the linked list
cl_int verify_linked_lists_on_host(int ci, cl_command_queue cmdq, cl_mem nodes, Node *pNodes2, cl_int ListLength, size_t numLists, cl_bool useNewAPI )
{
  cl_int error = CL_SUCCESS;

  Node *pNodes;
  if (useNewAPI == CL_FALSE)
  {
    pNodes = (Node*) clEnqueueMapBuffer(cmdq, nodes, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(Node)*ListLength * numLists, 0, NULL,NULL, &error);
    test_error2(error, pNodes, "clEnqueueMapBuffer failed");
  }
  else
  {
    pNodes = pNodes2;
    error = clEnqueueSVMMap(cmdq, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, pNodes2, sizeof(Node)*ListLength * numLists, 0, NULL,NULL);
    test_error2(error, pNodes, "clEnqueueSVMMap failed");
  }

  error = verify_linked_lists(pNodes, numLists, ListLength);
  if(error) return -1;

  if (useNewAPI == CL_FALSE)
  {
    error = clEnqueueUnmapMemObject(cmdq, nodes, pNodes, 0,NULL,NULL);
    test_error(error, "clEnqueueUnmapMemObject failed.");
  }
  else
  {
    error = clEnqueueSVMUnmap(cmdq, pNodes2, 0,NULL,NULL);
    test_error(error, "clEnqueueSVMUnmap failed.");
  }

  error = clFinish(cmdq);
  test_error(error, "clFinish failed.");
  return error;
}

cl_int create_linked_lists_on_device(int ci, cl_command_queue cmdq, cl_mem allocator, cl_kernel kernel_create_lists, size_t numLists  )
{
  cl_int error = CL_SUCCESS;
  log_info("SVM: creating linked list on device: %d ", ci);

  size_t *pAllocator = (size_t *)clEnqueueMapBuffer(
      cmdq, allocator, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(size_t),
      0, NULL, NULL, &error);
  test_error2(error, pAllocator, "clEnqueueMapBuffer failed");
  // reset allocator index
  *pAllocator = numLists;   // the first numLists elements of the nodes array are already allocated (they hold the head of each list).
  error = clEnqueueUnmapMemObject(cmdq, allocator, pAllocator, 0,NULL,NULL);
  test_error(error, " clEnqueueUnmapMemObject failed.");

  error = clEnqueueNDRangeKernel(cmdq, kernel_create_lists, 1, NULL, &numLists, NULL, 0, NULL, NULL);
  test_error(error, "clEnqueueNDRange failed.");
  error = clFinish(cmdq);
  test_error(error, "clFinish failed.");

  return error;
}

cl_int verify_linked_lists_on_device(int vi, cl_command_queue cmdq,cl_mem num_correct, cl_kernel kernel_verify_lists, cl_int ListLength, size_t numLists  )
{
  cl_int error = CL_SUCCESS;

  log_info(" and verifying on device: %d ", vi);

  cl_int *pNumCorrect = (cl_int*) clEnqueueMapBuffer(cmdq, num_correct, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(cl_int), 0, NULL,NULL, &error);
  test_error2(error, pNumCorrect, "clEnqueueMapBuffer failed");

  *pNumCorrect = 0;     // reset numCorrect to zero

  error = clEnqueueUnmapMemObject(cmdq, num_correct, pNumCorrect, 0,NULL,NULL);
  test_error(error, "clEnqueueUnmapMemObject failed.");

  error = clEnqueueNDRangeKernel(cmdq, kernel_verify_lists, 1, NULL, &numLists, NULL, 0, NULL, NULL);
  test_error(error,"clEnqueueNDRangeKernel failed");

  pNumCorrect = (cl_int*) clEnqueueMapBuffer(cmdq, num_correct, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(cl_int), 0, NULL,NULL, &error);
  test_error2(error, pNumCorrect, "clEnqueueMapBuffer failed");
  cl_int correct_count = *pNumCorrect;
  error = clEnqueueUnmapMemObject(cmdq, num_correct, pNumCorrect, 0,NULL,NULL);
  test_error(error, "clEnqueueUnmapMemObject failed");
  clFinish(cmdq);
  test_error(error,"clFinish failed");

  if(correct_count != ListLength * (cl_uint)numLists)
  {
    error = -1;
    log_info("Failed\n");
  }
  else
    log_info("Passed\n");

  return error;
}

// This tests that all devices and the host share a common address space; using only the coarse-grain features.
// This is done by creating a linked list on a device and then verifying the correctness of the list
// on another device or the host.  This basic test is performed for all combinations of devices and the host that exist within
// the platform.  The test passes only if every combination passes.
int shared_address_space_coarse_grain(cl_device_id deviceID, cl_context context2, cl_command_queue queue, int num_elements, cl_bool useNewAPI)
{
  clContextWrapper    context = NULL;
  clProgramWrapper    program = NULL;
  cl_uint     num_devices = 0;
  cl_int      error = CL_SUCCESS;
  clCommandQueueWrapper queues[MAXQ];

  error = create_cl_objects(deviceID, &linked_list_create_and_verify_kernels[0], &context, &program, &queues[0], &num_devices, CL_DEVICE_SVM_COARSE_GRAIN_BUFFER);
  if(error) return -1;

  size_t numLists =  num_elements;
  cl_int ListLength = 32;

  clKernelWrapper kernel_create_lists = clCreateKernel(program, "create_linked_lists", &error);
  test_error(error, "clCreateKernel failed");

  clKernelWrapper kernel_verify_lists = clCreateKernel(program, "verify_linked_lists", &error);
  test_error(error, "clCreateKernel failed");

  // this buffer holds the linked list nodes.
  Node* pNodes = (Node*) clSVMAlloc(context, CL_MEM_READ_WRITE, sizeof(Node)*ListLength*numLists, 0);

  {
    cl_bool usesSVMpointer = CL_FALSE;
    clMemWrapper nodes;
    if (useNewAPI == CL_FALSE)
    {
      nodes = clCreateBuffer(context, CL_MEM_USE_HOST_PTR, sizeof(Node)*ListLength*numLists, pNodes, &error);
      test_error(error, "clCreateBuffer failed.");

      // verify if buffer uses SVM pointer
      size_t paramSize = 0;
      error = clGetMemObjectInfo(nodes, CL_MEM_USES_SVM_POINTER, 0, 0, &paramSize);
      test_error(error, "clGetMemObjectInfo failed.");

      if (paramSize != sizeof(cl_bool))
      {
        log_error("clGetMemObjectInfo(CL_MEM_USES_SVM_POINTER) returned wrong size.");
        return -1;
      }

      error = clGetMemObjectInfo(nodes, CL_MEM_USES_SVM_POINTER, sizeof(cl_bool), &usesSVMpointer, 0);
      test_error(error, "clGetMemObjectInfo failed.");

      if (usesSVMpointer != CL_TRUE)
      {
        log_error("clGetMemObjectInfo(CL_MEM_USES_SVM_POINTER) returned CL_FALSE for buffer created from SVM pointer.");
        return -1;
      }
    }

    // this buffer holds an index into the nodes buffer, it is used for node allocation
    clMemWrapper allocator = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                            sizeof(size_t), NULL, &error);

    test_error(error, "clCreateBuffer failed.");

    error = clGetMemObjectInfo(allocator, CL_MEM_USES_SVM_POINTER, sizeof(cl_bool), &usesSVMpointer, 0);
    test_error(error, "clGetMemObjectInfo failed.");

    if (usesSVMpointer != CL_FALSE)
    {
      log_error("clGetMemObjectInfo(CL_MEM_USES_SVM_POINTER) returned CL_TRUE for non-SVM buffer.");
      return -1;
    }

    // this buffer holds the count of correct nodes, which is computed by the verify kernel.
    clMemWrapper num_correct = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int), NULL, &error);
    test_error(error, "clCreateBuffer failed.");

    if (useNewAPI == CL_TRUE)
      error |= clSetKernelArgSVMPointer(kernel_create_lists, 0, pNodes);
    else
      error |= clSetKernelArg(kernel_create_lists, 0, sizeof(void*), (void *) &nodes);

    error |= clSetKernelArg(kernel_create_lists, 1, sizeof(void*), (void *) &allocator);
    error |= clSetKernelArg(kernel_create_lists, 2, sizeof(cl_int),   (void *) &ListLength);

    error |= clSetKernelArgSVMPointer(kernel_verify_lists, 0, pNodes);
    error |= clSetKernelArg(kernel_verify_lists, 1, sizeof(void*), (void *) &num_correct);
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
          error = create_linked_lists_on_host(queues[0], nodes, pNodes, ListLength, numLists, useNewAPI);
          if(error) return -1;
        }
        else
        {
          error = create_linked_lists_on_device(ci, queues[ci], allocator, kernel_create_lists, numLists);
          if(error) return -1;
        }

        if(vi == num_devices)
        {
          error = verify_linked_lists_on_host(vi, queues[0], nodes, pNodes, ListLength, numLists, useNewAPI);
          if(error) return -1;
        }
        else
        {
          error = verify_linked_lists_on_device(vi, queues[vi], num_correct, kernel_verify_lists, ListLength, numLists);
          if(error) return -1;
        }
      }
    }
  }

  clSVMFree(context, pNodes);

  return 0;
}

REGISTER_TEST(svm_shared_address_space_coarse_grain_old_api)
{
    return shared_address_space_coarse_grain(device, context, queue,
                                             num_elements, CL_FALSE);
}

REGISTER_TEST(svm_shared_address_space_coarse_grain_new_api)
{
    return shared_address_space_coarse_grain(device, context, queue,
                                             num_elements, CL_TRUE);
}
