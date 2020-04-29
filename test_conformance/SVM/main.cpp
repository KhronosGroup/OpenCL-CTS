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
#include "harness/compat.h"

#include <stdio.h>
#include <vector>
#include <sstream>
#include "harness/testHarness.h"
#include "harness/kernelHelpers.h"

#include "common.h"

// SVM Atomic wrappers.
// Platforms that support SVM atomics (atomics that work across the host and devices) need to implement these host side functions correctly.
// Platforms that do not support SVM atomics can simpy implement these functions as empty stubs since the functions will not be called.
// For now only Windows x86 is implemented, add support for other platforms as needed.
cl_int AtomicLoadExplicit(volatile cl_int * pValue, cl_memory_order order)
{
#if (defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))) || (defined(__GNUC__) && (defined(__i386__) || defined(__x86_64__)))
  return *pValue;  // provided the value is aligned x86 doesn't need anything more than this for seq_cst.
#elif defined(__GNUC__)
	return __sync_add_and_fetch(pValue, 0);
#else
  log_error("ERROR: AtomicLoadExplicit function not implemented\n");
  return -1;
#endif
}
// all the x86 atomics are seq_cst, so don't need to do anything with the memory order parameter.
cl_int AtomicFetchAddExplicit(volatile cl_int *object, cl_int operand, cl_memory_order o)
{
#if (defined(_WIN32) || defined(_WIN64)) && defined(_MSC_VER)
  return InterlockedExchangeAdd( (volatile LONG*) object, operand);
#elif defined(__GNUC__)
  return __sync_fetch_and_add(object, operand);
#else
  log_error("ERROR: AtomicFetchAddExplicit function not implemented\n");
  return -1;
#endif
}

cl_int AtomicExchangeExplicit(volatile cl_int *object, cl_int desired, cl_memory_order mo)
{
#if (defined(_WIN32) || defined(_WIN64)) && defined(_MSC_VER)
  return InterlockedExchange( (volatile LONG*) object, desired);
#elif defined(__GNUC__)
  return __sync_lock_test_and_set(object, desired);
#else
  log_error("ERROR: AtomicExchangeExplicit function not implemented\n");
  return -1;
#endif
}


const char *linked_list_create_and_verify_kernels[] = {
  "typedef struct Node {\n"
  "    int global_id;\n"
  "    int position_in_list;\n"
  "    __global struct Node* pNext;\n"
  "} Node;\n"
  "\n"
  // The allocation_index parameter must be initialized on the host to N work-items
  // The first N nodes in pNodes will be the heads of the lists.
  "__kernel void create_linked_lists(__global Node* pNodes, volatile __attribute__((nosvm)) __global int* allocation_index, int list_length)\n"
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
  "        pNew = &pNodes[ atomic_inc(allocation_index) ];// allocate a new node\n"
  "        pNew->global_id = i;\n"
  "        pNew->position_in_list = j;\n"
  "        pNode->pNext = pNew;  // link new node onto end of list\n"
  "        pNode = pNew;   // move to end of list\n"
  "    }\n"
  "}\n"

  "__kernel void verify_linked_lists(__global Node* pNodes, volatile __global uint* num_correct, int list_length)\n"
  "{\n"
  "    size_t i = get_global_id(0);\n"
  "    __global Node *pNode = &pNodes[i];\n"
  "\n"
  "    for(int j=0; j < list_length; j++)\n"
  "    {\n"
  "        if( pNode->global_id == i && pNode->position_in_list == j)\n"
  "        {\n"
  "            atomic_inc(num_correct);\n"
  "        } \n"
  "        else {\n"
  "            break;\n"
  "        }\n"
  "        pNode = pNode->pNext;\n"
  "    }\n"
  "}\n"
};


// The first N nodes in pNodes will be the heads of the lists.
void create_linked_lists(Node* pNodes, size_t num_lists, int list_length)
{
  size_t allocation_index = num_lists;  // heads of lists are in first num_lists nodes.

  for(cl_uint i = 0; i < num_lists; i++)
  {
    Node *pNode = &pNodes[i];
    pNode->global_id = i;
    pNode->position_in_list = 0;
    Node *pNew;
    for(int j=1; j < list_length; j++)
    {
      pNew = &pNodes[ allocation_index++ ];// allocate a new node
      pNew->global_id = i;
      pNew->position_in_list = j;
      pNode->pNext = pNew;  // link new node onto end of list
      pNode = pNew;   // move to end of list
    }
  }
}

cl_int verify_linked_lists(Node* pNodes, size_t num_lists, int list_length)
{
  cl_int error = CL_SUCCESS;
  int numCorrect = 0;

  log_info(" and verifying on host ");
  for(cl_uint i=0; i < num_lists; i++)
  {
    Node *pNode = &pNodes[i];
    for(int j=0; j < list_length; j++)
    {
      if( pNode->global_id == i && pNode->position_in_list == j)
      {
        numCorrect++;
      }
      else {
        break;
      }
      pNode = pNode->pNext;
    }
  }
  if(numCorrect != list_length * (cl_uint)num_lists)
  {
    error = -1;
    log_info("Failed\n");
  }
  else
    log_info("Passed\n");

  return error;
}

// Note that we don't use the context provided by the test harness since it doesn't support multiple devices,
// so we create are own context here that has all devices, we use the same platform that the harness used.
cl_int create_cl_objects(cl_device_id device_from_harness, const char** ppCodeString, cl_context* context, cl_program *program, cl_command_queue *queues, cl_uint *num_devices, cl_device_svm_capabilities required_svm_caps, std::vector<std::string> extensions_list)
{
  cl_int error;

  cl_platform_id platform_id;
  // find out what platform the harness is using.
  error = clGetDeviceInfo(device_from_harness, CL_DEVICE_PLATFORM,sizeof(cl_platform_id),&platform_id,NULL);
  test_error(error,"clGetDeviceInfo failed");

  error = clGetDeviceIDs(platform_id,  CL_DEVICE_TYPE_ALL, 0, NULL, num_devices );
  test_error(error, "clGetDeviceIDs failed");

  std::vector<cl_device_id> devicesTmp(*num_devices), devices, capable_devices;

  error = clGetDeviceIDs(platform_id,  CL_DEVICE_TYPE_ALL, *num_devices, &devicesTmp[0], NULL );
  test_error(error, "clGetDeviceIDs failed");

  devices.push_back(device_from_harness);
  for (size_t i = 0; i < devicesTmp.size(); ++i)
  {
    if (device_from_harness != devicesTmp[i])
      devices.push_back(devicesTmp[i]);
  }

  // Select only the devices that support the SVM level needed for the test.
  // Note that if requested SVM capabilities are not supported by any device then the test still passes (even though it does not execute).
  cl_device_svm_capabilities caps;
  cl_uint num_capable_devices = 0;
  for(cl_uint i = 0; i < *num_devices; i++)
  {
    Version version = get_device_cl_version(devices[i]);

    if(device_from_harness != devices[i] && version < Version(2,0))
    {
      continue;
    }

    error = clGetDeviceInfo(devices[i], CL_DEVICE_SVM_CAPABILITIES, sizeof(cl_device_svm_capabilities), &caps, NULL);
    test_error(error,"clGetDeviceInfo failed for CL_DEVICE_SVM_CAPABILITIES");
    if(caps & (~(CL_DEVICE_SVM_COARSE_GRAIN_BUFFER | CL_DEVICE_SVM_FINE_GRAIN_BUFFER |  CL_DEVICE_SVM_FINE_GRAIN_SYSTEM | CL_DEVICE_SVM_ATOMICS)))
    {
      log_error("clGetDeviceInfo returned an invalid cl_device_svm_capabilities value");
      return -1;
    }
    bool extensions_supported = true;
    for (auto extension : extensions_list) 
    {
      if (!is_extension_available(devices[i], extension.c_str())) 
      {
        log_error("Required extension not found - device id %d - %s\n", i, extension.c_str());
        extensions_supported = false;
        break;
      }
    }
    if((caps & required_svm_caps) == required_svm_caps && extensions_supported)
    {
      capable_devices.push_back(devices[i]);
      ++num_capable_devices;
    }
  }
  devices = capable_devices;  // the only devices we care about from here on are the ones capable of supporting the requested SVM level.
  *num_devices = num_capable_devices;
  if(num_capable_devices == 0)
    //    if(svm_level > CL_DEVICE_COARSE_SVM && 0 == num_capable_devices)
  {
    log_info("Requested SVM level or required extensions not supported by any device on this platform, test not executed.\n");
    return 1; // 1 indicates do not execute, but counts as passing.
  }

  cl_context_properties context_properties[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platform_id, 0 };
  *context = clCreateContext(context_properties, *num_devices, &devices[0], NULL, NULL, &error);
  test_error(error, "Unable to create context" );

  //    *queues = (cl_command_queue *) malloc( *num_devices * sizeof( cl_command_queue ) );

  for(cl_uint i = 0; i < *num_devices; i++)
  {
    queues[i] = clCreateCommandQueueWithProperties(*context, devices[i], 0, &error);
    test_error(error, "clCreateCommandQueue failed");
  }

  if(ppCodeString)
  {
    error = create_single_kernel_helper(*context, program, 0, 1, ppCodeString, 0, "-cl-std=CL2.0");
    test_error( error, "failed to create program" );
  }

  return 0;
}

test_definition test_list[] = {
    ADD_TEST( svm_byte_granularity),
    ADD_TEST( svm_set_kernel_exec_info_svm_ptrs ),
    ADD_TEST( svm_fine_grain_memory_consistency ),
    ADD_TEST( svm_fine_grain_sync_buffers ),
    ADD_TEST( svm_shared_address_space_fine_grain ),
    ADD_TEST( svm_shared_sub_buffers ),
    ADD_TEST( svm_shared_address_space_fine_grain_buffers ),
    ADD_TEST( svm_allocate_shared_buffer ),
    ADD_TEST( svm_shared_address_space_coarse_grain_old_api ),
    ADD_TEST( svm_shared_address_space_coarse_grain_new_api ),
    ADD_TEST( svm_cross_buffer_pointers_coarse_grain ),
    ADD_TEST( svm_pointer_passing ),
    ADD_TEST( svm_enqueue_api ),
    ADD_TEST_VERSION( svm_migrate, Version(2, 1)),
};

const int test_num = ARRAY_SIZE( test_list );

test_status InitCL(cl_device_id device) {
  auto version = get_device_cl_version(device);
  auto expected_min_version = Version(2, 0);
  if (version < expected_min_version) {
    version_expected_info("Test", expected_min_version.to_string().c_str(), version.to_string().c_str());
    return TEST_SKIP;
  }

  int error;
  cl_device_svm_capabilities svm_caps;
  error = clGetDeviceInfo(device, CL_DEVICE_SVM_CAPABILITIES,
                          sizeof(svm_caps), &svm_caps, NULL);
  if (error != CL_SUCCESS) {
    print_error(error, "Unable to get svm capabilities");
    return TEST_FAIL;
  }

  if ((svm_caps == 0) && (version >= Version(3, 0)))
  {
      return TEST_SKIP;
  }

  return TEST_PASS;
}

int main(int argc, const char *argv[])
{
  return runTestHarnessWithCheck(argc, argv, test_num, test_list, true, 0, InitCL);
}

