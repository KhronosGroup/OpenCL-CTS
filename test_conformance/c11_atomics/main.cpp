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
#include "harness/testHarness.h"
#include <iostream>
#include <string>

bool gHost = false; // flag for testing native host threads (test verification)
bool gOldAPI = false; // flag for testing with old API (OpenCL 1.2) - test verification
bool gContinueOnError = false; // execute all cases even when errors detected
bool gNoGlobalVariables = false; // disable cases with global atomics in program scope
bool gNoGenericAddressSpace = false; // disable cases with generic address space
bool gUseHostPtr = false; // use malloc/free with CL_MEM_USE_HOST_PTR instead of clSVMAlloc/clSVMFree
bool gDebug = false; // always print OpenCL kernel code
int gInternalIterations = 10000; // internal test iterations for atomic operation, sufficient to verify atomicity
int gMaxDeviceThreads = 1024; // maximum number of threads executed on OCL device

extern int test_atomic_init(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_atomic_store(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_atomic_load(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_atomic_store_load(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_atomic_exchange(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_atomic_compare_exchange_weak(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_atomic_compare_exchange_strong(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_atomic_fetch_add(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_atomic_fetch_sub(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_atomic_fetch_and(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_atomic_fetch_or(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_atomic_fetch_orand(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_atomic_fetch_xor(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_atomic_fetch_xor2(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_atomic_fetch_min(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_atomic_fetch_max(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_atomic_flag(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_atomic_fence(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);

extern int test_svm_atomic_init(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_svm_atomic_store(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_svm_atomic_load(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_svm_atomic_store_load(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_svm_atomic_exchange(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_svm_atomic_compare_exchange_weak(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_svm_atomic_compare_exchange_strong(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_svm_atomic_fetch_add(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_svm_atomic_fetch_sub(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_svm_atomic_fetch_and(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_svm_atomic_fetch_or(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_svm_atomic_fetch_orand(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_svm_atomic_fetch_xor(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_svm_atomic_fetch_xor2(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_svm_atomic_fetch_min(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_svm_atomic_fetch_max(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_svm_atomic_flag(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_svm_atomic_fence(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);

test_definition test_list[] = {
    ADD_TEST( atomic_init ),
    ADD_TEST( atomic_store ),
    ADD_TEST( atomic_load ),
    ADD_TEST( atomic_exchange ),
    ADD_TEST( atomic_compare_exchange_weak ),
    ADD_TEST( atomic_compare_exchange_strong ),
    ADD_TEST( atomic_fetch_add ),
    ADD_TEST( atomic_fetch_sub ),
    ADD_TEST( atomic_fetch_and ),
    ADD_TEST( atomic_fetch_or ),
    ADD_TEST( atomic_fetch_orand ),
    ADD_TEST( atomic_fetch_xor ),
    ADD_TEST( atomic_fetch_xor2 ),
    ADD_TEST( atomic_fetch_min ),
    ADD_TEST( atomic_fetch_max ),
    ADD_TEST( atomic_flag ),
    ADD_TEST( atomic_fence ),

    ADD_TEST( svm_atomic_init ),
    ADD_TEST( svm_atomic_store ),
    ADD_TEST( svm_atomic_load ),
    ADD_TEST( svm_atomic_exchange ),
    ADD_TEST( svm_atomic_compare_exchange_weak ),
    ADD_TEST( svm_atomic_compare_exchange_strong ),
    ADD_TEST( svm_atomic_fetch_add ),
    ADD_TEST( svm_atomic_fetch_sub ),
    ADD_TEST( svm_atomic_fetch_and ),
    ADD_TEST( svm_atomic_fetch_or ),
    ADD_TEST( svm_atomic_fetch_orand ),
    ADD_TEST( svm_atomic_fetch_xor ),
    ADD_TEST( svm_atomic_fetch_xor2 ),
    ADD_TEST( svm_atomic_fetch_min ),
    ADD_TEST( svm_atomic_fetch_max ),
    ADD_TEST( svm_atomic_flag ),
    ADD_TEST( svm_atomic_fence ),
};

const int test_num = ARRAY_SIZE( test_list );

test_status InitCL(cl_device_id device) {
    auto version = get_device_cl_version(device);
    auto expected_min_version = Version(2, 0);
    if (version < expected_min_version)
    {
        version_expected_info("Test", expected_min_version.to_string().c_str(), version.to_string().c_str());
        return TEST_SKIP;
    }
    return TEST_PASS;
}

int main(int argc, const char *argv[])
{
  bool noCert = false;
  while(true)
  {
    if(std::string(argv[argc-1]) == "-h")
    {
      log_info("Test options:\n");
      log_info("  '-host'                    flag for testing native host threads (test verification)\n");
      log_info("  '-oldAPI'                  flag for testing with old API (OpenCL 1.2) - test verification\n");
      log_info("  '-continueOnError'         execute all cases even when errors detected\n");
      log_info("  '-noGlobalVariables'       disable cases with global atomics in program scope\n");
      log_info("  '-noGenericAddressSpace'   disable cases with generic address space\n");
      log_info("  '-useHostPtr'              use malloc/free with CL_MEM_USE_HOST_PTR instead of clSVMAlloc/clSVMFree\n");
      log_info("  '-debug'                   always print OpenCL kernel code\n");
      log_info("  '-internalIterations <X>'  internal test iterations for atomic operation, sufficient to verify atomicity\n");
      log_info("  '-maxDeviceThreads <X>'    maximum number of threads executed on OCL device");

      break;
    }
    if(std::string(argv[argc-1]) == "-host") // temporary option for testing native host threads
    {
      gHost = true;
      noCert = true;
    }
    else if(std::string(argv[argc-1]) == "-oldAPI") // temporary flag for testing with old API (OpenCL 1.2)
    {
      gOldAPI = true;
      gNoGlobalVariables = true;
      gNoGenericAddressSpace = true;
      gUseHostPtr = true;
      noCert = true;
    }
    else if(std::string(argv[argc-1]) == "-continueOnError") // execute all cases even when errors detected
      gContinueOnError = true;
    else if(std::string(argv[argc-1]) == "-noGlobalVariables") // disable cases with global atomics in program scope
    {
      gNoGlobalVariables = true;
      noCert = true;
    }
    else if(std::string(argv[argc-1]) == "-noGenericAddressSpace") // disable cases with generic address space
    {
      gNoGenericAddressSpace = true;
      noCert = true;
    }
    else if(std::string(argv[argc-1]) == "-useHostPtr") // use malloc/free with CL_MEM_USE_HOST_PTR instead of clSVMAlloc/clSVMFree
    {
      gUseHostPtr = true;
      noCert = true;
    }
    else if(std::string(argv[argc-1]) == "-debug") // print OpenCL kernel code
      gDebug = true;
    else if(argc > 2 && std::string(argv[argc-2]) == "-internalIterations") // internal test iterations for atomic operation, sufficient to verify atomicity
    {
      gInternalIterations = atoi(argv[argc-1]);
      if(gInternalIterations < 1)
      {
        log_info("Invalid value: Number of internal iterations (%d) must be > 0\n", gInternalIterations);
        return -1;
      }
      argc--;
      noCert = true;
    }
    else if(argc > 2 && std::string(argv[argc-2]) == "-maxDeviceThreads") // maximum number of threads executed on OCL device
    {
      gMaxDeviceThreads = atoi(argv[argc-1]);
      argc--;
      noCert = true;
    }
    else
      break;
    argc--;
  }
  if(noCert)
  {
    log_info("\n" );
    log_info("***                                                                        ***\n");
    log_info("*** WARNING: Test execution in debug mode (forced by command-line option)! ***\n");
    log_info("*** Use of this mode is not sufficient to verify correctness.              ***\n");
    log_info("***                                                                        ***\n");
  }
  return runTestHarnessWithCheck(argc, argv, test_num, test_list, false, false, InitCL);
}
