//
// Copyright (c) 2024 The Khronos Group Inc.
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
#include "harness/deviceInfo.h"
#include "harness/kernelHelpers.h"
#include <iostream>
#include <string>
#include "CL/cl_half.h"

bool gHost = false; // flag for testing native host threads (test verification)
bool gOldAPI = false; // flag for testing with old API (OpenCL 1.2) - test verification
bool gContinueOnError = false; // execute all cases even when errors detected
bool gNoGlobalVariables = false; // disable cases with global atomics in program scope
bool gNoGenericAddressSpace = false; // disable cases with generic address space
bool gUseHostPtr = false; // use malloc/free with CL_MEM_USE_HOST_PTR instead of clSVMAlloc/clSVMFree
bool gDebug = false; // always print OpenCL kernel code
int gInternalIterations = 10000; // internal test iterations for atomic operation, sufficient to verify atomicity
int gMaxDeviceThreads = 1024; // maximum number of threads executed on OCL device
cl_device_atomic_capabilities gAtomicMemCap,
    gAtomicFenceCap; // atomic memory and fence capabilities for this device
cl_half_rounding_mode gHalfRoundingMode = CL_HALF_RTE;
bool gFloatAtomicsSupported = false;
cl_device_fp_atomic_capabilities_ext gHalfAtomicCaps = 0;
cl_device_fp_atomic_capabilities_ext gDoubleAtomicCaps = 0;
cl_device_fp_atomic_capabilities_ext gFloatAtomicCaps = 0;
cl_device_fp_config gHalfCaps = 0;

test_status InitCL(cl_device_id device) {
    auto version = get_device_cl_version(device);
    auto expected_min_version = Version(2, 0);

    if (version < expected_min_version)
    {
        version_expected_info("Test", "OpenCL",
                              expected_min_version.to_string().c_str(),
                              version.to_string().c_str());
        return TEST_SKIP;
    }

    if (version >= Version(3, 0))
    {
        cl_int error;

        error = clGetDeviceInfo(device, CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES,
                                sizeof(gAtomicMemCap), &gAtomicMemCap, NULL);
        if (error != CL_SUCCESS)
        {
            print_error(error, "Unable to get atomic memory capabilities\n");
            return TEST_FAIL;
        }

        error =
            clGetDeviceInfo(device, CL_DEVICE_ATOMIC_FENCE_CAPABILITIES,
                            sizeof(gAtomicFenceCap), &gAtomicFenceCap, NULL);
        if (error != CL_SUCCESS)
        {
            print_error(error, "Unable to get atomic fence capabilities\n");
            return TEST_FAIL;
        }

        if ((gAtomicFenceCap
             & (CL_DEVICE_ATOMIC_ORDER_RELAXED | CL_DEVICE_ATOMIC_ORDER_ACQ_REL
                | CL_DEVICE_ATOMIC_SCOPE_WORK_GROUP))
            == 0)
        {
            log_info(
                "Minimum atomic fence capabilities unsupported by device\n");
            return TEST_FAIL;
        }

        if ((gAtomicMemCap
             & (CL_DEVICE_ATOMIC_ORDER_RELAXED
                | CL_DEVICE_ATOMIC_SCOPE_WORK_GROUP))
            == 0)
        {
            log_info(
                "Minimum atomic memory capabilities unsupported by device\n");
            return TEST_FAIL;
        }

        // Disable program scope global variable testing in the case that it is
        // not supported on an OpenCL-3.0 driver.
        size_t max_global_variable_size{};
        test_error_ret(clGetDeviceInfo(device,
                                       CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE,
                                       sizeof(max_global_variable_size),
                                       &max_global_variable_size, nullptr),
                       "Unable to get max global variable size\n", TEST_FAIL);
        if (0 == max_global_variable_size)
        {
            gNoGlobalVariables = true;
        }

        // Disable generic address space testing in the case that it is not
        // supported on an OpenCL-3.0 driver.
        cl_bool generic_address_space_support{};
        test_error_ret(
            clGetDeviceInfo(device, CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT,
                            sizeof(generic_address_space_support),
                            &generic_address_space_support, nullptr),
            "Unable to get generic address space support\n", TEST_FAIL);
        if (CL_FALSE == generic_address_space_support)
        {
            gNoGenericAddressSpace = true;
        }
    }
    else
    {
        // OpenCL 2.x device, default to all capabilities
        gAtomicMemCap = CL_DEVICE_ATOMIC_ORDER_RELAXED
            | CL_DEVICE_ATOMIC_ORDER_ACQ_REL | CL_DEVICE_ATOMIC_ORDER_SEQ_CST
            | CL_DEVICE_ATOMIC_SCOPE_WORK_GROUP | CL_DEVICE_ATOMIC_SCOPE_DEVICE
            | CL_DEVICE_ATOMIC_SCOPE_ALL_DEVICES;

        gAtomicFenceCap = CL_DEVICE_ATOMIC_ORDER_RELAXED
            | CL_DEVICE_ATOMIC_ORDER_ACQ_REL | CL_DEVICE_ATOMIC_ORDER_SEQ_CST
            | CL_DEVICE_ATOMIC_SCOPE_WORK_ITEM
            | CL_DEVICE_ATOMIC_SCOPE_WORK_GROUP | CL_DEVICE_ATOMIC_SCOPE_DEVICE
            | CL_DEVICE_ATOMIC_SCOPE_ALL_DEVICES;
    }

    if (is_extension_available(device, "cl_ext_float_atomics"))
    {
        gFloatAtomicsSupported = true;

        if (is_extension_available(device, "cl_khr_fp64"))
        {
            cl_int error = clGetDeviceInfo(
                device, CL_DEVICE_DOUBLE_FP_ATOMIC_CAPABILITIES_EXT,
                sizeof(gDoubleAtomicCaps), &gDoubleAtomicCaps, nullptr);
            test_error_ret(error, "clGetDeviceInfo failed!", TEST_FAIL);
        }

        cl_int error = clGetDeviceInfo(
            device, CL_DEVICE_SINGLE_FP_ATOMIC_CAPABILITIES_EXT,
            sizeof(gFloatAtomicCaps), &gFloatAtomicCaps, nullptr);
        test_error_ret(error, "clGetDeviceInfo failed!", TEST_FAIL);

        if (is_extension_available(device, "cl_khr_fp16"))
        {
            cl_int error = clGetDeviceInfo(
                device, CL_DEVICE_HALF_FP_ATOMIC_CAPABILITIES_EXT,
                sizeof(gHalfAtomicCaps), &gHalfAtomicCaps, nullptr);
            test_error_ret(error, "clGetDeviceInfo failed!", TEST_FAIL);

            const cl_device_fp_config fpConfigHalf =
                get_default_rounding_mode(device, CL_DEVICE_HALF_FP_CONFIG);
            if ((fpConfigHalf & CL_FP_ROUND_TO_NEAREST) != 0)
            {
                gHalfRoundingMode = CL_HALF_RTE;
            }
            else if ((fpConfigHalf & CL_FP_ROUND_TO_ZERO) != 0)
            {
                gHalfRoundingMode = CL_HALF_RTZ;
            }
            else
            {
                log_error("Error while acquiring half rounding mode\n");
                return TEST_FAIL;
            }

            error = clGetDeviceInfo(device, CL_DEVICE_HALF_FP_CONFIG,
                                    sizeof(gHalfCaps), &gHalfCaps, NULL);
            test_error_ret(error, "clGetDeviceInfo failed!", TEST_FAIL);
        }
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
  return runTestHarnessWithCheck(
      argc, argv, test_registry::getInstance().num_tests(),
      test_registry::getInstance().definitions(), false, false, InitCL);
}
