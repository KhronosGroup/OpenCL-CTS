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
#include <stdio.h>
#include <stdlib.h>

#include "harness/testHarness.h"
#include "utils.h"
#include "procs.h"


test_definition test_list[] = {
ADD_TEST( context_create ),
ADD_TEST( get_device_ids ),
ADD_TEST( api ),
ADD_TEST( kernel ),
ADD_TEST( other_data_types ),
ADD_TEST( memory_access ),
ADD_TEST( interop_user_sync )
};

const int test_num = ARRAY_SIZE(test_list);

clGetDeviceIDsFromDX9MediaAdapterKHR_fn clGetDeviceIDsFromDX9MediaAdapterKHR = NULL;
clCreateFromDX9MediaSurfaceKHR_fn clCreateFromDX9MediaSurfaceKHR = NULL;
clEnqueueAcquireDX9MediaSurfacesKHR_fn clEnqueueAcquireDX9MediaSurfacesKHR = NULL;
clEnqueueReleaseDX9MediaSurfacesKHR_fn clEnqueueReleaseDX9MediaSurfacesKHR = NULL;

cl_platform_id gPlatformIDdetected;
cl_device_id gDeviceIDdetected;
cl_device_type gDeviceTypeSelected = CL_DEVICE_TYPE_DEFAULT;

bool MediaSurfaceSharingExtensionInit()
{
  clGetDeviceIDsFromDX9MediaAdapterKHR = (clGetDeviceIDsFromDX9MediaAdapterKHR_fn)clGetExtensionFunctionAddressForPlatform(gPlatformIDdetected, "clGetDeviceIDsFromDX9MediaAdapterKHR");
  if (clGetDeviceIDsFromDX9MediaAdapterKHR == NULL)
  {
    log_error("clGetExtensionFunctionAddressForPlatform(clGetDeviceIDsFromDX9MediaAdapterKHR) returned NULL.\n");
    return false;
  }

  clCreateFromDX9MediaSurfaceKHR = (clCreateFromDX9MediaSurfaceKHR_fn)clGetExtensionFunctionAddressForPlatform(gPlatformIDdetected, "clCreateFromDX9MediaSurfaceKHR");
  if (clCreateFromDX9MediaSurfaceKHR == NULL)
  {
    log_error("clGetExtensionFunctionAddressForPlatform(clCreateFromDX9MediaSurfaceKHR) returned NULL.\n");
    return false;
  }

  clEnqueueAcquireDX9MediaSurfacesKHR = (clEnqueueAcquireDX9MediaSurfacesKHR_fn)clGetExtensionFunctionAddressForPlatform(gPlatformIDdetected, "clEnqueueAcquireDX9MediaSurfacesKHR");
  if (clEnqueueAcquireDX9MediaSurfacesKHR == NULL)
  {
    log_error("clGetExtensionFunctionAddressForPlatform(clEnqueueAcquireDX9MediaSurfacesKHR) returned NULL.\n");
    return false;
  }

  clEnqueueReleaseDX9MediaSurfacesKHR = (clEnqueueReleaseDX9MediaSurfacesKHR_fn)clGetExtensionFunctionAddressForPlatform(gPlatformIDdetected, "clEnqueueReleaseDX9MediaSurfacesKHR");
  if (clEnqueueReleaseDX9MediaSurfacesKHR == NULL)
  {
    log_error("clGetExtensionFunctionAddressForPlatform(clEnqueueReleaseDX9MediaSurfacesKHR) returned NULL.\n");
    return false;
  }

  return true;
}

bool DetectPlatformAndDevice()
{
  std::vector<cl_platform_id> platforms;
  cl_uint platformsNum = 0;
  cl_int error = clGetPlatformIDs(0, 0, &platformsNum);
  if (error != CL_SUCCESS)
  {
    print_error(error, "clGetPlatformIDs failed\n");
    return false;
  }

  platforms.resize(platformsNum);
  error = clGetPlatformIDs(platformsNum, &platforms[0], 0);
  if (error != CL_SUCCESS)
  {
    print_error(error, "clGetPlatformIDs failed\n");
    return false;
  }

  bool found = false;
  for (size_t i = 0; i < platformsNum; ++i)
  {
    std::vector<cl_device_id> devices;
    cl_uint devicesNum = 0;
    error = clGetDeviceIDs(platforms[i], gDeviceTypeSelected, 0, 0, &devicesNum);
    if (error != CL_SUCCESS)
    {
      print_error(error, "clGetDeviceIDs failed\n");
      return false;
    }

    devices.resize(devicesNum);
    error = clGetDeviceIDs(platforms[i], gDeviceTypeSelected, devicesNum, &devices[0], 0);
    if (error != CL_SUCCESS)
    {
      print_error(error, "clGetDeviceIDs failed\n");
      return false;
    }

    for (size_t j = 0; j < devicesNum; ++j)
    {
      if (is_extension_available(devices[j], "cl_khr_dx9_media_sharing"))
      {
        gPlatformIDdetected = platforms[i];
        gDeviceIDdetected = devices[j];
        found = true;
        break;
      }
    }
  }

  if (!found)
  {
    log_info("Test was not run, because the media surface sharing extension is not supported for any devices.\n");
    return false;
  }

  return true;
}

bool CmdlineParse(int argc, const char *argv[])
{
  char *env_mode = getenv( "CL_DEVICE_TYPE" );
  if( env_mode != NULL )
  {
    if(strcmp(env_mode, "gpu") == 0 || strcmp(env_mode, "CL_DEVICE_TYPE_GPU") == 0)
      gDeviceTypeSelected = CL_DEVICE_TYPE_GPU;
    else if(strcmp(env_mode, "cpu") == 0 || strcmp(env_mode, "CL_DEVICE_TYPE_CPU") == 0)
      gDeviceTypeSelected = CL_DEVICE_TYPE_CPU;
    else if(strcmp(env_mode, "accelerator") == 0 || strcmp(env_mode, "CL_DEVICE_TYPE_ACCELERATOR") == 0)
      gDeviceTypeSelected = CL_DEVICE_TYPE_ACCELERATOR;
    else if(strcmp(env_mode, "default") == 0 || strcmp(env_mode, "CL_DEVICE_TYPE_DEFAULT") == 0)
      gDeviceTypeSelected = CL_DEVICE_TYPE_DEFAULT;
    else
    {
      log_error("Unknown CL_DEVICE_TYPE env variable setting: %s.\nAborting...\n", env_mode);
      return false;
    }
  }

  for (int i = 0; i < argc; ++i)
  {
    if(strcmp(argv[i], "gpu") == 0 || strcmp(argv[i], "CL_DEVICE_TYPE_GPU") == 0)
    {
      gDeviceTypeSelected = CL_DEVICE_TYPE_GPU;
      continue;
    }
    else if(strcmp( argv[i], "cpu") == 0 || strcmp(argv[i], "CL_DEVICE_TYPE_CPU") == 0)
    {
      gDeviceTypeSelected = CL_DEVICE_TYPE_CPU;
      continue;
    }
    else if(strcmp( argv[i], "accelerator") == 0 || strcmp(argv[i], "CL_DEVICE_TYPE_ACCELERATOR") == 0)
    {
      gDeviceTypeSelected = CL_DEVICE_TYPE_ACCELERATOR;
      continue;
    }
    else if(strcmp(argv[i], "CL_DEVICE_TYPE_DEFAULT") == 0)
    {
      gDeviceTypeSelected = CL_DEVICE_TYPE_DEFAULT;
      continue;
    }
    else if (strcmp(argv[i], "sw") == 0 || strcmp(argv[i], "software") == 0)
    {
      CDeviceWrapper::AccelerationType(CDeviceWrapper::ACCELERATION_SW);
    }
  }

  return true;
}

int main(int argc, const char *argv[])
{
  if (!CmdlineParse(argc, argv))
    return TEST_FAIL;

  if (!DetectPlatformAndDevice())
  {
    log_info("Test was not run, because the media surface sharing extension is not supported\n");
    return TEST_SKIP;
  }

  if (!MediaSurfaceSharingExtensionInit())
    return TEST_FAIL;

  return runTestHarness(argc, argv, test_num, test_list, false, true, 0);
}
