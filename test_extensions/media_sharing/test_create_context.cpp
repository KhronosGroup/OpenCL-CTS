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
#include "utils.h"

int context_create(cl_device_id deviceID, cl_context context, cl_command_queue queue,
                   int num_elements, unsigned int width, unsigned int height,
                   TContextFuncType functionCreate, cl_dx9_media_adapter_type_khr adapterType,
                   TSurfaceFormat surfaceFormat, TSharedHandleType sharedHandle)
{
  CResult result;

  //create device
  std::auto_ptr<CDeviceWrapper> deviceWrapper;
  if (!DeviceCreate(adapterType, deviceWrapper))
  {
    result.ResultSub(CResult::TEST_ERROR);
    return result.Result();
  }

  //generate input data
  std::vector<cl_uchar> bufferIn(width * height * 3 / 2, 0);
  if(!YUVGenerate(surfaceFormat, bufferIn, width, height, 0, 255))
  {
    result.ResultSub(CResult::TEST_ERROR);
    return result.Result();
  }

  while (deviceWrapper->AdapterNext())
  {
    cl_int error;
    //check if the test can be run on the adapter
    if (CL_SUCCESS != (error = deviceExistForCLTest(gPlatformIDdetected, adapterType, deviceWrapper->Device(), result, sharedHandle)))
    {
      return result.Result();
    }

    if (surfaceFormat != SURFACE_FORMAT_NV12 && !SurfaceFormatCheck(adapterType, *deviceWrapper, surfaceFormat))
    {
      std::string sharedHandleStr = (sharedHandle == SHARED_HANDLE_ENABLED)? "yes": "no";
      std::string formatStr;
      std::string adapterStr;
      SurfaceFormatToString(surfaceFormat, formatStr);
      AdapterToString(adapterType, adapterStr);
      log_info("Skipping test case, image format is not supported by a device (adapter type: %s, format: %s, shared handle: %s)\n",
        adapterStr.c_str(), formatStr.c_str(), sharedHandleStr.c_str());
      return result.Result();
    }

    void *objectSharedHandle = 0;
    std::auto_ptr<CSurfaceWrapper> surface;
    if (!MediaSurfaceCreate(adapterType, width, height, surfaceFormat, *deviceWrapper, surface,
      (sharedHandle == SHARED_HANDLE_ENABLED) ? true: false, &objectSharedHandle))
    {
      log_error("Media surface creation failed for %i adapter\n", deviceWrapper->AdapterIdx());
      result.ResultSub(CResult::TEST_ERROR);
      return result.Result();
    }

    cl_context_properties contextProperties[] = {
      CL_CONTEXT_PLATFORM, (cl_context_properties)gPlatformIDdetected,
      AdapterTypeToContextInfo(adapterType), (cl_context_properties)deviceWrapper->Device(),
      0,
    };

    clContextWrapper ctx;
    switch(functionCreate)
    {
    case CONTEXT_CREATE_DEFAULT:
      ctx = clCreateContext(&contextProperties[0], 1, &gDeviceIDdetected, NULL, NULL, &error);
      break;
    case CONTEXT_CREATE_FROM_TYPE:
      ctx = clCreateContextFromType(&contextProperties[0], gDeviceTypeSelected, NULL, NULL, &error);
      break;
    default:
      log_error("Unknown context creation function enum\n");
      result.ResultSub(CResult::TEST_ERROR);
      return result.Result();
      break;
    }

    if (error != CL_SUCCESS)
    {
      std::string functionName;
      FunctionContextCreateToString(functionCreate, functionName);
      log_error("%s failed: %s\n", functionName.c_str(), IGetErrorString(error));
      result.ResultSub(CResult::TEST_FAIL);
      return result.Result();
    }

    if (!YUVSurfaceSet(surfaceFormat, surface, bufferIn, width, height))
    {
      result.ResultSub(CResult::TEST_ERROR);
      return result.Result();
    }

#if defined(_WIN32)
    cl_dx9_surface_info_khr surfaceInfo;
    surfaceInfo.resource = *(static_cast<CD3D9SurfaceWrapper *>(surface.get()));
    surfaceInfo.shared_handle = objectSharedHandle;
#else
    void *surfaceInfo = 0;
    return TEST_NOT_IMPLEMENTED;
#endif

    std::vector<cl_mem> memObjList;
    unsigned int planesNum = PlanesNum(surfaceFormat);
    std::vector<clMemWrapper> planesList(planesNum);
    for (unsigned int planeIdx = 0; planeIdx < planesNum; ++planeIdx)
    {
      planesList[planeIdx] = clCreateFromDX9MediaSurfaceKHR(ctx, CL_MEM_READ_WRITE, adapterType, &surfaceInfo, planeIdx, &error);
      if (error != CL_SUCCESS)
      {
        log_error("clCreateFromDX9MediaSurfaceKHR failed for plane %i: %s\n", planeIdx, IGetErrorString(error));
        result.ResultSub(CResult::TEST_FAIL);
        return result.Result();
      }
      memObjList.push_back(planesList[planeIdx]);
    }

    clCommandQueueWrapper cmdQueue = clCreateCommandQueueWithProperties(ctx, gDeviceIDdetected, 0, &error );
    if (error != CL_SUCCESS)
    {
      log_error("Unable to create command queue: %s\n", IGetErrorString(error));
      result.ResultSub(CResult::TEST_FAIL);
      return result.Result();
    }

    if (!ImageInfoVerify(adapterType, memObjList, width, height, surface, objectSharedHandle))
    {
      log_error("Image info verification failed\n");
      result.ResultSub(CResult::TEST_FAIL);
    }

    cl_event event;
    error = clEnqueueAcquireDX9MediaSurfacesKHR(cmdQueue, static_cast<cl_uint>(memObjList.size()),
      &memObjList.at(0), 0, NULL, &event);
    if (error != CL_SUCCESS)
    {
      log_error("clEnqueueAcquireDX9MediaSurfacesKHR failed: %s\n", IGetErrorString(error));
      result.ResultSub(CResult::TEST_FAIL);
      return result.Result();
    }

    cl_uint eventType = 0;
    error = clGetEventInfo( event, CL_EVENT_COMMAND_TYPE, sizeof(eventType), &eventType, NULL);
    if (error != CL_SUCCESS)
    {
      log_error("clGetEventInfo failed: %s\n", IGetErrorString(error));
      result.ResultSub(CResult::TEST_FAIL);
    }

    if(eventType != CL_COMMAND_ACQUIRE_DX9_MEDIA_SURFACES_KHR)
    {
      log_error("Invalid event != CL_COMMAND_ACQUIRE_DX9_MEDIA_SURFACES_KHR\n");
      result.ResultSub(CResult::TEST_FAIL);
    }

    clReleaseEvent(event);

    size_t origin[3] = {0,0,0};
    size_t offset = 0;
    size_t frameSize = width * height * 3 / 2;
    std::vector<cl_uchar> out( frameSize, 0 );
    for (size_t i = 0; i < memObjList.size(); ++i)
    {
      size_t planeWidth = (i == 0) ? width: width / 2;
      size_t planeHeight = (i == 0) ? height: height / 2;
      size_t regionPlane[3] = {planeWidth, planeHeight, 1};

      error = clEnqueueReadImage(cmdQueue, memObjList.at(i), CL_TRUE, origin, regionPlane, 0, 0, &out.at(offset), 0, 0, 0);
      if (error != CL_SUCCESS)
      {
        log_error("clEnqueueReadImage failed: %s\n", IGetErrorString(error));
        result.ResultSub(CResult::TEST_FAIL);
      }

      offset += planeWidth * planeHeight;
    }

    if (!YUVCompare(surfaceFormat, out, bufferIn, width, height))
    {
      log_error("OCL object verification failed - clEnqueueReadImage\n");
      result.ResultSub(CResult::TEST_FAIL);
    }

    error = clEnqueueReleaseDX9MediaSurfacesKHR(cmdQueue, static_cast<cl_uint>(memObjList.size()),
      &memObjList.at(0), 0, NULL, &event);
    if (error != CL_SUCCESS)
    {
      log_error("clEnqueueReleaseDX9MediaSurfacesKHR failed: %s\n", IGetErrorString(error));
      result.ResultSub(CResult::TEST_FAIL);
    }

    eventType = 0;
    error = clGetEventInfo( event, CL_EVENT_COMMAND_TYPE, sizeof(eventType), &eventType, NULL);
    if (error != CL_SUCCESS)
    {
      log_error("clGetEventInfo failed: %s\n", IGetErrorString(error));
      result.ResultSub(CResult::TEST_FAIL);
    }

    if(eventType != CL_COMMAND_RELEASE_DX9_MEDIA_SURFACES_KHR)
    {
      log_error("Invalid event != CL_COMMAND_RELEASE_DX9_MEDIA_SURFACES_KHR\n");
      result.ResultSub(CResult::TEST_FAIL);
    }

    clReleaseEvent(event);

    //object verification
    std::vector<cl_uchar> bufferOut(frameSize, 0);
    if (!YUVSurfaceGet(surfaceFormat, surface, bufferOut, width, height))
    {
      result.ResultSub(CResult::TEST_FAIL);
      return result.Result();
    }

    if (!YUVCompare(surfaceFormat, bufferOut, bufferIn, width, height))
    {
      log_error("Media surface is different than expected\n");
      result.ResultSub(CResult::TEST_FAIL);
      return result.Result();
    }
  }

  if (deviceWrapper->Status() != DEVICE_PASS)
  {
    std::string adapterName;
    AdapterToString(adapterType, adapterName);
    if (deviceWrapper->Status() == DEVICE_FAIL)
    {
    log_error("%s init failed\n", adapterName.c_str());
    result.ResultSub(CResult::TEST_FAIL);
    }
    else
    {
      log_error("%s init incomplete due to unsupported device\n", adapterName.c_str());
      result.ResultSub(CResult::TEST_NOTSUPPORTED);
    }
  }

  return result.Result();
}

int test_context_create(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
  const unsigned int WIDTH = 256;
  const unsigned int HEIGHT = 256;

  std::vector<cl_dx9_media_adapter_type_khr> adapterTypes;
#if defined(_WIN32)
  adapterTypes.push_back(CL_ADAPTER_D3D9_KHR);
  adapterTypes.push_back(CL_ADAPTER_D3D9EX_KHR);
  adapterTypes.push_back(CL_ADAPTER_DXVA_KHR);
#endif

  std::vector<TContextFuncType> contextFuncs;
  contextFuncs.push_back(CONTEXT_CREATE_DEFAULT);
  contextFuncs.push_back(CONTEXT_CREATE_FROM_TYPE);

  std::vector<TSurfaceFormat> formats;
  formats.push_back(SURFACE_FORMAT_NV12);
  formats.push_back(SURFACE_FORMAT_YV12);

  std::vector<TSharedHandleType> sharedHandleTypes;
  sharedHandleTypes.push_back(SHARED_HANDLE_DISABLED);
#if defined(_WIN32)
  sharedHandleTypes.push_back(SHARED_HANDLE_ENABLED);
#endif

  CResult result;
  for (size_t adapterTypeIdx = 0; adapterTypeIdx < adapterTypes.size(); ++adapterTypeIdx)
  {
    //iteration through all create context functions
    for (size_t contextFuncIdx = 0; contextFuncIdx < contextFuncs.size(); ++contextFuncIdx)
    {
      //iteration through surface formats
      for (size_t formatIdx = 0; formatIdx < formats.size(); ++formatIdx)
      {
        //shared handle enabled or disabled
        for (size_t sharedHandleIdx = 0; sharedHandleIdx < sharedHandleTypes.size(); ++sharedHandleIdx)
        {
          if (adapterTypes[adapterTypeIdx] == CL_ADAPTER_D3D9_KHR && sharedHandleTypes[sharedHandleIdx] == SHARED_HANDLE_ENABLED)
            continue;

          if(context_create(deviceID, context, queue, num_elements, WIDTH, HEIGHT,
            contextFuncs[contextFuncIdx], adapterTypes[adapterTypeIdx], formats[formatIdx],
            sharedHandleTypes[sharedHandleIdx]) != 0)
          {
            std::string sharedHandle = (sharedHandleTypes[sharedHandleIdx] == SHARED_HANDLE_ENABLED)? "shared handle": "no shared handle";
            std::string formatStr;
            std::string adapterTypeStr;
            SurfaceFormatToString(formats[formatIdx], formatStr);
            AdapterToString(adapterTypes[adapterTypeIdx], adapterTypeStr);

            log_error("\nTest case - clCreateContext (%s, %s, %s) failed\n\n", adapterTypeStr.c_str(), formatStr.c_str(), sharedHandle.c_str());
            result.ResultSub(CResult::TEST_FAIL);
          }
        }
      }
    }
  }

  return result.Result();
}
