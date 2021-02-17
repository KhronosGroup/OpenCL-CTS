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

int memory_access(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements,
                  unsigned int width, unsigned int height, cl_dx9_media_adapter_type_khr adapterType,
                  TSurfaceFormat surfaceFormat, TSharedHandleType sharedHandle)
{
  CResult result;

  std::auto_ptr<CDeviceWrapper> deviceWrapper;
  //creates device
  if (!DeviceCreate(adapterType, deviceWrapper))
  {
    result.ResultSub(CResult::TEST_ERROR);
    return result.Result();
  }

  //generate input and expected data
  size_t frameSize = width * height * 3 / 2;
  std::vector<cl_uchar> bufferRef0(frameSize, 0);
  std::vector<cl_uchar> bufferRef1(frameSize, 0);
  std::vector<cl_uchar> bufferRef2(frameSize, 0);
  if (!YUVGenerate(surfaceFormat, bufferRef0, width, height, 0, 90) ||
    !YUVGenerate(surfaceFormat, bufferRef1, width, height, 91, 180) ||
    !YUVGenerate(surfaceFormat, bufferRef2, width, height, 181, 255))
  {
    result.ResultSub(CResult::TEST_ERROR);
    return result.Result();
  }

  //iterates through all devices
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

    //creates surface
    if (!MediaSurfaceCreate(adapterType, width, height, surfaceFormat, *deviceWrapper, surface,
      (sharedHandle == SHARED_HANDLE_ENABLED) ? true: false, &objectSharedHandle))
    {
      log_error("Media surface creation failed for %i adapter\n", deviceWrapper->AdapterIdx());
      result.ResultSub(CResult::TEST_ERROR);
      return result.Result();
    }

    if (!YUVSurfaceSet(surfaceFormat, surface, bufferRef0, width, height))
    {
      result.ResultSub(CResult::TEST_ERROR);
      return result.Result();
    }

    cl_context_properties contextProperties[] = {
      CL_CONTEXT_PLATFORM, (cl_context_properties)gPlatformIDdetected,
      AdapterTypeToContextInfo(adapterType), (cl_context_properties)deviceWrapper->Device(),
      0,
    };

    clContextWrapper ctx = clCreateContext(&contextProperties[0], 1, &gDeviceIDdetected, NULL, NULL, &error);
    if (error != CL_SUCCESS)
    {
      log_error("clCreateContext failed: %s\n", IGetErrorString(error));
      result.ResultSub(CResult::TEST_FAIL);
      return result.Result();
    }

    clCommandQueueWrapper cmdQueue = clCreateCommandQueueWithProperties(ctx, gDeviceIDdetected, 0, &error );
    if (error != CL_SUCCESS)
    {
      log_error("Unable to create command queue: %s\n", IGetErrorString(error));
      result.ResultSub(CResult::TEST_FAIL);
      return result.Result();
    }

    { //memory access write
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
        planesList[planeIdx] = clCreateFromDX9MediaSurfaceKHR(ctx, CL_MEM_WRITE_ONLY, adapterType, &surfaceInfo, planeIdx, &error);
        if (error != CL_SUCCESS)
        {
          log_error("clCreateFromDX9MediaSurfaceKHR failed for WRITE_ONLY plane %i: %s\n", planeIdx, IGetErrorString(error));
          result.ResultSub(CResult::TEST_FAIL);
          return result.Result();
        }
        memObjList.push_back(planesList[planeIdx]);
      }

      error = clEnqueueAcquireDX9MediaSurfacesKHR(cmdQueue, static_cast<cl_uint>(memObjList.size()), &memObjList[0], 0, 0, 0);
      if (error != CL_SUCCESS)
      {
        log_error("clEnqueueAcquireDX9MediaSurfacesKHR failed: %s\n", IGetErrorString(error));
        result.ResultSub(CResult::TEST_FAIL);
        return result.Result();
      }

      size_t offset = 0;
      size_t origin[3] = {0,0,0};
      for (size_t i = 0; i < memObjList.size(); ++i)
      {
        size_t planeWidth = (i == 0) ? width: width / 2;
        size_t planeHeight = (i == 0) ? height: height / 2;
        size_t regionPlane[3] = {planeWidth, planeHeight, 1};

        error = clEnqueueWriteImage(cmdQueue, memObjList[i], CL_TRUE, origin, regionPlane,
          0, 0, &bufferRef1[offset], 0, 0, 0);
        if (error != CL_SUCCESS)
        {
          log_error("clEnqueueWriteImage failed: %s\n", IGetErrorString(error));
          result.ResultSub(CResult::TEST_FAIL);
        }

        offset += planeWidth * planeHeight;
      }

      error = clEnqueueReleaseDX9MediaSurfacesKHR(cmdQueue, static_cast<cl_uint>(memObjList.size()), &memObjList[0], 0, 0, 0);
      if (error != CL_SUCCESS)
      {
        log_error("clEnqueueReleaseDX9MediaSurfacesKHR failed: %s\n", IGetErrorString(error));
        result.ResultSub(CResult::TEST_FAIL);
      }
    }

    std::vector<cl_uchar> bufferOut0(frameSize, 0);
    if (!YUVSurfaceGet(surfaceFormat, surface, bufferOut0, width, height))
    {
      result.ResultSub(CResult::TEST_FAIL);
      return result.Result();
    }

    if (!YUVCompare(surfaceFormat, bufferOut0, bufferRef1, width, height))
    {
      log_error("Media surface is different than expected\n");
      result.ResultSub(CResult::TEST_FAIL);
    }

    { //memory access read
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
        planesList[planeIdx] = clCreateFromDX9MediaSurfaceKHR(ctx, CL_MEM_READ_ONLY, adapterType, &surfaceInfo, planeIdx, &error);
        if (error != CL_SUCCESS)
        {
          log_error("clCreateFromDX9MediaSurfaceKHR failed for READ_ONLY plane %i: %s\n", planeIdx, IGetErrorString(error));
          result.ResultSub(CResult::TEST_FAIL);
          return result.Result();
        }
        memObjList.push_back(planesList[planeIdx]);
      }

      error = clEnqueueAcquireDX9MediaSurfacesKHR(cmdQueue, static_cast<cl_uint>(memObjList.size()), &memObjList[0], 0, 0, 0);
      if (error != CL_SUCCESS)
      {
        log_error("clEnqueueAcquireDX9MediaSurfacesKHR failed: %s\n", IGetErrorString(error));
        result.ResultSub(CResult::TEST_FAIL);
        return result.Result();
      }

      std::vector<cl_uchar> out( frameSize, 0 );
      size_t offset = 0;
      size_t origin[3] = {0,0,0};

      for (size_t i = 0; i < memObjList.size(); ++i)
      {
        size_t planeWidth = (i == 0) ? width: width / 2;
        size_t planeHeight = (i == 0) ? height: height / 2;
        size_t regionPlane[3] = {planeWidth, planeHeight, 1};

        error = clEnqueueReadImage(cmdQueue, memObjList[i], CL_TRUE, origin, regionPlane, 0, 0,
          &out[offset], 0, 0, 0);
        if (error != CL_SUCCESS)
        {
          log_error("clEnqueueReadImage failed: %s\n", IGetErrorString(error));
          result.ResultSub(CResult::TEST_FAIL);
        }

        offset += planeWidth * planeHeight;
      }

      if (!YUVCompare(surfaceFormat, out, bufferRef1, width, height))
      {
        log_error("OCL image (READ_ONLY) is different then expected\n");
        result.ResultSub(CResult::TEST_FAIL);
      }

      error = clEnqueueReleaseDX9MediaSurfacesKHR(cmdQueue, static_cast<cl_uint>(memObjList.size()), &memObjList[0], 0, 0, 0);
      if (error != CL_SUCCESS)
      {
        log_error("clEnqueueReleaseDX9MediaSurfacesKHR failed: %s\n", IGetErrorString(error));
        result.ResultSub(CResult::TEST_FAIL);
      }
    }

    std::vector<cl_uchar> bufferOut1(frameSize, 0);
    if (!YUVSurfaceGet(surfaceFormat, surface, bufferOut1, width, height))
    {
      result.ResultSub(CResult::TEST_FAIL);
      return result.Result();
    }

    if (!YUVCompare(surfaceFormat, bufferOut1, bufferRef1, width, height))
    {
      log_error("Media surface is different than expected\n");
      result.ResultSub(CResult::TEST_FAIL);
    }

    { //memory access read write
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
          log_error("clCreateFromDX9MediaSurfaceKHR failed for READ_WRITE plane %i: %s\n", planeIdx, IGetErrorString(error));
          result.ResultSub(CResult::TEST_FAIL);
          return result.Result();
        }
        memObjList.push_back(planesList[planeIdx]);
      }

      error = clEnqueueAcquireDX9MediaSurfacesKHR(cmdQueue, static_cast<cl_uint>(memObjList.size()), &memObjList[0], 0, 0, 0);
      if (error != CL_SUCCESS)
      {
        log_error("clEnqueueAcquireDX9MediaSurfacesKHR failed: %s\n", IGetErrorString(error));
        result.ResultSub(CResult::TEST_FAIL);
        return result.Result();
      }

      { //read
        std::vector<cl_uchar> out( frameSize, 0 );
        size_t offset = 0;
        size_t origin[3] = {0,0,0};

        for (size_t i = 0; i < memObjList.size(); ++i)
        {
          size_t planeWidth = (i == 0) ? width: width / 2;
          size_t planeHeight = (i == 0) ? height: height / 2;
          size_t regionPlane[3] = {planeWidth, planeHeight, 1};

          error = clEnqueueReadImage(cmdQueue, memObjList[i], CL_TRUE, origin, regionPlane, 0, 0,
            &out[offset], 0, 0, 0);
          if (error != CL_SUCCESS)
          {
            log_error("clEnqueueReadImage failed: %s\n", IGetErrorString(error));
            result.ResultSub(CResult::TEST_FAIL);
          }

          offset += planeWidth * planeHeight;
        }

        if (!YUVCompare(surfaceFormat, out, bufferRef1, width, height))
        {
          log_error("OCL image (READ_WRITE) is different then expected\n");
          result.ResultSub(CResult::TEST_FAIL);
        }
      }

      { //write
        size_t offset = 0;
        size_t origin[3] = {0,0,0};
        for (size_t i = 0; i < memObjList.size(); ++i)
        {
          size_t planeWidth = (i == 0) ? width: width / 2;
          size_t planeHeight = (i == 0) ? height: height / 2;
          size_t regionPlane[3] = {planeWidth, planeHeight, 1};

          error = clEnqueueWriteImage(cmdQueue, memObjList[i], CL_TRUE, origin, regionPlane,
            0, 0, &bufferRef2[offset], 0, 0, 0);
          if (error != CL_SUCCESS)
          {
            log_error("clEnqueueWriteImage failed: %s\n", IGetErrorString(error));
            result.ResultSub(CResult::TEST_FAIL);
          }

          offset += planeWidth * planeHeight;
        }
      }

      error = clEnqueueReleaseDX9MediaSurfacesKHR(cmdQueue, static_cast<cl_uint>(memObjList.size()), &memObjList[0], 0, 0, 0);
      if (error != CL_SUCCESS)
      {
        log_error("clEnqueueReleaseDX9MediaSurfacesKHR failed: %s\n", IGetErrorString(error));
        result.ResultSub(CResult::TEST_FAIL);
      }
    }

    std::vector<cl_uchar> bufferOut2(frameSize, 0);
    if (!YUVSurfaceGet(surfaceFormat, surface, bufferOut2, width, height))
    {
      result.ResultSub(CResult::TEST_FAIL);
      return result.Result();
    }

    if (!YUVCompare(surfaceFormat, bufferOut2, bufferRef2, width, height))
    {
      log_error("Media surface is different than expected\n");
      result.ResultSub(CResult::TEST_FAIL);
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

int test_memory_access(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
  CResult result;

#if defined(_WIN32)
  //D3D9
  if(memory_access(deviceID, context, queue, num_elements, 256, 256, CL_ADAPTER_D3D9_KHR,
    SURFACE_FORMAT_NV12, SHARED_HANDLE_DISABLED) != 0)
  {
    log_error("\nTest case (D3D9, NV12, no shared handle) failed\n\n");
    result.ResultSub(CResult::TEST_FAIL);
  }

  if(memory_access(deviceID, context, queue, num_elements, 512, 256, CL_ADAPTER_D3D9_KHR,
    SURFACE_FORMAT_YV12, SHARED_HANDLE_DISABLED) != 0)
  {
    log_error("\nTest case (D3D9, YV12, no shared handle) failed\n\n");
    result.ResultSub(CResult::TEST_FAIL);
  }

  //D3D9EX
  if(memory_access(deviceID, context, queue, num_elements, 256, 512, CL_ADAPTER_D3D9EX_KHR,
    SURFACE_FORMAT_NV12, SHARED_HANDLE_DISABLED) != 0)
  {
    log_error("\nTest case (D3D9EX, NV12, no shared handle) failed\n\n");
    result.ResultSub(CResult::TEST_FAIL);
  }

  if(memory_access(deviceID, context, queue, num_elements, 512, 256, CL_ADAPTER_D3D9EX_KHR,
    SURFACE_FORMAT_NV12, SHARED_HANDLE_ENABLED) != 0)
  {
    log_error("\nTest case (D3D9EX, NV12, shared handle) failed\n\n");
    result.ResultSub(CResult::TEST_FAIL);
  }

  if(memory_access(deviceID, context, queue, num_elements, 256, 256, CL_ADAPTER_D3D9EX_KHR,
    SURFACE_FORMAT_YV12, SHARED_HANDLE_DISABLED) != 0)
  {
    log_error("\nTest case (D3D9EX, YV12, no shared handle) failed\n\n");
    result.ResultSub(CResult::TEST_FAIL);
  }

  if(memory_access(deviceID, context, queue, num_elements, 128, 128, CL_ADAPTER_D3D9EX_KHR,
    SURFACE_FORMAT_YV12, SHARED_HANDLE_ENABLED) != 0)
  {
    log_error("\nTest case (D3D9EX, YV12, shared handle) failed\n\n");
    result.ResultSub(CResult::TEST_FAIL);
  }

  //DXVA
  if(memory_access(deviceID, context, queue, num_elements, 128, 128, CL_ADAPTER_DXVA_KHR,
    SURFACE_FORMAT_NV12, SHARED_HANDLE_DISABLED) != 0)
  {
    log_error("\nTest case (DXVA, NV12, no shared handle) failed\n\n");
    result.ResultSub(CResult::TEST_FAIL);
  }

  if(memory_access(deviceID, context, queue, num_elements, 64, 64, CL_ADAPTER_DXVA_KHR,
    SURFACE_FORMAT_NV12, SHARED_HANDLE_ENABLED) != 0)
  {
    log_error("\nTest case (DXVA, NV12, shared handle) failed\n\n");
    result.ResultSub(CResult::TEST_FAIL);
  }

  if(memory_access(deviceID, context, queue, num_elements, 512, 512, CL_ADAPTER_DXVA_KHR,
    SURFACE_FORMAT_YV12, SHARED_HANDLE_DISABLED) != 0)
  {
    log_error("\nTest case (DXVA, YV12, no shared handle) failed\n\n");
    result.ResultSub(CResult::TEST_FAIL);
  }

  if(memory_access(deviceID, context, queue, num_elements, 1024, 1024, CL_ADAPTER_DXVA_KHR,
    SURFACE_FORMAT_YV12, SHARED_HANDLE_ENABLED) != 0)
  {
    log_error("\nTest case (DXVA, YV12, shared handle) failed\n\n");
    result.ResultSub(CResult::TEST_FAIL);
  }

#else
  return TEST_NOT_IMPLEMENTED;
#endif

  return result.Result();
}
