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

int api_functions(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements,
                  unsigned int iterationNum, unsigned int width, unsigned int height, cl_dx9_media_adapter_type_khr adapterType,
                  TSurfaceFormat surfaceFormat, TSharedHandleType sharedHandle)
{
  const unsigned int FRAME_NUM = 2;
  const cl_uchar MAX_VALUE = 255 / 2;
  CResult result;

  //create device
  std::auto_ptr<CDeviceWrapper> deviceWrapper;
  if (!DeviceCreate(adapterType, deviceWrapper))
  {
    result.ResultSub(CResult::TEST_ERROR);
    return result.Result();
  }

  //generate input and expected data
  std::vector<std::vector<cl_uchar> > bufferRef1(FRAME_NUM);
  std::vector<std::vector<cl_uchar> > bufferRef2(FRAME_NUM);
  std::vector<std::vector<cl_uchar> > bufferRef3(FRAME_NUM);
  size_t frameSize = width * height * 3 / 2;
  cl_uchar step = MAX_VALUE / FRAME_NUM;
  for (size_t i = 0; i < FRAME_NUM; ++i)
  {
    if (!YUVGenerate(surfaceFormat, bufferRef1[i], width, height, static_cast<cl_uchar>(step * i), static_cast<cl_uchar>(step * (i + 1))) ||
        !YUVGenerate(surfaceFormat, bufferRef2[i], width, height, static_cast<cl_uchar>(step * i), static_cast<cl_uchar>(step * (i + 1)), 0.2) ||
        !YUVGenerate(surfaceFormat, bufferRef3[i], width, height, static_cast<cl_uchar>(step * i), static_cast<cl_uchar>(step * (i + 1)), 0.4))
    {
      result.ResultSub(CResult::TEST_ERROR);
      return result.Result();
    }
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

    //create surface
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

    clContextWrapper ctx = clCreateContext(&contextProperties[0], 1, &gDeviceIDdetected, NULL, NULL, &error);
    if (error != CL_SUCCESS)
    {
      log_error("clCreateContext failed: %s\n", IGetErrorString(error));
      result.ResultSub(CResult::TEST_FAIL);
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

    for (size_t frameIdx = 0; frameIdx < iterationNum; ++frameIdx)
    {
      if (!YUVSurfaceSet(surfaceFormat, surface, bufferRef1[frameIdx % FRAME_NUM], width, height))
      {
        result.ResultSub(CResult::TEST_ERROR);
        return result.Result();
      }

      error = clEnqueueAcquireDX9MediaSurfacesKHR(cmdQueue, static_cast<cl_uint>(memObjList.size()), &memObjList[0], 0, 0, 0);
      if (error != CL_SUCCESS)
      {
        log_error("clEnqueueAcquireDX9MediaSurfacesKHR failed: %s\n", IGetErrorString(error));
        result.ResultSub(CResult::TEST_FAIL);
        return result.Result();
      }

      { //read operation
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

        if (!YUVCompare(surfaceFormat, out, bufferRef1[frameIdx % FRAME_NUM], width, height))
        {
          log_error("Frame idx: %i, OCL image is different then shared OCL object: clEnqueueReadImage\n", frameIdx);
          result.ResultSub(CResult::TEST_FAIL);
        }
      }

      { //write operation
        size_t offset = 0;
        size_t origin[3] = {0,0,0};
        for (size_t i = 0; i < memObjList.size(); ++i)
        {
          size_t planeWidth = (i == 0) ? width: width / 2;
          size_t planeHeight = (i == 0) ? height: height / 2;
          size_t regionPlane[3] = {planeWidth, planeHeight, 1};

          error = clEnqueueWriteImage(cmdQueue, memObjList[i], CL_TRUE, origin, regionPlane,
            0, 0, &bufferRef2[frameIdx % FRAME_NUM][offset], 0, 0, 0);
          if (error != CL_SUCCESS)
          {
            log_error("clEnqueueWriteImage failed: %s\n", IGetErrorString(error));
            result.ResultSub(CResult::TEST_FAIL);
          }

          offset += planeWidth * planeHeight;
        }
      }

      { //read operation
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

        if (!YUVCompare(surfaceFormat, out, bufferRef2[frameIdx % FRAME_NUM], width, height))
        {
          log_error("Frame idx: %i, Shared OCL image verification after clEnqueueWriteImage failed\n", frameIdx);
          result.ResultSub(CResult::TEST_FAIL);
        }
      }

      { //copy operation (shared OCL to OCL)
        size_t offset = 0;
        size_t origin[3] = {0,0,0};
        std::vector<cl_uchar> out( frameSize, 0 );
        for (size_t i = 0; i < memObjList.size(); ++i)
        {
          size_t planeWidth = (i == 0) ? width: width / 2;
          size_t planeHeight = (i == 0) ? height: height / 2;
          size_t regionPlane[3] = {planeWidth, planeHeight, 1};

          cl_image_format formatPlane;
          formatPlane.image_channel_data_type = CL_UNORM_INT8;
          formatPlane.image_channel_order = (surfaceFormat == SURFACE_FORMAT_NV12 && i > 0)? CL_RG: CL_R;

          cl_image_desc imageDesc = {0};
          imageDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
          imageDesc.image_width = planeWidth;
          imageDesc.image_height = planeHeight;

          clMemWrapper planeOCL = clCreateImage(ctx, CL_MEM_READ_WRITE, &formatPlane, &imageDesc, 0, &error);
          if (error != CL_SUCCESS)
          {
            log_error("clCreateImage failed: %s\n", IGetErrorString(error));
            result.ResultSub(CResult::TEST_FAIL);
          }

          error = clEnqueueCopyImage(cmdQueue, memObjList[i], planeOCL, origin, origin, regionPlane, 0, 0, 0);
          if (error != CL_SUCCESS)
          {
            log_error("clEnqueueCopyImage failed: %s\n", IGetErrorString(error));
            result.ResultSub(CResult::TEST_FAIL);
          }

          error = clEnqueueReadImage(cmdQueue, planeOCL, CL_TRUE, origin, regionPlane, 0, 0, &out[offset], 0, 0, 0);
          if (error != CL_SUCCESS)
          {
            log_error("clEnqueueReadImage failed: %s\n", IGetErrorString(error));
            result.ResultSub(CResult::TEST_FAIL);
          }

          offset += planeWidth * planeHeight;
        }

        if (!YUVCompare(surfaceFormat, out, bufferRef2[frameIdx % FRAME_NUM], width, height))
        {
          log_error("Frame idx: %i, OCL image verification after clEnqueueCopyImage (from shared OCL to OCL) failed\n", frameIdx);
          result.ResultSub(CResult::TEST_FAIL);
        }
      }

      { //copy operation (OCL to shared OCL)
        size_t offset = 0;
        size_t origin[3] = {0,0,0};
        std::vector<cl_uchar> out( frameSize, 0 );
        for (size_t i = 0; i < memObjList.size(); ++i)
        {
          size_t planeWidth = (i == 0) ? width: width / 2;
          size_t planeHeight = (i == 0) ? height: height / 2;
          size_t regionPlane[3] = {planeWidth, planeHeight, 1};
          size_t pitchSize = ((surfaceFormat == SURFACE_FORMAT_NV12 && i > 0)? width: planeWidth) * sizeof(cl_uchar);

          cl_image_format formatPlane;
          formatPlane.image_channel_data_type = CL_UNORM_INT8;
          formatPlane.image_channel_order = (surfaceFormat == SURFACE_FORMAT_NV12 && i > 0)? CL_RG: CL_R;

          cl_image_desc imageDesc = {0};
          imageDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
          imageDesc.image_width = planeWidth;
          imageDesc.image_height = planeHeight;
          imageDesc.image_row_pitch = pitchSize;

          clMemWrapper planeOCL = clCreateImage(ctx, CL_MEM_COPY_HOST_PTR, &formatPlane, &imageDesc, &bufferRef1[frameIdx % FRAME_NUM][offset], &error);
          if (error != CL_SUCCESS)
          {
            log_error("clCreateImage failed: %s\n", IGetErrorString(error));
            result.ResultSub(CResult::TEST_FAIL);
          }

          error = clEnqueueCopyImage(cmdQueue, planeOCL, memObjList[i], origin, origin, regionPlane, 0, 0, 0);
          if (error != CL_SUCCESS)
          {
            log_error("clEnqueueCopyImage failed: %s\n", IGetErrorString(error));
            result.ResultSub(CResult::TEST_FAIL);
          }

          error = clEnqueueReadImage(cmdQueue, memObjList[i], CL_TRUE, origin, regionPlane, 0, 0, &out[offset], 0, 0, 0);
          if (error != CL_SUCCESS)
          {
            log_error("clEnqueueReadImage failed: %s\n", IGetErrorString(error));
            result.ResultSub(CResult::TEST_FAIL);
          }

          offset += planeWidth * planeHeight;
        }

        if (!YUVCompare(surfaceFormat, out, bufferRef1[frameIdx % FRAME_NUM], width, height))
        {
          log_error("Frame idx: %i, OCL image verification after clEnqueueCopyImage (from OCL to shared OCL) failed\n", frameIdx);
          result.ResultSub(CResult::TEST_FAIL);
        }
      }

      { //copy from image to buffer
        size_t offset = 0;
        size_t origin[3] = {0,0,0};
        size_t bufferSize = sizeof(cl_uchar) * frameSize;
        clMemWrapper buffer = clCreateBuffer( ctx, CL_MEM_READ_WRITE, bufferSize, NULL, &error);
        if (error != CL_SUCCESS)
        {
          log_error("clCreateBuffer failed: %s\n", IGetErrorString(error));
          result.ResultSub(CResult::TEST_FAIL);
        }

        for (size_t i = 0; i < memObjList.size(); ++i)
        {
          size_t planeWidth = (i == 0) ? width: width / 2;
          size_t planeHeight = (i == 0) ? height: height / 2;
          size_t regionPlane[3] = {planeWidth, planeHeight, 1};

          error = clEnqueueCopyImageToBuffer(cmdQueue, memObjList[i], buffer, origin, regionPlane, offset, 0, 0, 0);
          if (error != CL_SUCCESS)
          {
            log_error("clEnqueueCopyImageToBuffer failed: %s\n", IGetErrorString(error));
            result.ResultSub(CResult::TEST_FAIL);
          }

          offset += planeWidth * planeHeight * sizeof(cl_uchar);
        }

        std::vector<cl_uchar> out( frameSize, 0 );
        error = clEnqueueReadBuffer( cmdQueue, buffer, CL_TRUE, 0, bufferSize, &out[0], 0, NULL, NULL );
        if (error != CL_SUCCESS)
        {
          log_error("Unable to read buffer");
          result.ResultSub(CResult::TEST_FAIL);
        }

        if (!YUVCompare(surfaceFormat, out, bufferRef1[frameIdx % FRAME_NUM], width, height))
        {
          log_error("Frame idx: %i, OCL buffer verification after clEnqueueCopyImageToBuffer (from shared OCL image to OCL buffer) failed\n", frameIdx);
          result.ResultSub(CResult::TEST_FAIL);
        }
      }

      { //copy buffer to image
        size_t bufferSize = sizeof(cl_uchar) * frameSize;
        clMemWrapper buffer = clCreateBuffer( ctx, CL_MEM_COPY_HOST_PTR, bufferSize, &bufferRef2[frameIdx % FRAME_NUM][0], &error);
        if (error != CL_SUCCESS)
        {
          log_error("clCreateBuffer failed: %s\n", IGetErrorString(error));
          result.ResultSub(CResult::TEST_FAIL);
        }

        size_t offset = 0;
        size_t origin[3] = {0,0,0};
        std::vector<cl_uchar> out( frameSize, 0 );
        for (size_t i = 0; i < memObjList.size(); ++i)
        {
          size_t planeWidth = (i == 0) ? width: width / 2;
          size_t planeHeight = (i == 0) ? height: height / 2;
          size_t regionPlane[3] = {planeWidth, planeHeight, 1};

          error = clEnqueueCopyBufferToImage(cmdQueue, buffer, memObjList[i], offset, origin, regionPlane, 0, 0, 0);
          if (error != CL_SUCCESS)
          {
            log_error("clEnqueueCopyBufferToImage failed: %s\n", IGetErrorString(error));
            result.ResultSub(CResult::TEST_FAIL);
          }

          error = clEnqueueReadImage(cmdQueue, memObjList[i], CL_TRUE, origin, regionPlane, 0, 0, &out[offset], 0, 0, 0);
          if (error != CL_SUCCESS)
          {
            log_error("clEnqueueReadImage failed: %s\n", IGetErrorString(error));
            result.ResultSub(CResult::TEST_FAIL);
          }

          offset += planeWidth * planeHeight * sizeof(cl_uchar);
        }

        if (!YUVCompare(surfaceFormat, out, bufferRef2[frameIdx % FRAME_NUM], width, height))
        {
          log_error("Frame idx: %i, OCL image verification after clEnqueueCopyBufferToImage (from OCL buffer to shared OCL image) failed\n", frameIdx);
          result.ResultSub(CResult::TEST_FAIL);
        }
      }

      { //map operation to read
        size_t offset = 0;
        size_t origin[3] = {0,0,0};
        std::vector<cl_uchar> out( frameSize, 0 );
        for (size_t i = 0; i < memObjList.size(); ++i)
        {
          size_t planeWidth = (i == 0) ? width: width / 2;
          size_t planeHeight = (i == 0) ? height: height / 2;
          size_t regionPlane[3] = {planeWidth, planeHeight, 1};
          size_t pitchSize = ((surfaceFormat == SURFACE_FORMAT_NV12 && i > 0)? width: planeWidth);

          size_t rowPitch = 0;
          size_t slicePitch = 0;
          void *mapPtr = clEnqueueMapImage(cmdQueue, memObjList[i], CL_TRUE, CL_MAP_READ, origin, regionPlane,
            &rowPitch, &slicePitch, 0, 0, 0, &error);
          if (error != CL_SUCCESS)
          {
            log_error("clEnqueueMapImage failed: %s\n", IGetErrorString(error));
            result.ResultSub(CResult::TEST_FAIL);
          }

          for (size_t y = 0; y < planeHeight; ++y)
            memcpy(&out[offset + y * pitchSize], static_cast<cl_uchar *>(mapPtr) + y * rowPitch / sizeof(cl_uchar), pitchSize * sizeof(cl_uchar));

          error = clEnqueueUnmapMemObject(cmdQueue, memObjList[i], mapPtr, 0, 0, 0);
          if (error != CL_SUCCESS)
          {
            log_error("clEnqueueUnmapMemObject failed: %s\n", IGetErrorString(error));
            result.ResultSub(CResult::TEST_FAIL);
          }

          offset += pitchSize * planeHeight;
        }

        if (!YUVCompare(surfaceFormat, out, bufferRef2[frameIdx % FRAME_NUM], width, height))
        {
          log_error("Frame idx: %i, Mapped shared OCL image is different then expected\n", frameIdx);
          result.ResultSub(CResult::TEST_FAIL);
        }
      }

      { //map operation to write
        size_t offset = 0;
        size_t origin[3] = {0,0,0};
        for (size_t i = 0; i < memObjList.size(); ++i)
        {
          size_t planeWidth = (i == 0) ? width: width / 2;
          size_t planeHeight = (i == 0) ? height: height / 2;
          size_t regionPlane[3] = {planeWidth, planeHeight, 1};
          size_t pitchSize = ((surfaceFormat == SURFACE_FORMAT_NV12 && i > 0)? width: planeWidth);

          size_t rowPitch = 0;
          size_t slicePitch = 0;
          void *mapPtr = clEnqueueMapImage(cmdQueue, memObjList[i], CL_TRUE, CL_MAP_WRITE, origin, regionPlane,
            &rowPitch, &slicePitch, 0, 0, 0, &error);
          if (error != CL_SUCCESS)
          {
            log_error("clEnqueueMapImage failed: %s\n", IGetErrorString(error));
            result.ResultSub(CResult::TEST_FAIL);
          }

          for (size_t y = 0; y < planeHeight; ++y)
            memcpy(static_cast<cl_uchar *>(mapPtr) + y * rowPitch / sizeof(cl_uchar), &bufferRef3[frameIdx % FRAME_NUM][offset + y * pitchSize], pitchSize * sizeof(cl_uchar));

          error = clEnqueueUnmapMemObject(cmdQueue, memObjList[i], mapPtr, 0, 0, 0);
          if (error != CL_SUCCESS)
          {
            log_error("clEnqueueUnmapMemObject failed: %s\n", IGetErrorString(error));
            result.ResultSub(CResult::TEST_FAIL);
          }

          offset += pitchSize * planeHeight;
        }
      }

      error = clEnqueueReleaseDX9MediaSurfacesKHR(cmdQueue, static_cast<cl_uint>(memObjList.size()), &memObjList[0], 0, 0, 0);
      if (error != CL_SUCCESS)
      {
        log_error("clEnqueueReleaseDX9MediaSurfacesKHR failed: %s\n", IGetErrorString(error));
        result.ResultSub(CResult::TEST_FAIL);
      }

      std::vector<cl_uchar> bufferOut(frameSize, 0);
      if (!YUVSurfaceGet(surfaceFormat, surface, bufferOut, width, height))
      {
        result.ResultSub(CResult::TEST_FAIL);
        return result.Result();
      }

      if (!YUVCompare(surfaceFormat, bufferOut, bufferRef3[frameIdx % FRAME_NUM], width, height))
      {
        log_error("Frame idx: %i, media surface is different than expected\n", frameIdx);
        result.ResultSub(CResult::TEST_FAIL);
      }
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

int test_api(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
  CResult result;

#if defined(_WIN32)
  //D3D9
  if(api_functions(deviceID, context, queue, num_elements, 10, 256, 256, CL_ADAPTER_D3D9_KHR,
    SURFACE_FORMAT_NV12, SHARED_HANDLE_DISABLED) != 0)
  {
    log_error("\nTest case (D3D9, NV12, no shared handle) failed\n\n");
    result.ResultSub(CResult::TEST_FAIL);
  }

  if(api_functions(deviceID, context, queue, num_elements, 3, 512, 256, CL_ADAPTER_D3D9_KHR,
    SURFACE_FORMAT_YV12, SHARED_HANDLE_DISABLED) != 0)
  {
    log_error("\nTest case (D3D9, YV12, no shared handle) failed\n\n");
    result.ResultSub(CResult::TEST_FAIL);
  }

  //D3D9EX
  if(api_functions(deviceID, context, queue, num_elements, 5, 256, 512, CL_ADAPTER_D3D9EX_KHR,
    SURFACE_FORMAT_NV12, SHARED_HANDLE_DISABLED) != 0)
  {
    log_error("\nTest case (D3D9EX, NV12, no shared handle) failed\n\n");
    result.ResultSub(CResult::TEST_FAIL);
  }

  if(api_functions(deviceID, context, queue, num_elements, 7, 512, 256, CL_ADAPTER_D3D9EX_KHR,
    SURFACE_FORMAT_NV12, SHARED_HANDLE_ENABLED) != 0)
  {
    log_error("\nTest case (D3D9EX, NV12, shared handle) failed\n\n");
    result.ResultSub(CResult::TEST_FAIL);
  }

  if(api_functions(deviceID, context, queue, num_elements, 10, 256, 256, CL_ADAPTER_D3D9EX_KHR,
    SURFACE_FORMAT_YV12, SHARED_HANDLE_DISABLED) != 0)
  {
    log_error("\nTest case (D3D9EX, YV12, no shared handle) failed\n\n");
    result.ResultSub(CResult::TEST_FAIL);
  }

  if(api_functions(deviceID, context, queue, num_elements, 15, 128, 128, CL_ADAPTER_D3D9EX_KHR,
    SURFACE_FORMAT_YV12, SHARED_HANDLE_ENABLED) != 0)
  {
    log_error("\nTest case (D3D9EX, YV12, shared handle) failed\n\n");
    result.ResultSub(CResult::TEST_FAIL);
  }

  //DXVA
  if(api_functions(deviceID, context, queue, num_elements, 20, 128, 128, CL_ADAPTER_DXVA_KHR,
    SURFACE_FORMAT_NV12, SHARED_HANDLE_DISABLED) != 0)
  {
    log_error("\nTest case (DXVA, NV12, no shared handle) failed\n\n");
    result.ResultSub(CResult::TEST_FAIL);
  }

  if(api_functions(deviceID, context, queue, num_elements, 40, 64, 64, CL_ADAPTER_DXVA_KHR,
    SURFACE_FORMAT_NV12, SHARED_HANDLE_ENABLED) != 0)
  {
    log_error("\nTest case (DXVA, NV12, shared handle) failed\n\n");
    result.ResultSub(CResult::TEST_FAIL);
  }

  if(api_functions(deviceID, context, queue, num_elements, 5, 512, 512, CL_ADAPTER_DXVA_KHR,
    SURFACE_FORMAT_YV12, SHARED_HANDLE_DISABLED) != 0)
  {
    log_error("\nTest case (DXVA, YV12, no shared handle) failed\n\n");
    result.ResultSub(CResult::TEST_FAIL);
  }

  if(api_functions(deviceID, context, queue, num_elements, 2, 1024, 1024, CL_ADAPTER_DXVA_KHR,
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
