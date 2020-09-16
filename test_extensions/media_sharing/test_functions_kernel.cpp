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

#include "harness/errorHelpers.h"
#include "harness/kernelHelpers.h"

#include "utils.h"

int kernel_functions(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements,
                     unsigned int iterationNum, unsigned int width, unsigned int height, cl_dx9_media_adapter_type_khr adapterType,
                     TSurfaceFormat surfaceFormat, TSharedHandleType sharedHandle)
{
  const unsigned int FRAME_NUM = 2;
  const cl_uchar MAX_VALUE = 255 / 2;
  const std::string PROGRAM_STR =
    "__kernel void TestFunction( read_only image2d_t planeIn, write_only image2d_t planeOut, "
    NL "                            sampler_t sampler, __global int *planeRes)"
    NL "{"
    NL "  int w = get_global_id(0);"
    NL "  int h = get_global_id(1);"
    NL "  int width = get_image_width(planeIn);"
    NL "  int height = get_image_height(planeOut);"
    NL "  float4 color0 = read_imagef(planeIn, sampler, (int2)(w,h)) + 0.2f;"
    NL "  float4 color1 = read_imagef(planeIn, sampler, (float2)(w,h)) + 0.2f;"
    NL "  color0 = (color0 == color1) ? color0: (float4)(0.5, 0.5, 0.5, 0.5);"
    NL "  write_imagef(planeOut, (int2)(w,h), color0);"
    NL "  if(w == 0 && h == 0)"
    NL "  {"
    NL "    planeRes[0] = width;"
    NL "    planeRes[1] = height;"
    NL "  }"
    NL "}";

  CResult result;

  std::auto_ptr<CDeviceWrapper> deviceWrapper;
  if (!DeviceCreate(adapterType, deviceWrapper))
  {
    result.ResultSub(CResult::TEST_ERROR);
    return result.Result();
  }

  std::vector<std::vector<cl_uchar> > bufferIn(FRAME_NUM);
  std::vector<std::vector<cl_uchar> > bufferExp(FRAME_NUM);
  size_t frameSize = width * height * 3 / 2;
  cl_uchar step = MAX_VALUE / FRAME_NUM;
  for (size_t i = 0; i < FRAME_NUM; ++i)
  {
    if (!YUVGenerate(surfaceFormat, bufferIn[i], width, height, static_cast<cl_uchar>(step * i), static_cast<cl_uchar>(step * (i + 1))) ||
        !YUVGenerate(surfaceFormat, bufferExp[i], width, height, static_cast<cl_uchar>(step * i), static_cast<cl_uchar>(step * (i + 1)), 0.2))
    {
      result.ResultSub(CResult::TEST_ERROR);
      return result.Result();
    }
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

    void *objectSrcHandle = 0;
    std::auto_ptr<CSurfaceWrapper> surfaceSrc;
    if (!MediaSurfaceCreate(adapterType, width, height, surfaceFormat, *deviceWrapper, surfaceSrc,
      (sharedHandle == SHARED_HANDLE_ENABLED) ? true: false, &objectSrcHandle))
    {
      log_error("Media surface creation failed for %i adapter\n", deviceWrapper->AdapterIdx());
      result.ResultSub(CResult::TEST_ERROR);
      return result.Result();
    }

    void *objectDstHandle = 0;
    std::auto_ptr<CSurfaceWrapper> surfaceDst;
    if (!MediaSurfaceCreate(adapterType, width, height, surfaceFormat, *deviceWrapper, surfaceDst,
      (sharedHandle == SHARED_HANDLE_ENABLED) ? true: false, &objectDstHandle))
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
    cl_dx9_surface_info_khr surfaceInfoSrc;
    surfaceInfoSrc.resource = *(static_cast<CD3D9SurfaceWrapper *>(surfaceSrc.get()));
    surfaceInfoSrc.shared_handle = objectSrcHandle;

    cl_dx9_surface_info_khr surfaceInfoDst;
    surfaceInfoDst.resource = *(static_cast<CD3D9SurfaceWrapper *>(surfaceDst.get()));
    surfaceInfoDst.shared_handle = objectDstHandle;
#else
    void *surfaceInfoSrc = 0;
    void *surfaceInfoDst = 0;
    return TEST_NOT_IMPLEMENTED;
#endif

    std::vector<cl_mem> memObjSrcList;
    std::vector<cl_mem> memObjDstList;
    unsigned int planesNum = PlanesNum(surfaceFormat);
    std::vector<clMemWrapper> planeSrcList(planesNum);
    std::vector<clMemWrapper> planeDstList(planesNum);
    for (unsigned int planeIdx = 0; planeIdx < planesNum; ++planeIdx)
    {
      planeSrcList[planeIdx] = clCreateFromDX9MediaSurfaceKHR(ctx, CL_MEM_READ_WRITE, adapterType, &surfaceInfoSrc, planeIdx, &error);
      if (error != CL_SUCCESS)
      {
        log_error("clCreateFromDX9MediaSurfaceKHR failed for plane %i: %s\n", planeIdx, IGetErrorString(error));
        result.ResultSub(CResult::TEST_FAIL);
        return result.Result();
      }
      memObjSrcList.push_back(planeSrcList[planeIdx]);

      planeDstList[planeIdx] = clCreateFromDX9MediaSurfaceKHR(ctx, CL_MEM_READ_WRITE, adapterType, &surfaceInfoDst, planeIdx, &error);
      if (error != CL_SUCCESS)
      {
        log_error("clCreateFromDX9MediaSurfaceKHR failed for plane %i: %s\n", planeIdx, IGetErrorString(error));
        result.ResultSub(CResult::TEST_FAIL);
        return result.Result();
      }
      memObjDstList.push_back(planeDstList[planeIdx]);
    }

    clCommandQueueWrapper cmdQueue = clCreateCommandQueueWithProperties(ctx, gDeviceIDdetected, 0, &error );
    if (error != CL_SUCCESS)
    {
      log_error("Unable to create command queue: %s\n", IGetErrorString(error));
      result.ResultSub(CResult::TEST_FAIL);
      return result.Result();
    }

    if (!ImageInfoVerify(adapterType, memObjSrcList, width, height, surfaceSrc, objectSrcHandle))
    {
      log_error("Image info verification failed\n");
      result.ResultSub(CResult::TEST_FAIL);
    }

    for (size_t frameIdx = 0; frameIdx < iterationNum; ++frameIdx)
    {
      if (!YUVSurfaceSet(surfaceFormat, surfaceSrc, bufferIn[frameIdx % FRAME_NUM], width, height))
      {
        result.ResultSub(CResult::TEST_ERROR);
        return result.Result();
      }

      error = clEnqueueAcquireDX9MediaSurfacesKHR(cmdQueue, static_cast<cl_uint>(memObjSrcList.size()), &memObjSrcList[0], 0, 0, 0);
      if (error != CL_SUCCESS)
      {
        log_error("clEnqueueAcquireDX9MediaSurfacesKHR failed: %s\n", IGetErrorString(error));
        result.ResultSub(CResult::TEST_FAIL);
        return result.Result();
      }

      error = clEnqueueAcquireDX9MediaSurfacesKHR(cmdQueue, static_cast<cl_uint>(memObjDstList.size()), &memObjDstList[0], 0, 0, 0);
      if (error != CL_SUCCESS)
      {
        log_error("clEnqueueAcquireDX9MediaSurfacesKHR failed: %s\n", IGetErrorString(error));
        result.ResultSub(CResult::TEST_FAIL);
        return result.Result();
      }

      clSamplerWrapper sampler = clCreateSampler( ctx, CL_FALSE, CL_ADDRESS_NONE, CL_FILTER_NEAREST, &error );
      if(error != CL_SUCCESS)
      {
        log_error("Unable to create sampler\n");
        result.ResultSub(CResult::TEST_FAIL);
      }

      clProgramWrapper program;
      clKernelWrapper kernel;
      const char *progPtr = PROGRAM_STR.c_str();
      if(create_single_kernel_helper(ctx, &program, &kernel, 1, (const char **)&progPtr, "TestFunction"))
        result.ResultSub(CResult::TEST_FAIL);

      size_t bufferSize = sizeof(cl_int) * 2;
      clMemWrapper imageRes = clCreateBuffer( ctx, CL_MEM_READ_WRITE, bufferSize, NULL, &error);
      if (error != CL_SUCCESS)
      {
        log_error("clCreateBuffer failed: %s\n", IGetErrorString(error));
        result.ResultSub(CResult::TEST_FAIL);
      }

      size_t offset = 0;
      size_t origin[3] = {0,0,0};
      std::vector<cl_uchar> out( frameSize, 0 );
      for (size_t i = 0; i < memObjSrcList.size(); ++i)
      {
        size_t planeWidth = (i == 0) ? width: width / 2;
        size_t planeHeight = (i == 0) ? height: height / 2;
        size_t regionPlane[3] = {planeWidth, planeHeight, 1};
        size_t threads[ 2 ] = { planeWidth, planeHeight };

        error = clSetKernelArg( kernel, 0, sizeof( memObjSrcList[i] ), &memObjSrcList[i] );
        if (error != CL_SUCCESS)
        {
          log_error("Unable to set kernel arguments" );
          result.ResultSub(CResult::TEST_FAIL);
        }

        error = clSetKernelArg( kernel, 1, sizeof( memObjDstList[i] ), &memObjDstList[i] );
        if (error != CL_SUCCESS)
        {
          log_error("Unable to set kernel arguments" );
          result.ResultSub(CResult::TEST_FAIL);
        }

        error = clSetKernelArg( kernel, 2, sizeof( sampler ), &sampler );
        if (error != CL_SUCCESS)
        {
          log_error("Unable to set kernel arguments" );
          result.ResultSub(CResult::TEST_FAIL);
        }

        error = clSetKernelArg( kernel, 3, sizeof( imageRes ), &imageRes );
        if (error != CL_SUCCESS)
        {
          log_error("Unable to set kernel arguments" );
          result.ResultSub(CResult::TEST_FAIL);
        }

        size_t localThreads[ 2 ];
        error = get_max_common_2D_work_group_size( ctx, kernel, threads, localThreads );
        if (error != CL_SUCCESS)
        {
          log_error("Unable to get work group size to use" );
          result.ResultSub(CResult::TEST_FAIL);
        }

        error = clEnqueueNDRangeKernel( cmdQueue, kernel, 2, NULL, threads, localThreads, 0, NULL, NULL );
        if (error != CL_SUCCESS)
        {
          log_error("Unable to execute test kernel" );
          result.ResultSub(CResult::TEST_FAIL);
        }

        std::vector<cl_uint> imageResOut(2, 0);
        error = clEnqueueReadBuffer( cmdQueue, imageRes, CL_TRUE, 0, bufferSize, &imageResOut[0], 0, NULL, NULL );
        if (error != CL_SUCCESS)
        {
          log_error("Unable to read buffer");
          result.ResultSub(CResult::TEST_FAIL);
        }

        if(imageResOut[0] != planeWidth)
        {
          log_error("Invalid width value, test = %i, expected = %i\n", imageResOut[0], planeWidth);
          result.ResultSub(CResult::TEST_FAIL);
        }

        if(imageResOut[1] != planeHeight)
        {
          log_error("Invalid height value, test = %i, expected = %i\n", imageResOut[1], planeHeight);
          result.ResultSub(CResult::TEST_FAIL);
        }

        error = clEnqueueReadImage(cmdQueue, memObjDstList[i], CL_TRUE, origin, regionPlane, 0, 0, &out[offset], 0, 0, 0);
        if (error != CL_SUCCESS)
        {
          log_error("clEnqueueReadImage failed: %s\n", IGetErrorString(error));
          result.ResultSub(CResult::TEST_FAIL);
        }

        offset += planeWidth * planeHeight;
      }

      if (!YUVCompare(surfaceFormat, out, bufferExp[frameIdx % FRAME_NUM], width, height))
      {
        log_error("Frame idx: %i, OCL objects are different than expected\n", frameIdx);
        result.ResultSub(CResult::TEST_FAIL);
      }

      error = clEnqueueReleaseDX9MediaSurfacesKHR(cmdQueue, static_cast<cl_uint>(memObjSrcList.size()), &memObjSrcList[0], 0, 0, 0);
      if (error != CL_SUCCESS)
      {
        log_error("clEnqueueReleaseDX9MediaSurfacesKHR failed: %s\n", IGetErrorString(error));
        result.ResultSub(CResult::TEST_FAIL);
      }

      error = clEnqueueReleaseDX9MediaSurfacesKHR(cmdQueue, static_cast<cl_uint>(memObjDstList.size()), &memObjDstList[0], 0, 0, 0);
      if (error != CL_SUCCESS)
      {
        log_error("clEnqueueReleaseDX9MediaSurfacesKHR failed: %s\n", IGetErrorString(error));
        result.ResultSub(CResult::TEST_FAIL);
      }

      std::vector<cl_uchar> bufferOut(frameSize, 0);
      if (!YUVSurfaceGet(surfaceFormat, surfaceDst, bufferOut, width, height))
      {
        result.ResultSub(CResult::TEST_FAIL);
        return result.Result();
      }

      if (!YUVCompare(surfaceFormat, bufferOut, bufferExp[frameIdx % FRAME_NUM], width, height))
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

int test_kernel(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
  CResult result;

#if defined(_WIN32)
  //D3D9
  if(kernel_functions(deviceID, context, queue, num_elements, 10, 256, 256, CL_ADAPTER_D3D9_KHR,
    SURFACE_FORMAT_NV12, SHARED_HANDLE_DISABLED) != 0)
  {
    log_error("\nTest case (D3D9, NV12, no shared handle) failed\n\n");
    result.ResultSub(CResult::TEST_FAIL);
  }

  if(kernel_functions(deviceID, context, queue, num_elements, 3, 256, 256, CL_ADAPTER_D3D9_KHR,
    SURFACE_FORMAT_YV12, SHARED_HANDLE_DISABLED) != 0)
  {
    log_error("\nTest case (D3D9, YV12, no shared handle) failed\n\n");
    result.ResultSub(CResult::TEST_FAIL);
  }

  //D3D9EX
  if(kernel_functions(deviceID, context, queue, num_elements, 5, 256, 512, CL_ADAPTER_D3D9EX_KHR,
    SURFACE_FORMAT_NV12, SHARED_HANDLE_DISABLED) != 0)
  {
    log_error("\nTest case (D3D9EX, NV12, no shared handle) failed\n\n");
    result.ResultSub(CResult::TEST_FAIL);
  }

  if(kernel_functions(deviceID, context, queue, num_elements, 7, 512, 256, CL_ADAPTER_D3D9EX_KHR,
    SURFACE_FORMAT_NV12, SHARED_HANDLE_ENABLED) != 0)
  {
    log_error("\nTest case (D3D9EX, NV12, shared handle) failed\n\n");
    result.ResultSub(CResult::TEST_FAIL);
  }

  if(kernel_functions(deviceID, context, queue, num_elements, 10, 256, 256, CL_ADAPTER_D3D9EX_KHR,
    SURFACE_FORMAT_YV12, SHARED_HANDLE_DISABLED) != 0)
  {
    log_error("\nTest case (D3D9EX, YV12, no shared handle) failed\n\n");
    result.ResultSub(CResult::TEST_FAIL);
  }

  if(kernel_functions(deviceID, context, queue, num_elements, 15, 128, 128, CL_ADAPTER_D3D9EX_KHR,
    SURFACE_FORMAT_YV12, SHARED_HANDLE_ENABLED) != 0)
  {
    log_error("\nTest case (D3D9EX, YV12, shared handle) failed\n\n");
    result.ResultSub(CResult::TEST_FAIL);
  }

  //DXVA
  if(kernel_functions(deviceID, context, queue, num_elements, 20, 128, 128, CL_ADAPTER_DXVA_KHR,
    SURFACE_FORMAT_NV12, SHARED_HANDLE_DISABLED) != 0)
  {
    log_error("\nTest case (DXVA, NV12, no shared handle) failed\n\n");
    result.ResultSub(CResult::TEST_FAIL);
  }

  if(kernel_functions(deviceID, context, queue, num_elements, 40, 64, 64, CL_ADAPTER_DXVA_KHR,
    SURFACE_FORMAT_NV12, SHARED_HANDLE_ENABLED) != 0)
  {
    log_error("\nTest case (DXVA, NV12, shared handle) failed\n\n");
    result.ResultSub(CResult::TEST_FAIL);
  }

  if(kernel_functions(deviceID, context, queue, num_elements, 5, 512, 512, CL_ADAPTER_DXVA_KHR,
    SURFACE_FORMAT_YV12, SHARED_HANDLE_DISABLED) != 0)
  {
    log_error("\nTest case (DXVA, YV12, no shared handle) failed\n\n");
    result.ResultSub(CResult::TEST_FAIL);
  }

  if(kernel_functions(deviceID, context, queue, num_elements, 2, 1024, 1024, CL_ADAPTER_DXVA_KHR,
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
