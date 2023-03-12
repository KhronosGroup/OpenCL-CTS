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
#include "harness/imageHelpers.h"
#include "harness/kernelHelpers.h"

#include "utils.h"

template <typename T>
int other_data_types(cl_device_id deviceID, cl_context context,
                     cl_command_queue queue, int num_elements,
                     unsigned int iterationNum, unsigned int width,
                     unsigned int height,
                     cl_dx9_media_adapter_type_khr adapterType,
                     TSurfaceFormat surfaceFormat,
                     TSharedHandleType sharedHandle)
{
    const unsigned int FRAME_NUM = 2;
    const float MAX_VALUE = 0.6f;
    const std::string PROGRAM_STR =
        "__kernel void TestFunction( read_only image2d_t imageIn, write_only "
        "image2d_t imageOut, " NL "                            sampler_t "
        "sampler, __global int *imageRes)" NL "{" NL
        "  int w = get_global_id(0);" NL "  int h = get_global_id(1);" NL
        "  int width = get_image_width(imageIn);" NL
        "  int height = get_image_height(imageOut);" NL
        "  float4 color0 = read_imagef(imageIn, sampler, (int2)(w,h)) - "
        "0.2f;" NL "  float4 color1 = read_imagef(imageIn, sampler, "
        "(float2)(w,h)) - 0.2f;" NL
        "  color0 = (color0 == color1) ? color0: (float4)(0.5, 0.5, 0.5, "
        "0.5);" NL "  write_imagef(imageOut, (int2)(w,h), color0);" NL
        "  if(w == 0 && h == 0)" NL "  {" NL "    imageRes[0] = width;" NL
        "    imageRes[1] = height;" NL "  }" NL "}";

    CResult result;

    cl_image_format format;
    if (!SurfaceFormatToOCL(surfaceFormat, format))
    {
        result.ResultSub(CResult::TEST_ERROR);
        return result.Result();
    }

    std::auto_ptr<CDeviceWrapper> deviceWrapper;
    if (!DeviceCreate(adapterType, deviceWrapper))
    {
        result.ResultSub(CResult::TEST_ERROR);
        return result.Result();
    }

    while (deviceWrapper->AdapterNext())
    {
        cl_int error;
        // check if the test can be run on the adapter
        if (CL_SUCCESS
            != (error = deviceExistForCLTest(gPlatformIDdetected, adapterType,
                                             deviceWrapper->Device(), result,
                                             sharedHandle)))
        {
            return result.Result();
        }

        cl_context_properties contextProperties[] = {
            CL_CONTEXT_PLATFORM,
            (cl_context_properties)gPlatformIDdetected,
            AdapterTypeToContextInfo(adapterType),
            (cl_context_properties)deviceWrapper->Device(),
            0,
        };

        clContextWrapper ctx = clCreateContext(
            &contextProperties[0], 1, &gDeviceIDdetected, NULL, NULL, &error);
        if (error != CL_SUCCESS)
        {
            log_error("clCreateContext failed: %s\n", IGetErrorString(error));
            result.ResultSub(CResult::TEST_FAIL);
            return result.Result();
        }

        clCommandQueueWrapper cmdQueue = clCreateCommandQueueWithProperties(
            ctx, gDeviceIDdetected, 0, &error);
        if (error != CL_SUCCESS)
        {
            log_error("Unable to create command queue: %s\n",
                      IGetErrorString(error));
            result.ResultSub(CResult::TEST_FAIL);
            return result.Result();
        }

        if (!SurfaceFormatCheck(adapterType, *deviceWrapper, surfaceFormat))
        {
            std::string sharedHandleStr =
                (sharedHandle == SHARED_HANDLE_ENABLED) ? "yes" : "no";
            std::string formatStr;
            std::string adapterStr;
            SurfaceFormatToString(surfaceFormat, formatStr);
            AdapterToString(adapterType, adapterStr);
            log_info(
                "Skipping test case, image format is not supported by a device "
                "(adapter type: %s, format: %s, shared handle: %s)\n",
                adapterStr.c_str(), formatStr.c_str(), sharedHandleStr.c_str());
            return result.Result();
        }

        if (!ImageFormatCheck(ctx, CL_MEM_OBJECT_IMAGE2D, format))
        {
            std::string sharedHandleStr =
                (sharedHandle == SHARED_HANDLE_ENABLED) ? "yes" : "no";
            std::string formatStr;
            std::string adapterStr;
            SurfaceFormatToString(surfaceFormat, formatStr);
            AdapterToString(adapterType, adapterStr);
            log_info("Skipping test case, image format is not supported by OCL "
                     "(adapter type: %s, format: %s, shared handle: %s)\n",
                     adapterStr.c_str(), formatStr.c_str(),
                     sharedHandleStr.c_str());
            return result.Result();
        }

        if (format.image_channel_data_type == CL_HALF_FLOAT)
        {
            if (DetectFloatToHalfRoundingMode(cmdQueue))
            {
                log_error("Unable to detect rounding mode\n");
                result.ResultSub(CResult::TEST_FAIL);
                return result.Result();
            }
        }

        std::vector<std::vector<T>> bufferIn(FRAME_NUM);
        std::vector<std::vector<T>> bufferExp(FRAME_NUM);
        float step = MAX_VALUE / static_cast<float>(FRAME_NUM);
        unsigned int planeNum = ChannelNum(surfaceFormat);
        for (size_t i = 0; i < FRAME_NUM; ++i)
        {
            DataGenerate(surfaceFormat, format.image_channel_data_type,
                         bufferIn[i], width, height, planeNum, step * i,
                         step * (i + 1));
            DataGenerate(surfaceFormat, format.image_channel_data_type,
                         bufferExp[i], width, height, planeNum, step * i,
                         step * (i + 1), 0.2f);
        }

        void *objectSrcHandle = 0;
        std::auto_ptr<CSurfaceWrapper> surfaceSrc;
        if (!MediaSurfaceCreate(adapterType, width, height, surfaceFormat,
                                *deviceWrapper, surfaceSrc,
                                (sharedHandle == SHARED_HANDLE_ENABLED) ? true
                                                                        : false,
                                &objectSrcHandle))
        {
            log_error("Media surface creation failed for %i adapter\n",
                      deviceWrapper->AdapterIdx());
            result.ResultSub(CResult::TEST_ERROR);
            return result.Result();
        }

        void *objectDstHandle = 0;
        std::auto_ptr<CSurfaceWrapper> surfaceDst;
        if (!MediaSurfaceCreate(adapterType, width, height, surfaceFormat,
                                *deviceWrapper, surfaceDst,
                                (sharedHandle == SHARED_HANDLE_ENABLED) ? true
                                                                        : false,
                                &objectDstHandle))
        {
            log_error("Media surface creation failed for %i adapter\n",
                      deviceWrapper->AdapterIdx());
            result.ResultSub(CResult::TEST_ERROR);
            return result.Result();
        }

#if defined(_WIN32)
        cl_dx9_surface_info_khr surfaceSrcInfo;
        CD3D9SurfaceWrapper *dx9SurfaceSrc =
            (static_cast<CD3D9SurfaceWrapper *>(surfaceSrc.get()));
        surfaceSrcInfo.resource = *dx9SurfaceSrc;
        surfaceSrcInfo.shared_handle = objectSrcHandle;

        cl_dx9_surface_info_khr surfaceDstInfo;
        CD3D9SurfaceWrapper *dx9SurfaceDst =
            (static_cast<CD3D9SurfaceWrapper *>(surfaceDst.get()));
        surfaceDstInfo.resource = *dx9SurfaceDst;
        surfaceDstInfo.shared_handle = objectDstHandle;
#else
        void *surfaceSrcInfo = 0;
        void *surfaceDstInfo = 0;
        return TEST_NOT_IMPLEMENTED;
#endif

        // create OCL shared object
        clMemWrapper objectSrcShared = clCreateFromDX9MediaSurfaceKHR(
            ctx, CL_MEM_READ_WRITE, adapterType, &surfaceSrcInfo, 0, &error);
        if (error != CL_SUCCESS)
        {
            log_error("clCreateFromDX9MediaSurfaceKHR failed: %s\n",
                      IGetErrorString(error));
            result.ResultSub(CResult::TEST_FAIL);
            return result.Result();
        }

        clMemWrapper objectDstShared = clCreateFromDX9MediaSurfaceKHR(
            ctx, CL_MEM_READ_WRITE, adapterType, &surfaceDstInfo, 0, &error);
        if (error != CL_SUCCESS)
        {
            log_error("clCreateFromDX9MediaSurfaceKHR failed: %s\n",
                      IGetErrorString(error));
            result.ResultSub(CResult::TEST_FAIL);
            return result.Result();
        }

        std::vector<cl_mem> memObjList;
        memObjList.push_back(objectSrcShared);
        memObjList.push_back(objectDstShared);

        if (!GetMemObjInfo(objectSrcShared, adapterType, surfaceSrc,
                           objectSrcHandle))
        {
            log_error("Invalid memory object info\n");
            result.ResultSub(CResult::TEST_FAIL);
        }

        if (!GetImageInfo(objectSrcShared, format, sizeof(T) * planeNum,
                          width * sizeof(T) * planeNum, 0, width, height, 0, 0))
        {
            log_error("clGetImageInfo failed\n");
            result.ResultSub(CResult::TEST_FAIL);
        }

        for (size_t frameIdx = 0; frameIdx < iterationNum; ++frameIdx)
        {
            // surface set
#if defined(_WIN32)
            D3DLOCKED_RECT rect;
            if (FAILED((*dx9SurfaceSrc)->LockRect(&rect, NULL, 0)))
            {
                log_error("Surface lock failed\n");
                result.ResultSub(CResult::TEST_ERROR);
                return result.Result();
            }

            size_t pitch = rect.Pitch / sizeof(T);
            size_t lineSize = width * planeNum * sizeof(T);
            T *ptr = static_cast<T *>(rect.pBits);

            for (size_t y = 0; y < height; ++y)
                memcpy(ptr + y * pitch,
                       &bufferIn[frameIdx % FRAME_NUM][y * width * planeNum],
                       lineSize);

            (*dx9SurfaceSrc)->UnlockRect();
#else
            void *surfaceInfo = 0;
            return TEST_NOT_IMPLEMENTED;
#endif

            error = clEnqueueAcquireDX9MediaSurfacesKHR(
                cmdQueue, static_cast<cl_uint>(memObjList.size()),
                &memObjList[0], 0, NULL, NULL);
            if (error != CL_SUCCESS)
            {
                log_error("clEnqueueAcquireMediaSurfaceKHR failed: %s\n",
                          IGetErrorString(error));
                result.ResultSub(CResult::TEST_FAIL);
                return result.Result();
            }

            size_t origin[3] = { 0, 0, 0 };
            size_t region[3] = { width, height, 1 };

            { // read operation
                std::vector<T> out(planeNum * width * height, 0);
                error =
                    clEnqueueReadImage(cmdQueue, objectSrcShared, CL_TRUE,
                                       origin, region, 0, 0, &out[0], 0, 0, 0);
                if (error != CL_SUCCESS)
                {
                    log_error("clEnqueueReadImage failed: %s\n",
                              IGetErrorString(error));
                    result.ResultSub(CResult::TEST_FAIL);
                }

                if (!DataCompare(surfaceFormat, format.image_channel_data_type,
                                 out, bufferIn[frameIdx % FRAME_NUM], width,
                                 height, planeNum))
                {
                    log_error("Frame idx: %i, OCL object is different then "
                              "expected\n",
                              frameIdx);
                    result.ResultSub(CResult::TEST_FAIL);
                }
            }

            { // write operation
                error = clEnqueueWriteImage(
                    cmdQueue, objectSrcShared, CL_TRUE, origin, region, 0, 0,
                    &bufferExp[frameIdx % FRAME_NUM][0], 0, 0, 0);
                if (error != CL_SUCCESS)
                {
                    log_error("clEnqueueWriteImage failed: %s\n",
                              IGetErrorString(error));
                    result.ResultSub(CResult::TEST_FAIL);
                }
            }

            { // kernel operations
                clSamplerWrapper sampler = clCreateSampler(
                    ctx, CL_FALSE, CL_ADDRESS_NONE, CL_FILTER_NEAREST, &error);
                if (error != CL_SUCCESS)
                {
                    log_error("Unable to create sampler\n");
                    result.ResultSub(CResult::TEST_FAIL);
                }

                size_t threads[2] = { width, height };
                clProgramWrapper program;
                clKernelWrapper kernel;
                const char *progPtr = PROGRAM_STR.c_str();
                if (create_single_kernel_helper(ctx, &program, &kernel, 1,
                                                (const char **)&progPtr,
                                                "TestFunction"))
                    result.ResultSub(CResult::TEST_FAIL);

                error = clSetKernelArg(kernel, 0, sizeof(objectSrcShared),
                                       &(objectSrcShared));
                if (error != CL_SUCCESS)
                {
                    log_error("Unable to set kernel arguments");
                    result.ResultSub(CResult::TEST_FAIL);
                }

                error = clSetKernelArg(kernel, 1, sizeof(objectDstShared),
                                       &(objectDstShared));
                if (error != CL_SUCCESS)
                {
                    log_error("Unable to set kernel arguments");
                    result.ResultSub(CResult::TEST_FAIL);
                }

                error = clSetKernelArg(kernel, 2, sizeof(sampler), &sampler);
                if (error != CL_SUCCESS)
                {
                    log_error("Unable to set kernel arguments");
                    result.ResultSub(CResult::TEST_FAIL);
                }

                size_t bufferSize = sizeof(cl_int) * 2;
                clMemWrapper imageRes = clCreateBuffer(
                    ctx, CL_MEM_READ_WRITE, bufferSize, NULL, &error);
                if (error != CL_SUCCESS)
                {
                    log_error("clCreateBuffer failed: %s\n",
                              IGetErrorString(error));
                    result.ResultSub(CResult::TEST_FAIL);
                }

                error = clSetKernelArg(kernel, 3, sizeof(imageRes), &imageRes);

                size_t localThreads[2];
                error = get_max_common_2D_work_group_size(ctx, kernel, threads,
                                                          localThreads);
                if (error != CL_SUCCESS)
                {
                    log_error("Unable to get work group size to use");
                    result.ResultSub(CResult::TEST_FAIL);
                }

                error =
                    clEnqueueNDRangeKernel(cmdQueue, kernel, 2, NULL, threads,
                                           localThreads, 0, NULL, NULL);
                if (error != CL_SUCCESS)
                {
                    log_error("Unable to execute test kernel");
                    result.ResultSub(CResult::TEST_FAIL);
                }

                std::vector<cl_uint> imageResOut(2, 0);
                error = clEnqueueReadBuffer(cmdQueue, imageRes, CL_TRUE, 0,
                                            bufferSize, &imageResOut[0], 0,
                                            NULL, NULL);
                if (error != CL_SUCCESS)
                {
                    log_error("Unable to read buffer");
                    result.ResultSub(CResult::TEST_FAIL);
                }

                if (imageResOut[0] != width)
                {
                    log_error("Invalid width value, test = %i, expected = %i\n",
                              imageResOut[0], width);
                    result.ResultSub(CResult::TEST_FAIL);
                }

                if (imageResOut[1] != height)
                {
                    log_error(
                        "Invalid height value, test = %i, expected = %i\n",
                        imageResOut[1], height);
                    result.ResultSub(CResult::TEST_FAIL);
                }
            }

            { // map operation
                size_t mapOrigin[3] = { 0, 0, 0 };
                size_t mapRegion[3] = { width, height, 1 };

                std::vector<T> out(width * height * planeNum, 0);
                size_t rowPitch = 0;
                size_t slicePitch = 0;
                void *mapPtr = clEnqueueMapImage(
                    cmdQueue, objectDstShared, CL_TRUE,
                    CL_MAP_READ | CL_MAP_WRITE, mapOrigin, mapRegion, &rowPitch,
                    &slicePitch, 0, 0, 0, &error);
                if (error != CL_SUCCESS)
                {
                    log_error("clEnqueueMapImage failed: %s\n",
                              IGetErrorString(error));
                    result.ResultSub(CResult::TEST_FAIL);
                }

                for (size_t y = 0; y < height; ++y)
                    memcpy(&out[y * width * planeNum],
                           static_cast<T *>(mapPtr) + y * rowPitch / sizeof(T),
                           width * planeNum * sizeof(T));

                if (!DataCompare(surfaceFormat, format.image_channel_data_type,
                                 out, bufferIn[frameIdx % FRAME_NUM], width,
                                 height, planeNum))
                {
                    log_error("Frame idx: %i, Mapped OCL object is different "
                              "then expected\n",
                              frameIdx);
                    result.ResultSub(CResult::TEST_FAIL);
                }

                for (size_t y = 0; y < height; ++y)
                    memcpy(
                        static_cast<T *>(mapPtr) + y * rowPitch / sizeof(T),
                        &bufferExp[frameIdx % FRAME_NUM][y * width * planeNum],
                        width * planeNum * sizeof(T));

                error = clEnqueueUnmapMemObject(cmdQueue, objectDstShared,
                                                mapPtr, 0, 0, 0);
                if (error != CL_SUCCESS)
                {
                    log_error("clEnqueueUnmapMemObject failed: %s\n",
                              IGetErrorString(error));
                    result.ResultSub(CResult::TEST_FAIL);
                }
            }

            error = clEnqueueReleaseDX9MediaSurfacesKHR(
                cmdQueue, static_cast<cl_uint>(memObjList.size()),
                &memObjList[0], 0, NULL, NULL);
            if (error != CL_SUCCESS)
            {
                log_error("clEnqueueReleaseMediaSurfaceKHR failed: %s\n",
                          IGetErrorString(error));
                result.ResultSub(CResult::TEST_FAIL);
            }

            std::vector<T> out(width * height * planeNum, 0);
            // surface get
#if defined(_WIN32)
            if (FAILED((*dx9SurfaceDst)->LockRect(&rect, NULL, 0)))
            {
                log_error("Surface lock failed\n");
                result.ResultSub(CResult::TEST_ERROR);
                return result.Result();
            }

            pitch = rect.Pitch / sizeof(T);
            lineSize = width * planeNum * sizeof(T);
            ptr = static_cast<T *>(rect.pBits);
            for (size_t y = 0; y < height; ++y)
                memcpy(&out[y * width * planeNum], ptr + y * pitch, lineSize);

            (*dx9SurfaceDst)->UnlockRect();
#else
            return TEST_NOT_IMPLEMENTED;
#endif

            if (!DataCompare(surfaceFormat, format.image_channel_data_type, out,
                             bufferExp[frameIdx % FRAME_NUM], width, height,
                             planeNum))
            {
                log_error(
                    "Frame idx: %i, media object is different then expected\n",
                    frameIdx);
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
            log_error("%s init incomplete due to unsupported device\n",
                      adapterName.c_str());
            result.ResultSub(CResult::TEST_NOTSUPPORTED);
        }
    }

    return result.Result();
}

int test_other_data_types(cl_device_id deviceID, cl_context context,
                          cl_command_queue queue, int num_elements)
{
    CResult result;

#if defined(_WIN32)
    // D3D9
    if (other_data_types<cl_float>(deviceID, context, queue, num_elements, 10,
                                   64, 256, CL_ADAPTER_D3D9_KHR,
                                   SURFACE_FORMAT_R32F, SHARED_HANDLE_DISABLED)
        != 0)
    {
        log_error("\nTest case (D3D9, R32F, no shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_half>(deviceID, context, queue, num_elements, 10,
                                  256, 128, CL_ADAPTER_D3D9_KHR,
                                  SURFACE_FORMAT_R16F, SHARED_HANDLE_DISABLED)
        != 0)
    {
        log_error("\nTest case (D3D9, R16F, no shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_ushort>(deviceID, context, queue, num_elements, 10,
                                    512, 256, CL_ADAPTER_D3D9_KHR,
                                    SURFACE_FORMAT_L16, SHARED_HANDLE_DISABLED)
        != 0)
    {
        log_error("\nTest case (D3D9, L16, no shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_uchar>(deviceID, context, queue, num_elements, 10,
                                   256, 512, CL_ADAPTER_D3D9_KHR,
                                   SURFACE_FORMAT_A8, SHARED_HANDLE_DISABLED)
        != 0)
    {
        log_error("\nTest case (D3D9, A8, no shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_uchar>(deviceID, context, queue, num_elements, 10,
                                   1024, 32, CL_ADAPTER_D3D9_KHR,
                                   SURFACE_FORMAT_L8, SHARED_HANDLE_DISABLED)
        != 0)
    {
        log_error("\nTest case (D3D9, L8, no shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_float>(
            deviceID, context, queue, num_elements, 10, 32, 1024,
            CL_ADAPTER_D3D9_KHR, SURFACE_FORMAT_G32R32F, SHARED_HANDLE_DISABLED)
        != 0)
    {
        log_error("\nTest case (D3D9, G32R32F, no shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_half>(
            deviceID, context, queue, num_elements, 10, 64, 64,
            CL_ADAPTER_D3D9_KHR, SURFACE_FORMAT_G16R16F, SHARED_HANDLE_DISABLED)
        != 0)
    {
        log_error("\nTest case (D3D9, G16R16F, no shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_ushort>(
            deviceID, context, queue, num_elements, 10, 256, 256,
            CL_ADAPTER_D3D9_KHR, SURFACE_FORMAT_G16R16, SHARED_HANDLE_DISABLED)
        != 0)
    {
        log_error("\nTest case (D3D9, G16R16, no shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_uchar>(deviceID, context, queue, num_elements, 10,
                                   512, 128, CL_ADAPTER_D3D9_KHR,
                                   SURFACE_FORMAT_A8L8, SHARED_HANDLE_DISABLED)
        != 0)
    {
        log_error("\nTest case (D3D9, A8L8, no shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_float>(deviceID, context, queue, num_elements, 10,
                                   128, 512, CL_ADAPTER_D3D9_KHR,
                                   SURFACE_FORMAT_A32B32G32R32F,
                                   SHARED_HANDLE_DISABLED)
        != 0)
    {
        log_error(
            "\nTest case (D3D9, A32B32G32R32F, no shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_half>(deviceID, context, queue, num_elements, 10,
                                  128, 128, CL_ADAPTER_D3D9_KHR,
                                  SURFACE_FORMAT_A16B16G16R16F,
                                  SHARED_HANDLE_DISABLED)
        != 0)
    {
        log_error(
            "\nTest case (D3D9, A16B16G16R16F, no shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_ushort>(deviceID, context, queue, num_elements, 10,
                                    64, 128, CL_ADAPTER_D3D9_KHR,
                                    SURFACE_FORMAT_A16B16G16R16,
                                    SHARED_HANDLE_DISABLED)
        != 0)
    {
        log_error(
            "\nTest case (D3D9, A16B16G16R16, no shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_uchar>(deviceID, context, queue, num_elements, 10,
                                   128, 64, CL_ADAPTER_D3D9_KHR,
                                   SURFACE_FORMAT_A8B8G8R8,
                                   SHARED_HANDLE_DISABLED)
        != 0)
    {
        log_error("\nTest case (D3D9, A8B8G8R8, no shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_uchar>(deviceID, context, queue, num_elements, 10,
                                   16, 512, CL_ADAPTER_D3D9_KHR,
                                   SURFACE_FORMAT_X8B8G8R8,
                                   SHARED_HANDLE_DISABLED)
        != 0)
    {
        log_error("\nTest case (D3D9, X8B8G8R8, no shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_uchar>(deviceID, context, queue, num_elements, 10,
                                   512, 16, CL_ADAPTER_D3D9_KHR,
                                   SURFACE_FORMAT_A8R8G8B8,
                                   SHARED_HANDLE_DISABLED)
        != 0)
    {
        log_error("\nTest case (D3D9, A8R8G8B8, no shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_uchar>(deviceID, context, queue, num_elements, 10,
                                   256, 256, CL_ADAPTER_D3D9_KHR,
                                   SURFACE_FORMAT_X8R8G8B8,
                                   SHARED_HANDLE_DISABLED)
        != 0)
    {
        log_error("\nTest case (D3D9, X8R8G8B8, no shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    // D3D9EX

    if (other_data_types<cl_float>(deviceID, context, queue, num_elements, 10,
                                   64, 256, CL_ADAPTER_D3D9EX_KHR,
                                   SURFACE_FORMAT_R32F, SHARED_HANDLE_DISABLED)
        != 0)
    {
        log_error("\nTest case (D3D9EX, R32F, no shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_float>(deviceID, context, queue, num_elements, 10,
                                   64, 256, CL_ADAPTER_D3D9EX_KHR,
                                   SURFACE_FORMAT_R32F, SHARED_HANDLE_ENABLED)
        != 0)
    {
        log_error("\nTest case (D3D9EX, R32F, shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_half>(deviceID, context, queue, num_elements, 10,
                                  256, 128, CL_ADAPTER_D3D9EX_KHR,
                                  SURFACE_FORMAT_R16F, SHARED_HANDLE_DISABLED)
        != 0)
    {
        log_error("\nTest case (D3D9EX, R16F, no shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_half>(deviceID, context, queue, num_elements, 10,
                                  256, 128, CL_ADAPTER_D3D9EX_KHR,
                                  SURFACE_FORMAT_R16F, SHARED_HANDLE_ENABLED)
        != 0)
    {
        log_error("\nTest case (D3D9EX, R16F, shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_ushort>(deviceID, context, queue, num_elements, 10,
                                    512, 256, CL_ADAPTER_D3D9EX_KHR,
                                    SURFACE_FORMAT_L16, SHARED_HANDLE_DISABLED)
        != 0)
    {
        log_error("\nTest case (D3D9EX, L16, no shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_ushort>(deviceID, context, queue, num_elements, 10,
                                    512, 256, CL_ADAPTER_D3D9EX_KHR,
                                    SURFACE_FORMAT_L16, SHARED_HANDLE_ENABLED)
        != 0)
    {
        log_error("\nTest case (D3D9EX, L16, shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_uchar>(deviceID, context, queue, num_elements, 10,
                                   256, 512, CL_ADAPTER_D3D9EX_KHR,
                                   SURFACE_FORMAT_A8, SHARED_HANDLE_DISABLED)
        != 0)
    {
        log_error("\nTest case (D3D9EX, A8, no shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_uchar>(deviceID, context, queue, num_elements, 10,
                                   256, 512, CL_ADAPTER_D3D9EX_KHR,
                                   SURFACE_FORMAT_A8, SHARED_HANDLE_ENABLED)
        != 0)
    {
        log_error("\nTest case (D3D9EX, A8, shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_uchar>(deviceID, context, queue, num_elements, 10,
                                   1024, 32, CL_ADAPTER_D3D9EX_KHR,
                                   SURFACE_FORMAT_L8, SHARED_HANDLE_DISABLED)
        != 0)
    {
        log_error("\nTest case (D3D9EX, L8, no shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_uchar>(deviceID, context, queue, num_elements, 10,
                                   1024, 32, CL_ADAPTER_D3D9EX_KHR,
                                   SURFACE_FORMAT_L8, SHARED_HANDLE_ENABLED)
        != 0)
    {
        log_error("\nTest case (D3D9EX, L8, shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_float>(deviceID, context, queue, num_elements, 10,
                                   32, 1024, CL_ADAPTER_D3D9EX_KHR,
                                   SURFACE_FORMAT_G32R32F,
                                   SHARED_HANDLE_DISABLED)
        != 0)
    {
        log_error("\nTest case (D3D9EX, G32R32F, no shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_float>(deviceID, context, queue, num_elements, 10,
                                   32, 1024, CL_ADAPTER_D3D9EX_KHR,
                                   SURFACE_FORMAT_G32R32F,
                                   SHARED_HANDLE_ENABLED)
        != 0)
    {
        log_error("\nTest case (D3D9EX, G32R32F, shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_half>(deviceID, context, queue, num_elements, 10,
                                  64, 64, CL_ADAPTER_D3D9EX_KHR,
                                  SURFACE_FORMAT_G16R16F,
                                  SHARED_HANDLE_DISABLED)
        != 0)
    {
        log_error("\nTest case (D3D9EX, G16R16F, no shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_half>(deviceID, context, queue, num_elements, 10,
                                  64, 64, CL_ADAPTER_D3D9EX_KHR,
                                  SURFACE_FORMAT_G16R16F, SHARED_HANDLE_ENABLED)
        != 0)
    {
        log_error("\nTest case (D3D9EX, G16R16F, shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_ushort>(deviceID, context, queue, num_elements, 10,
                                    256, 256, CL_ADAPTER_D3D9EX_KHR,
                                    SURFACE_FORMAT_G16R16,
                                    SHARED_HANDLE_DISABLED)
        != 0)
    {
        log_error("\nTest case (D3D9EX, G16R16, no shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_ushort>(
            deviceID, context, queue, num_elements, 10, 256, 256,
            CL_ADAPTER_D3D9EX_KHR, SURFACE_FORMAT_G16R16, SHARED_HANDLE_ENABLED)
        != 0)
    {
        log_error("\nTest case (D3D9EX, G16R16, shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_uchar>(deviceID, context, queue, num_elements, 10,
                                   512, 128, CL_ADAPTER_D3D9EX_KHR,
                                   SURFACE_FORMAT_A8L8, SHARED_HANDLE_DISABLED)
        != 0)
    {
        log_error("\nTest case (D3D9EX, A8L8, no shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_uchar>(deviceID, context, queue, num_elements, 10,
                                   512, 128, CL_ADAPTER_D3D9EX_KHR,
                                   SURFACE_FORMAT_A8L8, SHARED_HANDLE_ENABLED)
        != 0)
    {
        log_error("\nTest case (D3D9EX, A8L8, shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_float>(deviceID, context, queue, num_elements, 10,
                                   128, 512, CL_ADAPTER_D3D9EX_KHR,
                                   SURFACE_FORMAT_A32B32G32R32F,
                                   SHARED_HANDLE_DISABLED)
        != 0)
    {
        log_error(
            "\nTest case (D3D9EX, A32B32G32R32F, no shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_float>(deviceID, context, queue, num_elements, 10,
                                   128, 512, CL_ADAPTER_D3D9EX_KHR,
                                   SURFACE_FORMAT_A32B32G32R32F,
                                   SHARED_HANDLE_ENABLED)
        != 0)
    {
        log_error(
            "\nTest case (D3D9EX, A32B32G32R32F, shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_half>(deviceID, context, queue, num_elements, 10,
                                  128, 128, CL_ADAPTER_D3D9EX_KHR,
                                  SURFACE_FORMAT_A16B16G16R16F,
                                  SHARED_HANDLE_DISABLED)
        != 0)
    {
        log_error(
            "\nTest case (D3D9EX, A16B16G16R16F, no shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_half>(deviceID, context, queue, num_elements, 10,
                                  128, 128, CL_ADAPTER_D3D9EX_KHR,
                                  SURFACE_FORMAT_A16B16G16R16F,
                                  SHARED_HANDLE_ENABLED)
        != 0)
    {
        log_error(
            "\nTest case (D3D9EX, A16B16G16R16F, shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_ushort>(deviceID, context, queue, num_elements, 10,
                                    64, 128, CL_ADAPTER_D3D9EX_KHR,
                                    SURFACE_FORMAT_A16B16G16R16,
                                    SHARED_HANDLE_DISABLED)
        != 0)
    {
        log_error(
            "\nTest case (D3D9EX, A16B16G16R16, no shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_ushort>(deviceID, context, queue, num_elements, 10,
                                    64, 128, CL_ADAPTER_D3D9EX_KHR,
                                    SURFACE_FORMAT_A16B16G16R16,
                                    SHARED_HANDLE_ENABLED)
        != 0)
    {
        log_error(
            "\nTest case (D3D9EX, A16B16G16R16, shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_uchar>(deviceID, context, queue, num_elements, 10,
                                   128, 64, CL_ADAPTER_D3D9EX_KHR,
                                   SURFACE_FORMAT_A8B8G8R8,
                                   SHARED_HANDLE_DISABLED)
        != 0)
    {
        log_error(
            "\nTest case (D3D9EX, A8B8G8R8, no shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_uchar>(deviceID, context, queue, num_elements, 10,
                                   128, 64, CL_ADAPTER_D3D9EX_KHR,
                                   SURFACE_FORMAT_A8B8G8R8,
                                   SHARED_HANDLE_ENABLED)
        != 0)
    {
        log_error("\nTest case (D3D9EX, A8B8G8R8, shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_uchar>(deviceID, context, queue, num_elements, 10,
                                   16, 512, CL_ADAPTER_D3D9EX_KHR,
                                   SURFACE_FORMAT_X8B8G8R8,
                                   SHARED_HANDLE_DISABLED)
        != 0)
    {
        log_error(
            "\nTest case (D3D9EX, X8B8G8R8, no shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_uchar>(deviceID, context, queue, num_elements, 10,
                                   16, 512, CL_ADAPTER_D3D9EX_KHR,
                                   SURFACE_FORMAT_X8B8G8R8,
                                   SHARED_HANDLE_ENABLED)
        != 0)
    {
        log_error("\nTest case (D3D9EX, X8B8G8R8, shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_uchar>(deviceID, context, queue, num_elements, 10,
                                   512, 16, CL_ADAPTER_D3D9EX_KHR,
                                   SURFACE_FORMAT_A8R8G8B8,
                                   SHARED_HANDLE_DISABLED)
        != 0)
    {
        log_error(
            "\nTest case (D3D9EX, A8R8G8B8, no shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_uchar>(deviceID, context, queue, num_elements, 10,
                                   512, 16, CL_ADAPTER_D3D9EX_KHR,
                                   SURFACE_FORMAT_A8R8G8B8,
                                   SHARED_HANDLE_ENABLED)
        != 0)
    {
        log_error("\nTest case (D3D9EX, A8R8G8B8, shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_uchar>(deviceID, context, queue, num_elements, 10,
                                   256, 256, CL_ADAPTER_D3D9EX_KHR,
                                   SURFACE_FORMAT_X8R8G8B8,
                                   SHARED_HANDLE_DISABLED)
        != 0)
    {
        log_error(
            "\nTest case (D3D9EX, X8R8G8B8, no shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_uchar>(deviceID, context, queue, num_elements, 10,
                                   256, 256, CL_ADAPTER_D3D9EX_KHR,
                                   SURFACE_FORMAT_X8R8G8B8,
                                   SHARED_HANDLE_ENABLED)
        != 0)
    {
        log_error("\nTest case (D3D9EX, X8R8G8B8, shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    // DXVA

    if (other_data_types<cl_float>(deviceID, context, queue, num_elements, 10,
                                   64, 256, CL_ADAPTER_DXVA_KHR,
                                   SURFACE_FORMAT_R32F, SHARED_HANDLE_DISABLED)
        != 0)
    {
        log_error("\nTest case (DXVA, R32F, no shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_float>(deviceID, context, queue, num_elements, 10,
                                   64, 256, CL_ADAPTER_DXVA_KHR,
                                   SURFACE_FORMAT_R32F, SHARED_HANDLE_ENABLED)
        != 0)
    {
        log_error("\nTest case (DXVA, R32F, shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_half>(deviceID, context, queue, num_elements, 10,
                                  256, 128, CL_ADAPTER_DXVA_KHR,
                                  SURFACE_FORMAT_R16F, SHARED_HANDLE_DISABLED)
        != 0)
    {
        log_error("\nTest case (DXVA, R16F, no shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_half>(deviceID, context, queue, num_elements, 10,
                                  256, 128, CL_ADAPTER_DXVA_KHR,
                                  SURFACE_FORMAT_R16F, SHARED_HANDLE_ENABLED)
        != 0)
    {
        log_error("\nTest case (DXVA, R16F, shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_ushort>(deviceID, context, queue, num_elements, 10,
                                    512, 256, CL_ADAPTER_DXVA_KHR,
                                    SURFACE_FORMAT_L16, SHARED_HANDLE_DISABLED)
        != 0)
    {
        log_error("\nTest case (DXVA, L16, no shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_ushort>(deviceID, context, queue, num_elements, 10,
                                    512, 256, CL_ADAPTER_DXVA_KHR,
                                    SURFACE_FORMAT_L16, SHARED_HANDLE_ENABLED)
        != 0)
    {
        log_error("\nTest case (DXVA, L16, shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_uchar>(deviceID, context, queue, num_elements, 10,
                                   256, 512, CL_ADAPTER_DXVA_KHR,
                                   SURFACE_FORMAT_A8, SHARED_HANDLE_DISABLED)
        != 0)
    {
        log_error("\nTest case (DXVA, A8, no shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_uchar>(deviceID, context, queue, num_elements, 10,
                                   256, 512, CL_ADAPTER_DXVA_KHR,
                                   SURFACE_FORMAT_A8, SHARED_HANDLE_ENABLED)
        != 0)
    {
        log_error("\nTest case (DXVA, A8, shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_uchar>(deviceID, context, queue, num_elements, 10,
                                   1024, 32, CL_ADAPTER_DXVA_KHR,
                                   SURFACE_FORMAT_L8, SHARED_HANDLE_DISABLED)
        != 0)
    {
        log_error("\nTest case (DXVA, L8, no shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_uchar>(deviceID, context, queue, num_elements, 10,
                                   1024, 32, CL_ADAPTER_DXVA_KHR,
                                   SURFACE_FORMAT_L8, SHARED_HANDLE_ENABLED)
        != 0)
    {
        log_error("\nTest case (DXVA, L8, shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_float>(
            deviceID, context, queue, num_elements, 10, 32, 1024,
            CL_ADAPTER_DXVA_KHR, SURFACE_FORMAT_G32R32F, SHARED_HANDLE_DISABLED)
        != 0)
    {
        log_error("\nTest case (DXVA, G32R32F, no shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_float>(
            deviceID, context, queue, num_elements, 10, 32, 1024,
            CL_ADAPTER_DXVA_KHR, SURFACE_FORMAT_G32R32F, SHARED_HANDLE_ENABLED)
        != 0)
    {
        log_error("\nTest case (DXVA, G32R32F, shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_half>(
            deviceID, context, queue, num_elements, 10, 64, 64,
            CL_ADAPTER_DXVA_KHR, SURFACE_FORMAT_G16R16F, SHARED_HANDLE_DISABLED)
        != 0)
    {
        log_error("\nTest case (DXVA, G16R16F, no shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_half>(deviceID, context, queue, num_elements, 10,
                                  64, 64, CL_ADAPTER_DXVA_KHR,
                                  SURFACE_FORMAT_G16R16F, SHARED_HANDLE_ENABLED)
        != 0)
    {
        log_error("\nTest case (DXVA, G16R16F, shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_ushort>(
            deviceID, context, queue, num_elements, 10, 256, 256,
            CL_ADAPTER_DXVA_KHR, SURFACE_FORMAT_G16R16, SHARED_HANDLE_DISABLED)
        != 0)
    {
        log_error("\nTest case (DXVA, G16R16, no shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_ushort>(
            deviceID, context, queue, num_elements, 10, 256, 256,
            CL_ADAPTER_DXVA_KHR, SURFACE_FORMAT_G16R16, SHARED_HANDLE_ENABLED)
        != 0)
    {
        log_error("\nTest case (DXVA, G16R16, shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_uchar>(deviceID, context, queue, num_elements, 10,
                                   512, 128, CL_ADAPTER_DXVA_KHR,
                                   SURFACE_FORMAT_A8L8, SHARED_HANDLE_DISABLED)
        != 0)
    {
        log_error("\nTest case (DXVA, A8L8, no shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_uchar>(deviceID, context, queue, num_elements, 10,
                                   512, 128, CL_ADAPTER_DXVA_KHR,
                                   SURFACE_FORMAT_A8L8, SHARED_HANDLE_ENABLED)
        != 0)
    {
        log_error("\nTest case (DXVA, A8L8, shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_float>(deviceID, context, queue, num_elements, 10,
                                   128, 512, CL_ADAPTER_DXVA_KHR,
                                   SURFACE_FORMAT_A32B32G32R32F,
                                   SHARED_HANDLE_DISABLED)
        != 0)
    {
        log_error(
            "\nTest case (DXVA, A32B32G32R32F, no shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_float>(deviceID, context, queue, num_elements, 10,
                                   128, 512, CL_ADAPTER_DXVA_KHR,
                                   SURFACE_FORMAT_A32B32G32R32F,
                                   SHARED_HANDLE_ENABLED)
        != 0)
    {
        log_error(
            "\nTest case (DXVA, A32B32G32R32F, shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_half>(deviceID, context, queue, num_elements, 10,
                                  128, 128, CL_ADAPTER_DXVA_KHR,
                                  SURFACE_FORMAT_A16B16G16R16F,
                                  SHARED_HANDLE_DISABLED)
        != 0)
    {
        log_error(
            "\nTest case (DXVA, A16B16G16R16F, no shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_half>(deviceID, context, queue, num_elements, 10,
                                  128, 128, CL_ADAPTER_DXVA_KHR,
                                  SURFACE_FORMAT_A16B16G16R16F,
                                  SHARED_HANDLE_ENABLED)
        != 0)
    {
        log_error(
            "\nTest case (DXVA, A16B16G16R16F, shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_ushort>(deviceID, context, queue, num_elements, 10,
                                    64, 128, CL_ADAPTER_DXVA_KHR,
                                    SURFACE_FORMAT_A16B16G16R16,
                                    SHARED_HANDLE_DISABLED)
        != 0)
    {
        log_error(
            "\nTest case (DXVA, A16B16G16R16, no shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_ushort>(deviceID, context, queue, num_elements, 10,
                                    64, 128, CL_ADAPTER_DXVA_KHR,
                                    SURFACE_FORMAT_A16B16G16R16,
                                    SHARED_HANDLE_ENABLED)
        != 0)
    {
        log_error("\nTest case (DXVA, A16B16G16R16, shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_uchar>(deviceID, context, queue, num_elements, 10,
                                   128, 64, CL_ADAPTER_DXVA_KHR,
                                   SURFACE_FORMAT_A8B8G8R8,
                                   SHARED_HANDLE_DISABLED)
        != 0)
    {
        log_error("\nTest case (DXVA, A8B8G8R8, no shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_uchar>(
            deviceID, context, queue, num_elements, 10, 128, 64,
            CL_ADAPTER_DXVA_KHR, SURFACE_FORMAT_A8B8G8R8, SHARED_HANDLE_ENABLED)
        != 0)
    {
        log_error("\nTest case (DXVA, A8B8G8R8, shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_uchar>(deviceID, context, queue, num_elements, 10,
                                   16, 512, CL_ADAPTER_DXVA_KHR,
                                   SURFACE_FORMAT_X8B8G8R8,
                                   SHARED_HANDLE_DISABLED)
        != 0)
    {
        log_error("\nTest case (DXVA, X8B8G8R8, no shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_uchar>(
            deviceID, context, queue, num_elements, 10, 16, 512,
            CL_ADAPTER_DXVA_KHR, SURFACE_FORMAT_X8B8G8R8, SHARED_HANDLE_ENABLED)
        != 0)
    {
        log_error("\nTest case (DXVA, X8B8G8R8, shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_uchar>(deviceID, context, queue, num_elements, 10,
                                   512, 16, CL_ADAPTER_DXVA_KHR,
                                   SURFACE_FORMAT_A8R8G8B8,
                                   SHARED_HANDLE_DISABLED)
        != 0)
    {
        log_error("\nTest case (DXVA, A8R8G8B8, no shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_uchar>(
            deviceID, context, queue, num_elements, 10, 512, 16,
            CL_ADAPTER_DXVA_KHR, SURFACE_FORMAT_A8R8G8B8, SHARED_HANDLE_ENABLED)
        != 0)
    {
        log_error("\nTest case (DXVA, A8R8G8B8, shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_uchar>(deviceID, context, queue, num_elements, 10,
                                   256, 256, CL_ADAPTER_DXVA_KHR,
                                   SURFACE_FORMAT_X8R8G8B8,
                                   SHARED_HANDLE_DISABLED)
        != 0)
    {
        log_error("\nTest case (DXVA, X8R8G8B8, no shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (other_data_types<cl_uchar>(
            deviceID, context, queue, num_elements, 10, 256, 256,
            CL_ADAPTER_DXVA_KHR, SURFACE_FORMAT_X8R8G8B8, SHARED_HANDLE_ENABLED)
        != 0)
    {
        log_error("\nTest case (DXVA, X8R8G8B8, shared handle) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

#else
    return TEST_NOT_IMPLEMENTED;
#endif

    return result.Result();
}
