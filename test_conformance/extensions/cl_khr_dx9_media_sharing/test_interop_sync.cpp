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

int interop_user_sync(cl_device_id deviceID, cl_context context,
                      cl_command_queue queue, int num_elements,
                      unsigned int width, unsigned int height,
                      TContextFuncType functionCreate,
                      cl_dx9_media_adapter_type_khr adapterType,
                      TSurfaceFormat surfaceFormat,
                      TSharedHandleType sharedHandle, cl_bool userSync)
{
    CResult result;

    // create device
    std::auto_ptr<CDeviceWrapper> deviceWrapper;
    if (!DeviceCreate(adapterType, deviceWrapper))
    {
        result.ResultSub(CResult::TEST_ERROR);
        return result.Result();
    }

    // generate input data
    std::vector<cl_uchar> bufferIn(width * height * 3 / 2, 0);
    if (!YUVGenerate(surfaceFormat, bufferIn, width, height, 0, 255))
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

        if (surfaceFormat != SURFACE_FORMAT_NV12
            && !SurfaceFormatCheck(adapterType, *deviceWrapper, surfaceFormat))
        {
            std::string sharedHandleStr =
                (sharedHandle == SHARED_HANDLE_ENABLED) ? "yes" : "no";
            std::string syncStr = (userSync == CL_TRUE) ? "yes" : "no";
            std::string formatStr;
            std::string adapterStr;
            SurfaceFormatToString(surfaceFormat, formatStr);
            AdapterToString(adapterType, adapterStr);
            log_info("Skipping test case, image format is not supported by a "
                     "device (adapter type: %s, format: %s, shared handle: %s, "
                     "user sync: %s)\n",
                     adapterStr.c_str(), formatStr.c_str(),
                     sharedHandleStr.c_str(), syncStr.c_str());
            return result.Result();
        }

        void *objectSharedHandle = 0;
        std::auto_ptr<CSurfaceWrapper> surface;
        if (!MediaSurfaceCreate(
                adapterType, width, height, surfaceFormat, *deviceWrapper,
                surface, (sharedHandle == SHARED_HANDLE_ENABLED) ? true : false,
                &objectSharedHandle))
        {
            log_error("Media surface creation failed for %i adapter\n",
                      deviceWrapper->AdapterIdx());
            result.ResultSub(CResult::TEST_ERROR);
            return result.Result();
        }

        cl_context_properties contextProperties[] = {
            CL_CONTEXT_PLATFORM,
            (cl_context_properties)gPlatformIDdetected,
            AdapterTypeToContextInfo(adapterType),
            (cl_context_properties)deviceWrapper->Device(),
            CL_CONTEXT_INTEROP_USER_SYNC,
            userSync,
            0,
        };


        clContextWrapper ctx;
        switch (functionCreate)
        {
            case CONTEXT_CREATE_DEFAULT:
                ctx = clCreateContext(&contextProperties[0], 1,
                                      &gDeviceIDdetected, NULL, NULL, &error);
                break;
            case CONTEXT_CREATE_FROM_TYPE:
                ctx = clCreateContextFromType(&contextProperties[0],
                                              gDeviceTypeSelected, NULL, NULL,
                                              &error);
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
            log_error("%s failed: %s\n", functionName.c_str(),
                      IGetErrorString(error));
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
        surfaceInfo.resource =
            *(static_cast<CD3D9SurfaceWrapper *>(surface.get()));
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
            planesList[planeIdx] = clCreateFromDX9MediaSurfaceKHR(
                ctx, CL_MEM_READ_WRITE, adapterType, &surfaceInfo, planeIdx,
                &error);
            if (error != CL_SUCCESS)
            {
                log_error(
                    "clCreateFromDX9MediaSurfaceKHR failed for plane %i: %s\n",
                    planeIdx, IGetErrorString(error));
                result.ResultSub(CResult::TEST_FAIL);
                return result.Result();
            }
            memObjList.push_back(planesList[planeIdx]);
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

        if (!ImageInfoVerify(adapterType, memObjList, width, height, surface,
                             objectSharedHandle))
        {
            log_error("Image info verification failed\n");
            result.ResultSub(CResult::TEST_FAIL);
        }

        if (userSync == CL_TRUE)
        {
#if defined(_WIN32)
            IDirect3DQuery9 *eventQuery = NULL;
            switch (adapterType)
            {
                case CL_ADAPTER_D3D9_KHR: {
                    LPDIRECT3DDEVICE9 device =
                        (LPDIRECT3DDEVICE9)deviceWrapper->Device();
                    device->CreateQuery(D3DQUERYTYPE_EVENT, &eventQuery);
                    eventQuery->Issue(D3DISSUE_END);

                    while (S_FALSE
                           == eventQuery->GetData(NULL, 0, D3DGETDATA_FLUSH))
                        ;
                }
                break;
                case CL_ADAPTER_D3D9EX_KHR: {
                    LPDIRECT3DDEVICE9EX device =
                        (LPDIRECT3DDEVICE9EX)deviceWrapper->Device();
                    device->CreateQuery(D3DQUERYTYPE_EVENT, &eventQuery);
                    eventQuery->Issue(D3DISSUE_END);

                    while (S_FALSE
                           == eventQuery->GetData(NULL, 0, D3DGETDATA_FLUSH))
                        ;
                }
                break;
                case CL_ADAPTER_DXVA_KHR: {
                    CDXVAWrapper *DXVADevice =
                        dynamic_cast<CDXVAWrapper *>(&(*deviceWrapper));
                    LPDIRECT3DDEVICE9EX device =
                        (LPDIRECT3DDEVICE9EX)(DXVADevice->D3D9()).Device();
                    device->CreateQuery(D3DQUERYTYPE_EVENT, &eventQuery);
                    eventQuery->Issue(D3DISSUE_END);

                    while (S_FALSE
                           == eventQuery->GetData(NULL, 0, D3DGETDATA_FLUSH))
                        ;
                }
                break;
                default:
                    log_error("Unknown adapter type\n");
                    return false;
                    break;
            }
            if (eventQuery)
            {
                eventQuery->Release();
            }
#else
            return TEST_NOT_IMPLEMENTED;
#endif
        }

        error = clEnqueueAcquireDX9MediaSurfacesKHR(
            cmdQueue, static_cast<cl_uint>(memObjList.size()),
            &memObjList.at(0), 0, NULL, NULL);
        if (error != CL_SUCCESS)
        {
            log_error("clEnqueueAcquireDX9MediaSurfacesKHR failed: %s\n",
                      IGetErrorString(error));
            result.ResultSub(CResult::TEST_FAIL);
            return result.Result();
        }

        size_t origin[3] = { 0, 0, 0 };
        size_t offset = 0;
        size_t frameSize = width * height * 3 / 2;
        std::vector<cl_uchar> out(frameSize, 0);
        for (size_t i = 0; i < memObjList.size(); ++i)
        {
            size_t planeWidth = (i == 0) ? width : width / 2;
            size_t planeHeight = (i == 0) ? height : height / 2;
            size_t regionPlane[3] = { planeWidth, planeHeight, 1 };

            error =
                clEnqueueReadImage(cmdQueue, memObjList.at(i), CL_TRUE, origin,
                                   regionPlane, 0, 0, &out.at(offset), 0, 0, 0);
            if (error != CL_SUCCESS)
            {
                log_error("clEnqueueReadImage failed: %s\n",
                          IGetErrorString(error));
                result.ResultSub(CResult::TEST_FAIL);
            }

            offset += planeWidth * planeHeight;
        }

        if (!YUVCompare(surfaceFormat, out, bufferIn, width, height))
        {
            log_error("OCL object verification failed - clEnqueueReadImage\n");
            result.ResultSub(CResult::TEST_FAIL);
        }

        error = clEnqueueReleaseDX9MediaSurfacesKHR(
            cmdQueue, static_cast<cl_uint>(memObjList.size()),
            &memObjList.at(0), 0, NULL, NULL);
        if (error != CL_SUCCESS)
        {
            log_error("clEnqueueReleaseDX9MediaSurfacesKHR failed: %s\n",
                      IGetErrorString(error));
            result.ResultSub(CResult::TEST_FAIL);
        }

        if (userSync == CL_TRUE)
        {
            error = clFinish(cmdQueue);
            if (error != CL_SUCCESS)
            {
                log_error("clFinish failed: %s\n", IGetErrorString(error));
                result.ResultSub(CResult::TEST_FAIL);
            }
        }

        // shared object verification
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

int test_interop_user_sync(cl_device_id deviceID, cl_context context,
                           cl_command_queue queue, int num_elements)
{
    const unsigned int WIDTH = 256;
    const unsigned int HEIGHT = 256;

    std::vector<cl_dx9_media_adapter_type_khr> adapters;
#if defined(_WIN32)
    adapters.push_back(CL_ADAPTER_D3D9_KHR);
    adapters.push_back(CL_ADAPTER_D3D9EX_KHR);
    adapters.push_back(CL_ADAPTER_DXVA_KHR);
#else
    return TEST_NOT_IMPLEMENTED;
#endif

    std::vector<TContextFuncType> contextFuncs;
    contextFuncs.push_back(CONTEXT_CREATE_DEFAULT);
    contextFuncs.push_back(CONTEXT_CREATE_FROM_TYPE);

    std::vector<TSurfaceFormat> formats;
    formats.push_back(SURFACE_FORMAT_NV12);
    formats.push_back(SURFACE_FORMAT_YV12);

    std::vector<TSharedHandleType> sharedHandleTypes;
    sharedHandleTypes.push_back(SHARED_HANDLE_DISABLED);
    sharedHandleTypes.push_back(SHARED_HANDLE_ENABLED);

    std::vector<cl_bool> sync;
    sync.push_back(CL_FALSE);
    sync.push_back(CL_TRUE);

    CResult result;
    for (size_t adapterIdx = 0; adapterIdx < adapters.size(); ++adapterIdx)
    {
        // iteration through all create context functions
        for (size_t contextFuncIdx = 0; contextFuncIdx < contextFuncs.size();
             ++contextFuncIdx)
        {
            // iteration through YUV formats
            for (size_t formatIdx = 0; formatIdx < formats.size(); ++formatIdx)
            {
                // shared handle enabled or disabled
                for (size_t sharedHandleIdx = 0;
                     sharedHandleIdx < sharedHandleTypes.size();
                     ++sharedHandleIdx)
                {
                    // user sync interop disabled or enabled
                    for (size_t syncIdx = 0; syncIdx < sync.size(); ++syncIdx)
                    {
                        if (adapters[adapterIdx] == CL_ADAPTER_D3D9_KHR
                            && sharedHandleTypes[sharedHandleIdx]
                                == SHARED_HANDLE_ENABLED)
                            continue;

                        if (interop_user_sync(
                                deviceID, context, queue, num_elements, WIDTH,
                                HEIGHT, contextFuncs[contextFuncIdx],
                                adapters[adapterIdx], formats[formatIdx],
                                sharedHandleTypes[sharedHandleIdx],
                                sync[syncIdx])
                            != 0)
                        {
                            std::string syncStr = (sync[syncIdx] == CL_TRUE)
                                ? "user sync enabled"
                                : "user sync disabled";
                            std::string sharedHandle =
                                (sharedHandleTypes[sharedHandleIdx]
                                 == SHARED_HANDLE_ENABLED)
                                ? "shared handle"
                                : "no shared handle";
                            std::string adapterStr;
                            std::string formatStr;
                            SurfaceFormatToString(formats[formatIdx],
                                                  formatStr);
                            AdapterToString(adapters[adapterIdx], adapterStr);

                            log_error("\nTest case - clCreateContext (%s, %s, "
                                      "%s, %s) failed\n\n",
                                      adapterStr.c_str(), formatStr.c_str(),
                                      sharedHandle.c_str(), syncStr.c_str());
                            result.ResultSub(CResult::TEST_FAIL);
                        }
                    }
                }
            }
        }
    }

    return result.Result();
}
