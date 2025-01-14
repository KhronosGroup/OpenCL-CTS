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

int get_device_ids(cl_device_id deviceID, cl_context context,
                   cl_command_queue queue, int num_elements,
                   cl_dx9_media_adapter_type_khr adapterType)
{
    CResult result;

    std::unique_ptr<CDeviceWrapper> deviceWrapper;
    if (!DeviceCreate(adapterType, deviceWrapper))
    {
        result.ResultSub(CResult::TEST_ERROR);
        return result.Result();
    }

    cl_uint devicesExpectedNum = 0;
    cl_int error = clGetDeviceIDs(gPlatformIDdetected, CL_DEVICE_TYPE_ALL, 0, 0,
                                  &devicesExpectedNum);
    if (error != CL_SUCCESS || devicesExpectedNum < 1)
    {
        log_error("clGetDeviceIDs failed: %s\n", IGetErrorString(error));
        result.ResultSub(CResult::TEST_FAIL);
        return result.Result();
    }

    std::vector<cl_device_id> devicesExpected(devicesExpectedNum);
    error = clGetDeviceIDs(gPlatformIDdetected, CL_DEVICE_TYPE_ALL,
                           devicesExpectedNum, &devicesExpected[0], 0);
    if (error != CL_SUCCESS)
    {
        log_error("clGetDeviceIDs failed: %s\n", IGetErrorString(error));
        result.ResultSub(CResult::TEST_FAIL);
        return result.Result();
    }

    while (deviceWrapper->AdapterNext())
    {
        std::vector<cl_dx9_media_adapter_type_khr> mediaAdapterTypes;
        mediaAdapterTypes.push_back(adapterType);

        std::vector<void *> mediaDevices;
        mediaDevices.push_back(deviceWrapper->Device());

        // check if the test can be run on the adapter
        if (CL_SUCCESS
            != (error = deviceExistForCLTest(gPlatformIDdetected, adapterType,
                                             deviceWrapper->Device(), result)))
        {
            return result.Result();
        }

        cl_uint devicesAllNum = 0;
        error = clGetDeviceIDsFromDX9MediaAdapterKHR(
            gPlatformIDdetected, 1, &mediaAdapterTypes[0], &mediaDevices[0],
            CL_ALL_DEVICES_FOR_DX9_MEDIA_ADAPTER_KHR, 0, 0, &devicesAllNum);
        if (error != CL_SUCCESS && error != CL_DEVICE_NOT_FOUND)
        {
            log_error("clGetDeviceIDsFromDX9MediaAdapterKHR failed: %s\n",
                      IGetErrorString(error));
            result.ResultSub(CResult::TEST_FAIL);
            return result.Result();
        }

        std::vector<cl_device_id> devicesAll;
        if (devicesAllNum > 0)
        {
            devicesAll.resize(devicesAllNum);
            error = clGetDeviceIDsFromDX9MediaAdapterKHR(
                gPlatformIDdetected, 1, &mediaAdapterTypes[0], &mediaDevices[0],
                CL_ALL_DEVICES_FOR_DX9_MEDIA_ADAPTER_KHR, devicesAllNum,
                &devicesAll[0], 0);
            if (error != CL_SUCCESS)
            {
                log_error("clGetDeviceIDsFromDX9MediaAdapterKHR failed: %s\n",
                          IGetErrorString(error));
                result.ResultSub(CResult::TEST_FAIL);
                return result.Result();
            }
        }

        cl_uint devicesPreferredNum = 0;
        error = clGetDeviceIDsFromDX9MediaAdapterKHR(
            gPlatformIDdetected, 1, &mediaAdapterTypes[0], &mediaDevices[0],
            CL_PREFERRED_DEVICES_FOR_DX9_MEDIA_ADAPTER_KHR, 0, 0,
            &devicesPreferredNum);
        if (error != CL_SUCCESS && error != CL_DEVICE_NOT_FOUND)
        {
            log_error("clGetDeviceIDsFromDX9MediaAdapterKHR failed: %s\n",
                      IGetErrorString(error));
            result.ResultSub(CResult::TEST_FAIL);
            return result.Result();
        }

        std::vector<cl_device_id> devicesPreferred;
        if (devicesPreferredNum > 0)
        {
            devicesPreferred.resize(devicesPreferredNum);
            error = clGetDeviceIDsFromDX9MediaAdapterKHR(
                gPlatformIDdetected, 1, &mediaAdapterTypes[0], &mediaDevices[0],
                CL_PREFERRED_DEVICES_FOR_DX9_MEDIA_ADAPTER_KHR,
                devicesPreferredNum, &devicesPreferred[0], 0);
            if (error != CL_SUCCESS)
            {
                log_error("clGetDeviceIDsFromDX9MediaAdapterKHR failed: %s\n",
                          IGetErrorString(error));
                result.ResultSub(CResult::TEST_FAIL);
                return result.Result();
            }
        }

        if (devicesAllNum < devicesPreferredNum)
        {
            log_error("Invalid number of preferred devices. It should be a "
                      "subset of all devices\n");
            result.ResultSub(CResult::TEST_FAIL);
        }

        for (cl_uint i = 0; i < devicesPreferredNum; ++i)
        {
            cl_uint j = 0;
            for (; j < devicesAllNum; ++j)
            {
                if (devicesPreferred[i] == devicesAll[j]) break;
            }

            if (j == devicesAllNum)
            {
                log_error("Preferred device is not a subset of all devices\n");
                result.ResultSub(CResult::TEST_FAIL);
            }
        }

        for (cl_uint i = 0; i < devicesAllNum; ++i)
        {
            cl_uint j = 0;
            for (; j < devicesExpectedNum; ++j)
            {
                if (devicesAll[i] == devicesExpected[j]) break;
            }

            if (j == devicesExpectedNum)
            {
                log_error("CL_ALL_DEVICES_FOR_MEDIA_ADAPTER_KHR should be a "
                          "subset of all devices for selected platform\n");
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

int test_get_device_ids(cl_device_id deviceID, cl_context context,
                        cl_command_queue queue, int num_elements)
{
    CResult result;

#if defined(_WIN32)
    if (get_device_ids(deviceID, context, queue, num_elements,
                       CL_ADAPTER_D3D9_KHR)
        != 0)
    {
        log_error("\nTest case (D3D9) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (get_device_ids(deviceID, context, queue, num_elements,
                       CL_ADAPTER_D3D9EX_KHR)
        != 0)
    {
        log_error("\nTest case (D3D9EX) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

    if (get_device_ids(deviceID, context, queue, num_elements,
                       CL_ADAPTER_DXVA_KHR)
        != 0)
    {
        log_error("\nTest case (DXVA) failed\n\n");
        result.ResultSub(CResult::TEST_FAIL);
    }

#else
    return TEST_NOT_IMPLEMENTED;
#endif

    return result.Result();
}
