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
#if defined( _WIN32 )

#define _CRT_SECURE_NO_WARNINGS
#include <vector>
#include <algorithm>
#include "harness.h"
#include "harness/testHarness.h"
#include "harness/parseParameters.h"

int main(int argc, const char* argv[])
{
    cl_int result;
    cl_platform_id platform = NULL;
    cl_uint num_devices_tested = 0;

    argc = parseCustomParam(argc, argv);

    // get the platforms to test
    result = clGetPlatformIDs(1, &platform, NULL); NonTestRequire(result == CL_SUCCESS, "Failed to get any platforms.");

    HarnessD3D11_Initialize(platform);

    // for each adapter...
    IDXGIFactory* pFactory = NULL;
    HRESULT hr = CreateDXGIFactory(IID_IDXGIFactory, (void**)(&pFactory) );
    NonTestRequire(SUCCEEDED(hr), "Failed to create DXGI factory.");
    for (UINT adapter = 0;; ++adapter)
    {
        IDXGIAdapter* pAdapter = NULL;
        ID3D11Device* pDevice = NULL;
        ID3D11DeviceContext* pDC = NULL;
        HRESULT hr = pFactory->EnumAdapters(adapter, &pAdapter);
        if (FAILED(hr))
        {
            break;
        }

        // print data about the adapter
        DXGI_ADAPTER_DESC desc;
        hr = pAdapter->GetDesc(&desc);
        NonTestRequire(SUCCEEDED(hr), "IDXGIAdapter::GetDesc failed.");

        TestPrint("=====================================\n");
        TestPrint("Testing DXGI Adapter and D3D11 Device\n");
        TestPrint("Description=%ls, VendorID=%x, DeviceID=%x\n", desc.Description, desc.VendorId, desc.DeviceId);
        TestPrint("=====================================\n");

        // run the test on the adapter
        HarnessD3D11_CreateDevice(pAdapter, &pDevice, &pDC);

        cl_uint num_devices = 0;

        // test adapter and device enumeration
        TestAdapterEnumeration(platform, pAdapter, pDevice, &num_devices);

        // if there were any devices found in enumeration, run the tests on them
        if (num_devices)
        {
            TestAdapterDevices(platform, pAdapter, pDevice, pDC, num_devices);
        }
        num_devices_tested += num_devices;

        // destroy the D3D11 device
        if (pDevice)
        {
            HarnessD3D11_DestroyDevice();
        }

        pAdapter->Release();
    }
    pFactory->Release();

    // allow the test to be waived in automation
    // NonTestRequire(num_devices_tested, "No D3D11 compatible cl_device_ids were found.");

    HarnessD3D11_TestStats();
}

void TestAdapterEnumeration(
    cl_platform_id platform,
    IDXGIAdapter* pAdapter,
    ID3D11Device* pDevice,
    cl_uint* num_devices)
{
    cl_uint num_adapter_devices = 0;
    cl_device_id* adapter_devices = NULL;

    cl_uint num_device_devices = 0;
    cl_device_id* device_devices = NULL;

     cl_int result;

    HarnessD3D11_TestBegin("cl_device_id Enumeration");

    // get the cl_device_ids for the adapter
    {
        result = clGetDeviceIDsFromD3D11KHR(
            platform,
            CL_D3D11_DXGI_ADAPTER_KHR,
            pAdapter,
            CL_ALL_DEVICES_FOR_D3D11_KHR,
            0,
            NULL,
            &num_adapter_devices);
        TestRequire(
            (result == CL_SUCCESS || result == CL_DEVICE_NOT_FOUND),
            "clGetDeviceIDsFromD3D11KHR failed.");

        if (result == CL_DEVICE_NOT_FOUND)
        {
            TestPrint("No devices found for adapter.\n");
        }
        else
        {
            // if there were devices, query them
            adapter_devices = new cl_device_id[num_adapter_devices];
            result = clGetDeviceIDsFromD3D11KHR(
                platform,
                CL_D3D11_DXGI_ADAPTER_KHR,
                pAdapter,
                CL_ALL_DEVICES_FOR_D3D11_KHR,
                num_adapter_devices,
                adapter_devices,
                NULL);
            TestRequire(
                (result == CL_SUCCESS),
                "clGetDeviceIDsFromD3D11KHR failed.");
        }
    }

    // get the cl_device_ids for the device (if it was successfully created)
    if (pDevice)
    {
        result = clGetDeviceIDsFromD3D11KHR(
            platform,
            CL_D3D11_DEVICE_KHR,
            pDevice,
            CL_ALL_DEVICES_FOR_D3D11_KHR,
            0,
            NULL,
            &num_device_devices);
        TestRequire(
            (result == CL_SUCCESS || result == CL_DEVICE_NOT_FOUND),
            "clGetDeviceIDsFromD3D11KHR failed.");

        if (result == CL_DEVICE_NOT_FOUND)
        {
            TestPrint("No devices found for D3D device.\n");
        }
        else
        {
            // if there were devices, query them
            device_devices = new cl_device_id[num_device_devices];
            result = clGetDeviceIDsFromD3D11KHR(
                platform,
                CL_D3D11_DEVICE_KHR,
                pDevice,
                CL_ALL_DEVICES_FOR_D3D11_KHR,
                num_device_devices,
                device_devices,
                NULL);
            TestRequire(
                (result == CL_SUCCESS),
                "clGetDeviceIDsFromD3D11KHR failed.");
        }

    }

Cleanup:

    if (adapter_devices)
    {
        delete[] adapter_devices;
    }
    if (device_devices)
    {
        delete[] device_devices;
    }

    *num_devices = num_device_devices;

    HarnessD3D11_TestEnd();
}

void TestAdapterDevices(
    cl_platform_id platform,
    IDXGIAdapter* pAdapter,
    ID3D11Device* pDevice,
    ID3D11DeviceContext* pDC,
    cl_uint num_devices_expected)
{
    cl_int result;
    cl_uint num_devices = 0;
    cl_device_id* devices = NULL;

    devices = new cl_device_id[num_devices_expected];
    NonTestRequire(
        devices,
        "Memory allocation failure.");

    result = clGetDeviceIDsFromD3D11KHR(
        platform,
        CL_D3D11_DEVICE_KHR,
        pDevice,
        CL_ALL_DEVICES_FOR_D3D11_KHR,
        num_devices_expected,
        devices,
        &num_devices);
    NonTestRequire(
        (result == CL_SUCCESS),
        "clGetDeviceIDsFromD3D11KHR failed.");
    NonTestRequire(
        (num_devices == num_devices_expected),
        "clGetDeviceIDsFromD3D11KHR returned an unexpected number of devices.");

    for (cl_uint i = 0; i < num_devices; ++i)
    {
        TestDevice(devices[i], pDevice, pDC);
    }
}

void TestDevice(
    cl_device_id device,
    ID3D11Device* pDevice,
    ID3D11DeviceContext* pDC)
{
    char device_name[1024];
    cl_int result = CL_SUCCESS;
    cl_context context = NULL;
    cl_command_queue command_queue = NULL;
    ID3D11Device* clDevice = NULL;
    cl_uint prefer_shared_resources;

    result = clGetDeviceInfo(
        device,
        CL_DEVICE_NAME,
        sizeof(device_name),
        device_name,
        NULL);
    NonTestRequire(CL_SUCCESS == result, "clGetDeviceInfo with CL_DEVICE_NAME failed");
    TestPrint("--------------------\n");
    TestPrint("Testing cl_device_id\n");
    TestPrint("Name=%s\n", device_name);
    TestPrint("--------------------\n");

    if (!TestDeviceContextCreate(device, pDevice, &context, &command_queue) )
    {
        return;
    }

    // make sure that we can query the shared resource preference
    result = clGetContextInfo(
        context,
        CL_CONTEXT_D3D11_PREFER_SHARED_RESOURCES_KHR,
        sizeof(prefer_shared_resources),
        &prefer_shared_resources,
        NULL);
    NonTestRequire(CL_SUCCESS == result, "clGetContextInfo with CL_CONTEXT_D3D10_PREFER_SHARED_RESOURCES_KHR failed");

    // run buffer tests
    TestDeviceBuffer(
        context,
        command_queue,
        pDevice,
        pDC);

    // run 2D texture tests
    TestDeviceTexture2D(
        device,
        context,
        command_queue,
        pDevice,
        pDC);

    // run 3D texture tests
    TestDeviceTexture3D(
        device,
        context,
        command_queue,
        pDevice,
        pDC);

    // run misc tests
    TestDeviceMisc(
        device,
        context,
        command_queue,
        pDevice);

    clReleaseContext(context);
    clReleaseCommandQueue(command_queue);
}

bool TestDeviceContextCreate(
    cl_device_id device,
    ID3D11Device* pDevice,
    cl_context* out_context,
    cl_command_queue* out_command_queue)
{
    cl_int result = CL_SUCCESS;
    cl_context context = NULL;
    cl_command_queue command_queue = NULL;

    ID3D11Device* clDevice = NULL;

    bool succeeded = false;

    HarnessD3D11_TestBegin("Context creation");

    cl_context_properties properties[5];

    // create the context
    properties[0] = (cl_context_properties)CL_CONTEXT_D3D11_DEVICE_KHR;
    properties[1] = (cl_context_properties)pDevice;
    properties[2] = (cl_context_properties)CL_CONTEXT_INTEROP_USER_SYNC;
    properties[3] = (cl_context_properties)CL_TRUE;
    properties[4] = (cl_context_properties)0;
    context = clCreateContext(
        properties,
        1,
        &device,
        NULL,
        NULL,
        &result);
    TestRequire(
        (result == CL_SUCCESS),
        "clCreateContext with CL_CONTEXT_D3D11_DEVICE_KHR failed");
    result = clReleaseContext(context);
    TestRequire(
        (result == CL_SUCCESS),
        "clReleaseContext with CL_CONTEXT_D3D11_DEVICE_KHR failed");

    // create the context
    properties[0] = (cl_context_properties)CL_CONTEXT_D3D11_DEVICE_KHR;
    properties[1] = (cl_context_properties)pDevice;
    properties[2] = (cl_context_properties)CL_CONTEXT_INTEROP_USER_SYNC;
    properties[3] = (cl_context_properties)CL_FALSE;
    properties[4] = (cl_context_properties)0;
    context = clCreateContext(
        properties,
        1,
        &device,
        NULL,
        NULL,
        &result);
    TestRequire(
        (result == CL_SUCCESS),
        "clCreateContext with CL_CONTEXT_D3D11_DEVICE_KHR failed");
    result = clReleaseContext(context);
    TestRequire(
        (result == CL_SUCCESS),
        "clReleaseContext with CL_CONTEXT_D3D11_DEVICE_KHR failed");

    // create the context
    properties[0] = (cl_context_properties)CL_CONTEXT_D3D11_DEVICE_KHR;
    properties[1] = (cl_context_properties)pDevice;
    properties[2] = (cl_context_properties)0;
    context = clCreateContext(
        properties,
        1,
        &device,
        NULL,
        NULL,
        &result);
    TestRequire(
        (result == CL_SUCCESS),
        "clCreateContext with CL_CONTEXT_D3D11_DEVICE_KHR failed");

    // check CL_CONTEXT_D3D11_DEVICE_KHR
    {
        size_t param_value_size_ret;
        result = clGetContextInfo(context, CL_CONTEXT_PROPERTIES, 0, NULL, &param_value_size_ret);
        TestRequire(
            (result == CL_SUCCESS),
            "clGetContextInfo with CL_CONTEXT_PROPERTIES failed");

        TestRequire(
            ((param_value_size_ret % sizeof(cl_context_properties)) == 0),
            "param_value_size_ret is not a multiple of sizeof(cl_context_properties)");

        std::vector<cl_context_properties> contextProperties(param_value_size_ret / sizeof(cl_context_properties));
        result = clGetContextInfo(context, CL_CONTEXT_PROPERTIES, param_value_size_ret, &contextProperties[0], NULL);
        TestRequire(
            (result == CL_SUCCESS),
            "clGetContextInfo with CL_CONTEXT_PROPERTIES failed");

        TestRequire(contextProperties.size() % 2 == 1, "Property list size is not odd.");
        TestRequire(contextProperties[contextProperties.size() - 1] == 0, "last property is not zero");

        std::vector<cl_context_properties>::const_iterator iter;
        for (iter = contextProperties.begin(); *iter != 0; iter+=2)
        {
            if (CL_CONTEXT_D3D11_DEVICE_KHR == *iter)
            {
                TestRequire((ID3D11Device*)*(iter+1) == pDevice, "CL_CONTEXT_D3D11_DEVICE_KHR returned invalid value");
                break;
            }
        }

        TestRequire((iter != contextProperties.end()), "CL_CONTEXT_PROPERTIES doesn't include CL_CONTEXT_D3D11_DEVICE_KHR");
    }


    // create the command queue
    TestPrint("Creating a command queue.\n");
    command_queue = clCreateCommandQueueWithProperties(
        context,
        device,
        NULL,
        &result);
    TestRequire(
        (result == CL_SUCCESS),
        "clCreateContext with CL_CONTEXT_D3D11_DEVICE_KHR failed");

    succeeded = true;

Cleanup:

    if (succeeded)
    {
        *out_context = context;
        *out_command_queue = command_queue;
    }
    else
    {
        if (context)
        {
            clReleaseContext(context);
        }
        if (command_queue)
        {
            clReleaseCommandQueue(command_queue);
        }
    }
    HarnessD3D11_TestEnd();
    return succeeded;
}

#else

#include "errorHelpers.h"

int main(int argc, char* argv[])
{
    log_info( "Windows-specific test skipped.\n" );
    return 0;
}

#endif
