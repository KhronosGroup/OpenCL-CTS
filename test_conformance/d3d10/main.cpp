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
#include <memory>
#include <unordered_map>
#include <vector>
#include "harness.h"
#include "harness/testHarness.h"

namespace {

struct D3D10SuiteState
{
    bool initialized = false;
    cl_platform_id platform = nullptr;
    std::unique_ptr<DirectX10Wrapper> wrapper;
    std::unordered_map<cl_device_id, size_t> device_to_target;
};

static D3D10SuiteState State;

test_status InitState(cl_device_id selected_device)
{
    if (State.initialized)
    {
        return State.device_to_target.count(selected_device)
            ? TEST_PASS
            : TEST_SKIPPED_ITSELF;
    }

    cl_int result =
        clGetDeviceInfo(selected_device, CL_DEVICE_PLATFORM,
                        sizeof(State.platform), &State.platform, NULL);
    if (result != CL_SUCCESS)
    {
        log_error("clGetDeviceInfo(CL_DEVICE_PLATFORM) failed during D3D10 "
                  "suite initialization (%d)\n",
                  result);
        return TEST_FAIL;
    }

    try
    {
        HarnessD3D10_Initialize(State.platform);

        State.wrapper.reset(new DirectX10Wrapper());
        for (size_t target_index = 0;
             target_index < State.wrapper->devices.size(); ++target_index)
        {
            const auto& device_entry = State.wrapper->devices[target_index];

            cl_uint num_devices = 0;
            cl_int query_result = clGetDeviceIDsFromD3D10KHR(
                State.platform, CL_D3D10_DEVICE_KHR,
                device_entry.dx_device.Get(), CL_ALL_DEVICES_FOR_D3D10_KHR, 0,
                NULL, &num_devices);
            if (query_result != CL_DEVICE_NOT_FOUND)
            {
                if (query_result != CL_SUCCESS)
                {
                    log_error("clGetDeviceIDsFromD3D10KHR failed during D3D10 "
                              "target discovery (%d)\n",
                              query_result);
                    return TEST_FAIL;
                }

                std::vector<cl_device_id> devices(num_devices);
                query_result = clGetDeviceIDsFromD3D10KHR(
                    State.platform, CL_D3D10_DEVICE_KHR,
                    device_entry.dx_device.Get(), CL_ALL_DEVICES_FOR_D3D10_KHR,
                    num_devices, devices.data(), NULL);
                if (query_result != CL_SUCCESS)
                {
                    log_error("clGetDeviceIDsFromD3D10KHR failed while "
                              "retrieving discovered D3D10 targets (%d)\n",
                              query_result);
                    return TEST_FAIL;
                }

                for (cl_device_id device_id : devices)
                {
                    State.device_to_target[device_id] = target_index;
                }
            }
        }
    } catch (const std::exception& e)
    {
        log_error("D3D10 suite initialization failed: %s\n", e.what());
        return TEST_FAIL;
    } catch (...)
    {
        log_error(
            "D3D10 suite initialization failed with an unknown exception\n");
        return TEST_FAIL;
    }

    State.initialized = true;
    return State.device_to_target.count(selected_device) ? TEST_PASS
                                                         : TEST_SKIPPED_ITSELF;
}

test_status InitD3D10Device(cl_device_id device)
{
    if (!is_extension_available(device, "cl_khr_d3d10_sharing"))
    {
        return TEST_SKIPPED_ITSELF;
    }

    return InitState(device);
}

} // namespace

int main(int argc, const char* argv[])
{
    return runTestHarnessWithCheck(argc, argv, true, 0, InitD3D10Device);
}

template <typename TargetFn>
test_status RunD3D10TargetTest(cl_device_id device, TargetFn target_fn)
{
    test_status init_status = InitState(device);
    if (init_status != TEST_PASS)
    {
        return init_status;
    }

    auto iter = State.device_to_target.find(device);
    if (iter == State.device_to_target.end())
    {
        return TEST_SKIPPED_ITSELF;
    }

    const DirectX10Wrapper::DeviceEntry* target =
        &State.wrapper->devices[iter->second];
    target_fn(target);

    return TEST_PASS;
}

template <typename TargetFn>
test_status RunD3D10InteropTest(cl_device_id device, TargetFn target_fn)
{
    return RunD3D10TargetTest(
        device, [&](const DirectX10Wrapper::DeviceEntry* target) {
            cl_context interop_context = NULL;
            cl_command_queue interop_queue = NULL;

            if (!TestDeviceContextCreate(device, target->dx_device.Get(),
                                         &interop_context, &interop_queue))
            {
                return TEST_FAIL;
            }

            target_fn(target, interop_context, interop_queue);

            if (interop_queue != NULL)
            {
                cl_int result = clReleaseCommandQueue(interop_queue);
                if (result != CL_SUCCESS)
                {
                    return TEST_FAIL;
                }
            }

            if (interop_context != NULL)
            {
                cl_int result = clReleaseContext(interop_context);
                if (result != CL_SUCCESS)
                {
                    return TEST_FAIL;
                }
            }

            return TEST_PASS;
        });
}

REGISTER_TEST(enumeration)
{
    return RunD3D10TargetTest(
        device, [&](const DirectX10Wrapper::DeviceEntry* target) {
            cl_uint num_devices = 0;
            TestAdapterEnumeration(State.platform, target->dx_adapter.Get(),
                                   target->dx_device.Get(), &num_devices);
        });
}

REGISTER_TEST(create_context)
{
    return RunD3D10InteropTest(device,
                               [](const DirectX10Wrapper::DeviceEntry*,
                                  cl_context, cl_command_queue) {});
}

REGISTER_TEST(buffer)
{
    return RunD3D10InteropTest(
        device,
        [](const DirectX10Wrapper::DeviceEntry* target,
           cl_context interop_context, cl_command_queue interop_queue) {
            TestDeviceBuffer(interop_context, interop_queue,
                             target->dx_device.Get());
        });
}

REGISTER_TEST(texture2d)
{
    return RunD3D10InteropTest(
        device,
        [device](const DirectX10Wrapper::DeviceEntry* target,
                 cl_context interop_context, cl_command_queue interop_queue) {
            TestDeviceTexture2D(device, interop_context, interop_queue,
                                target->dx_device.Get());
        });
}

REGISTER_TEST(texture3d)
{
    return RunD3D10InteropTest(
        device,
        [device](const DirectX10Wrapper::DeviceEntry* target,
                 cl_context interop_context, cl_command_queue interop_queue) {
            TestDeviceTexture3D(device, interop_context, interop_queue,
                                target->dx_device.Get());
        });
}

REGISTER_TEST(misc)
{
    return RunD3D10InteropTest(
        device,
        [device](const DirectX10Wrapper::DeviceEntry* target,
                 cl_context interop_context, cl_command_queue interop_queue) {
            TestDeviceMisc(device, interop_context, interop_queue,
                           target->dx_device.Get());
        });
}

void TestAdapterEnumeration(cl_platform_id platform, IDXGIAdapter* pAdapter, ID3D10Device* pDevice, cl_uint* num_devices)
{
    cl_uint num_adapter_devices = 0;
    cl_device_id* adapter_devices = NULL;

    cl_uint num_device_devices = 0;
    cl_device_id* device_devices = NULL;

     cl_int result;

     log_info("cl_device_id Enumeration");

     // get the cl_device_ids for the adapter
     {
         result = clGetDeviceIDsFromD3D10KHR(
             platform, CL_D3D10_DXGI_ADAPTER_KHR, pAdapter,
             CL_ALL_DEVICES_FOR_D3D10_KHR, 0, NULL, &num_adapter_devices);
         TestRequire((result == CL_SUCCESS || result == CL_DEVICE_NOT_FOUND),
                     "clGetDeviceIDsFromD3D10KHR failed.");

         if (result == CL_DEVICE_NOT_FOUND)
         {
             TestPrint("No devices found for adapter.\n");
         }
         else
         {
             // if there were devices, query them
             adapter_devices = new cl_device_id[num_adapter_devices];
             result = clGetDeviceIDsFromD3D10KHR(
                 platform, CL_D3D10_DXGI_ADAPTER_KHR, pAdapter,
                 CL_ALL_DEVICES_FOR_D3D10_KHR, num_adapter_devices,
                 adapter_devices, NULL);
             TestRequire((result == CL_SUCCESS),
                         "clGetDeviceIDsFromD3D10KHR failed.");
         }
     }

    // get the cl_device_ids for the device (if it was successfully created)
    if (pDevice)
    {
        result = clGetDeviceIDsFromD3D10KHR(
            platform,
            CL_D3D10_DEVICE_KHR,
            pDevice,
            CL_ALL_DEVICES_FOR_D3D10_KHR,
            0,
            NULL,
            &num_device_devices);
        TestRequire(
            (result == CL_SUCCESS || result == CL_DEVICE_NOT_FOUND),
            "clGetDeviceIDsFromD3D10KHR failed.");

        if (result == CL_DEVICE_NOT_FOUND)
        {
            TestPrint("No devices found for D3D device.\n");
        }
        else
        {
            // if there were devices, query them
            device_devices = new cl_device_id[num_device_devices];
            result = clGetDeviceIDsFromD3D10KHR(
                platform,
                CL_D3D10_DEVICE_KHR,
                pDevice,
                CL_ALL_DEVICES_FOR_D3D10_KHR,
                num_device_devices,
                device_devices,
                NULL);
            TestRequire(
                (result == CL_SUCCESS),
                "clGetDeviceIDsFromD3D10KHR failed.");
        }

        // require that each cl_device_id returned for the ID3D10Device was among the devices listed for the adapter
        for (cl_uint device_device = 0; device_device < num_device_devices; ++device_device)
        {
            cl_uint adapter_device;
            for (adapter_device = 0; adapter_device < num_adapter_devices; ++adapter_device)
            {
                if (device_devices[device_device] == adapter_devices[adapter_device])
                {
                    break;
                }
            }
            TestRequire(
                (adapter_device != num_adapter_devices),
                "CL_D3D10_DEVICE_KHR devices not a subset of CL_D3D10_DXGI_ADAPTER_KHR devices");
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
}

bool TestDeviceContextCreate(
    cl_device_id device,
    ID3D10Device* pDevice,
    cl_context* out_context,
    cl_command_queue* out_command_queue)
{
    cl_int result = CL_SUCCESS;
    cl_context context = NULL;
    cl_command_queue command_queue = NULL;

    ID3D10Device* clDevice = NULL;

    bool succeeded = false;

    log_info("Context creation");

    cl_context_properties properties[5];

    // create the context
    properties[0] = (cl_context_properties)CL_CONTEXT_D3D10_DEVICE_KHR;
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
        "clCreateContext with CL_CONTEXT_D3D10_DEVICE_KHR failed");
    result = clReleaseContext(context);
    TestRequire(
        (result == CL_SUCCESS),
        "clReleaseContext with CL_CONTEXT_D3D10_DEVICE_KHR failed");

    // create the context
    properties[0] = (cl_context_properties)CL_CONTEXT_D3D10_DEVICE_KHR;
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
        "clCreateContext with CL_CONTEXT_D3D10_DEVICE_KHR failed");
    result = clReleaseContext(context);
    TestRequire(
        (result == CL_SUCCESS),
        "clReleaseContext with CL_CONTEXT_D3D10_DEVICE_KHR failed");

    // create the context
    properties[0] = (cl_context_properties)CL_CONTEXT_D3D10_DEVICE_KHR;
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
        "clCreateContext with CL_CONTEXT_D3D10_DEVICE_KHR failed");

    // create the command queue
    TestPrint("Creating a command queue.\n");
    command_queue = clCreateCommandQueueWithProperties(
        context,
        device,
        NULL,
        &result);
    TestRequire(
        (result == CL_SUCCESS),
        "clCreateContext with CL_CONTEXT_D3D10_DEVICE_KHR failed");

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
    return succeeded;
}
