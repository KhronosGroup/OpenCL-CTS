//
// Copyright (c) 2023 The Khronos Group Inc.
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

#include "svm_command_basic.h"

//--------------------------------------------------------------------------

bool BasicSVMCommandBufferTest::Skip()
{
    if (BasicCommandBufferTest::Skip()) return true;

    Version version = get_device_cl_version(device);
    if (version < Version(2, 0))
    {
        log_info("test requires OpenCL 2.x/3.0 device");
        return true;
    }

    cl_device_svm_capabilities svm_capabilities;
    cl_int error =
        clGetDeviceInfo(device, CL_DEVICE_SVM_CAPABILITIES,
                        sizeof(svm_capabilities), &svm_capabilities, NULL);
    if (error != CL_SUCCESS)
    {
        print_error(error, "Unable to query CL_DEVICE_SVM_CAPABILITIES");
        return true;
    }

    if (svm_capabilities == 0)
    {
        log_info("Device property CL_DEVICE_SVM_COARSE_GRAIN_BUFFER not "
                 "supported \n");
        return true;
    }

    if (init_extension_functions() != CL_SUCCESS)
    {
        log_error("Unable to initialise extension functions");
        return true;
    }

    return false;
}

//--------------------------------------------------------------------------

cl_int BasicSVMCommandBufferTest::SetUpKernelArgs(void)
{
    size_t size = sizeof(cl_int) * num_elements * buffer_size_multiplier;
    svm_in_mem = clSVMWrapper(context, size);
    if (svm_in_mem() == nullptr)
    {
        log_error("Unable to allocate SVM memory");
        return CL_OUT_OF_RESOURCES;
    }
    svm_out_mem = clSVMWrapper(context, size);
    if (svm_out_mem() == nullptr)
    {
        log_error("Unable to allocate SVM memory");
        return CL_OUT_OF_RESOURCES;
    }
    return CL_SUCCESS;
}

//--------------------------------------------------------------------------

cl_int BasicSVMCommandBufferTest::init_extension_functions()
{
    cl_int error = BasicCommandBufferTest::init_extension_functions();
    test_error(error, "Unable to initialise extension functions");

    cl_platform_id platform;
    error = clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(cl_platform_id),
                            &platform, nullptr);
    test_error(error, "clGetDeviceInfo for CL_DEVICE_PLATFORM failed");

    GET_EXTENSION_ADDRESS(clCommandSVMMemFillKHR);
    GET_EXTENSION_ADDRESS(clCommandSVMMemcpyKHR);

    return CL_SUCCESS;
}
