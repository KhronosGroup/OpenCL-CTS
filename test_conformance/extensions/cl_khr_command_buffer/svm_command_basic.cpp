
#include "basic_command_buffer.h"
#include "command_buffer_test_base.h"
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
    int error =
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
    if (clCommandSVMMemcpyKHR == nullptr || clCommandSVMMemfillKHR == nullptr)
    {
        log_info("Platform does not support clCommandSVMMemcpyKHR or "
                 "clCommandSVMMemfillKHR\n");
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
    BasicCommandBufferTest::init_extension_functions();

    cl_platform_id platform;
    cl_int error = clGetDeviceInfo(device, CL_DEVICE_PLATFORM,
                                   sizeof(cl_platform_id), &platform, nullptr);
    test_error(error, "clGetDeviceInfo for CL_DEVICE_PLATFORM failed");

    GET_EXTENSION_ADDRESS(clCommandSVMMemfillKHR);
    GET_EXTENSION_ADDRESS(clCommandSVMMemcpyKHR);

    return CL_SUCCESS;
}
