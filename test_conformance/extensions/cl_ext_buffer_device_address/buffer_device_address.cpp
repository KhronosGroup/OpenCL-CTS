// Copyright (c) 2024 The Khronos Group Inc.
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

#include "harness/typeWrappers.h"
#include <cinttypes>

#define BUF_SIZE 1024
#define BUF_SIZE_STR "1024"

namespace {

static const char *program_source =
    R"raw(
    // A kernel that gets the device-seen address of the buffer.
    __kernel void get_addr (__global int *buffer,
                          __global ulong* addr) {
      for (int i = 0; i < BUF_SIZE; ++i)
      buffer[i] += 1;
      *addr = (ulong)buffer;
    }

    // A kernel that accesses another buffer indirectly.
    __kernel void indirect_access (__global long* in_addr,
                                 __global int* out) {
      *out = **(int __global* __global*)in_addr;
    }

    // A kernel that gets passed a pointer to a middle of a buffer,
    // with the data _before_ the passed pointer. Tests the property
    // of sub-buffers to synchronize the whole parent buffer when
    // using the CL_MEM_BUFFER_DEVICE_ADDRESS flag.
    __kernel void ptr_arith (__global int* in_addr,
                           __global int* out) {
      *out = *(in_addr - 2);
    }
    )raw";

class BufferDeviceAddressTest {

public:
    BufferDeviceAddressTest(cl_device_id device, cl_context context,
                            cl_command_queue queue,
                            cl_mem_properties address_type)
        : device(device), context(context), queue(queue),
          address_type(address_type)
    {}

    bool Skip()
    {
        cl_int error = 0;

        cl_mem_properties buf_props[] = { address_type, CL_TRUE, 0 };
        clMemWrapper TempBuffer = clCreateBufferWithProperties(
            context, buf_props, CL_MEM_READ_WRITE,
            (size_t)BUF_SIZE * sizeof(cl_int), nullptr, &error);
        return (error != CL_SUCCESS);
    }

    cl_int RunTest()
    {
        cl_int error;

        clProgramWrapper program;
        clKernelWrapper get_addr_kernel, indirect_access_kernel,
            ptr_arith_kernel;
        clMemWrapper dev_addr_buffer, buffer_long, buffer_int,
            dev_addr_no_host_buffer;

        error = create_single_kernel_helper(context, &program, &get_addr_kernel,
                                            1, &program_source, "get_addr",
                                            "-DBUF_SIZE=" BUF_SIZE_STR);
        test_error(error, "Failed to create program with source\n");

        indirect_access_kernel =
            clCreateKernel(program, "indirect_access", &error);
        test_error(error, "Failed to create kernel indirect_access\n");

        ptr_arith_kernel = clCreateKernel(program, "ptr_arith", &error);
        test_error(error, "Failed to create kernel ptr_arith\n");

        buffer_long = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                     sizeof(cl_ulong), nullptr, &error);
        test_error(error, "clCreateBuffer failed\n");

        buffer_int = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int),
                                    nullptr, &error);
        test_error(error, "clCreateBuffer failed\n");

        // Test a buffer with hostptr copied data
        cl_mem_properties buf_props[] = { address_type, CL_TRUE, 0 };
        dev_addr_buffer = clCreateBufferWithProperties(
            context, buf_props, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            sizeof(cl_int) * BUF_SIZE, BufferHost, &error);
        test_error(error, "clCreateBuffer with device address 1 failed\n");

        if (test_buffer(dev_addr_buffer, buffer_long, get_addr_kernel)
            != TEST_PASS)
            test_fail("test_buffer_host failed\n");

        // Test a buffer which doesn't have any hostptr associated with it.
        dev_addr_no_host_buffer = clCreateBufferWithProperties(
            context, buf_props, CL_MEM_READ_WRITE, sizeof(cl_int) * BUF_SIZE,
            nullptr, &error);
        test_error(error, "clCreateBuffer with device address 2 failed\n");

        if (test_buffer(dev_addr_no_host_buffer, buffer_long, get_addr_kernel)
            != TEST_PASS)
            test_fail("test_buffer_no_host failed\n");

        // Test a buffer passed indirectly
        if (test_indirect_buffer(dev_addr_buffer, buffer_long, buffer_int,
                                 indirect_access_kernel)
            != TEST_PASS)
            test_fail("test_indirect_buffer failed\n");

        if (test_set_kernel_arg(dev_addr_buffer, buffer_int, ptr_arith_kernel)
            != TEST_PASS)
            test_fail("test_set_kernel_arg failed\n");

        if (test_svm_buffer() == TEST_FAIL)
            test_fail("test_svm_buffer failed\n");

        return TEST_PASS;
    }

private:
    int BufferHost[BUF_SIZE];
    size_t global_size_one[3] = { 1, 1, 1 };
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_mem_properties address_type;

    int check_device_address_from_api(cl_mem buf,
                                      cl_mem_device_address_ext &Addr)
    {
        Addr = 0;
        cl_int error = clGetMemObjectInfo(buf, CL_MEM_DEVICE_ADDRESS_EXT,
                                          sizeof(Addr), &Addr, NULL);
        test_error(error,
                   "clGetMemObjectInfo(CL_MEM_DEVICE_ADDRESS_EXT) failed\n");
        if (Addr == 0)
        {
            print_error(error,
                        "clGetMemObjectInfo(CL_MEM_DEVICE_ADDRESS_EXT) "
                        "returned 0 as address\n");
            return CL_INVALID_VALUE;
        }
        return CL_SUCCESS;
    }

    int test_svm_buffer()
    {
        clSVMWrapper svm_buffer;
        clMemWrapper buffer;
        cl_int error = 0;

        cl_device_svm_capabilities svm_caps = 0;
        error = clGetDeviceInfo(device, CL_DEVICE_SVM_CAPABILITIES,
                                sizeof(svm_caps), &svm_caps, NULL);
        if (error != CL_SUCCESS)
        {
            print_missing_feature(error,
                                  "Unable to get SVM capabilities, "
                                  "skipping");
            return TEST_SKIP;
        }
        if ((svm_caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER) == 0)
        {
            print_missing_feature(error,
                                  "Device doesn't support "
                                  "CL_DEVICE_SVM_COARSE_"
                                  "GRAIN_BUFFER, skipping");
            return TEST_SKIP;
        }

        svm_buffer =
            clSVMWrapper(context, sizeof(cl_int) * BUF_SIZE,
                         CL_DEVICE_SVM_COARSE_GRAIN_BUFFER | CL_MEM_READ_WRITE);
        if (svm_buffer() == nullptr)
        {
            test_error(CL_OUT_OF_RESOURCES, "SVM allocation failed");
        }

        cl_mem_properties buf_props[] = { address_type, CL_TRUE, 0 };
        buffer = clCreateBufferWithProperties(
            context, buf_props, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
            sizeof(cl_int) * BUF_SIZE, svm_buffer(), &error);
        test_error(error, "clCreateBuffer with device address 1 failed\n");

        cl_mem_device_address_ext Addr = 0;
        error = clGetMemObjectInfo(buffer, CL_MEM_DEVICE_ADDRESS_EXT,
                                   sizeof(Addr), &Addr, NULL);
        test_error(error,
                   "clGetMemObjectInfo(CL_MEM_DEVICE_ADDRESS_EXT) failed\n");

        if ((void *)Addr != svm_buffer())
        {
            test_fail("clGetMemObjectInfo(CL_MEM_DEVICE_ADDRESS_EXT) "
                      "returned different address than clSVMAlloc\n");
        }
        return TEST_PASS;
    }


    int test_buffer(clMemWrapper &dev_addr_buffer, clMemWrapper &plain_buffer,
                    clKernelWrapper &get_addr_kernel)
    {
        cl_int error = 0;
        cl_ulong DeviceAddrFromKernel = 0;
        cl_mem_device_address_ext DeviceAddrFromAPI = 0;

        for (int i = 0; i < BUF_SIZE; ++i)
        {
            BufferHost[i] = i;
        }

        error =
            check_device_address_from_api(dev_addr_buffer, DeviceAddrFromAPI);
        test_error_fail(error,
                        "device address buffer does not have device address");

        error = clEnqueueWriteBuffer(queue, dev_addr_buffer,
                                     CL_FALSE, // block
                                     0, BUF_SIZE * sizeof(cl_int), BufferHost,
                                     0, NULL, NULL);
        test_error_fail(error,
                        "clEnqueueWriteBuffer of dev_addr_buffer failed\n");

        error = clSetKernelArg(get_addr_kernel, 0, sizeof(cl_mem),
                               &dev_addr_buffer);
        test_error_fail(error, "clSetKernelArg 0 failed\n");
        error =
            clSetKernelArg(get_addr_kernel, 1, sizeof(cl_mem), &plain_buffer);
        test_error_fail(error, "clSetKernelArg 1 failed\n");

        error = clEnqueueNDRangeKernel(queue, get_addr_kernel, 1, NULL,
                                       global_size_one, NULL, 0, NULL, NULL);
        test_error_fail(error, "clNDRangeKernel failed\n");

        error = clEnqueueReadBuffer(queue, dev_addr_buffer, CL_FALSE, 0,
                                    sizeof(cl_int) * BUF_SIZE, BufferHost, 0,
                                    NULL, NULL);
        test_error_fail(error,
                        "clEnqueueReadBuffer of dev_addr_buffer failed\n");

        error = clEnqueueReadBuffer(queue, plain_buffer, CL_FALSE, 0,
                                    sizeof(cl_ulong), &DeviceAddrFromKernel, 0,
                                    NULL, NULL);
        test_error_fail(error, "clEnqueueReadBuffer of buffer failed\n");

        error = clFinish(queue);
        test_error_fail(error, "clFinish failed\n");

        for (int i = 0; i < BUF_SIZE; ++i)
        {
            if (BufferHost[i] != (i + 1))
            {
                test_fail("BufferHost[%i] expected "
                          "to be: %i, but is: %i\n",
                          i, i + 1, BufferHost[i]);
            }
        }

        if (DeviceAddrFromAPI != DeviceAddrFromKernel)
        {
            test_fail("DeviceAddrFromAPI(%" PRIu64
                      ") != DeviceAddrFromKernel(%" PRIu64 ")\n",
                      (uint64_t)DeviceAddrFromAPI,
                      (uint64_t)DeviceAddrFromKernel);
        }
        return TEST_PASS;
    }

    int test_indirect_buffer(clMemWrapper &dev_addr_buffer,
                             clMemWrapper &buffer_in_long,
                             clMemWrapper &buffer_out_int,
                             clKernelWrapper &ind_access_kernel)
    {
        cl_int error = 0;
        cl_mem_device_address_ext DeviceAddrFromAPI = 0;

        int DataIn = 0x12348765;
        int DataOut = -1;

        // A devaddr buffer with the payload data.
        error = clEnqueueWriteBuffer(queue, dev_addr_buffer,
                                     CL_TRUE, // block
                                     0, sizeof(cl_int), &DataIn, 0, NULL, NULL);
        test_error_fail(error,
                        "clEnqueueWriteBuffer of dev_addr_buffer failed\n");

        error =
            check_device_address_from_api(dev_addr_buffer, DeviceAddrFromAPI);
        test_error_fail(error,
                        "device address buffer does not have device address")

            // A basic buffer used to pass the other buffer's address.
            error = clEnqueueWriteBuffer(queue, buffer_in_long,
                                         CL_TRUE, // block
                                         0, sizeof(DeviceAddrFromAPI),
                                         &DeviceAddrFromAPI, 0, NULL, NULL);
        test_error_fail(error,
                        "clEnqueueWriteBuffer of dev_addr_buffer failed\n");

        error = clSetKernelArg(ind_access_kernel, 0, sizeof(cl_mem),
                               &buffer_in_long);
        test_error_fail(error, "clSetKernelArg 0 failed\n");
        error = clSetKernelArg(ind_access_kernel, 1, sizeof(cl_mem),
                               &buffer_out_int);
        test_error_fail(error, "clSetKernelArg 1 failed\n");

        error = clSetKernelExecInfo(
            ind_access_kernel, CL_KERNEL_EXEC_INFO_DEVICE_PTRS_EXT,
            sizeof(DeviceAddrFromAPI), &DeviceAddrFromAPI);
        test_error_fail(error,
                        "Setting indirect access for "
                        "device ptrs failed!\n");

        error = clEnqueueNDRangeKernel(queue, ind_access_kernel, 1, NULL,
                                       global_size_one, NULL, 0, NULL, NULL);
        test_error_fail(error, "clNDRangeKernel failed\n");

        error = clEnqueueReadBuffer(queue, buffer_out_int, CL_FALSE, 0,
                                    sizeof(cl_int), &DataOut, 0, NULL, NULL);
        test_error_fail(error, "clEnqueueReadBuffer of buffer failed\n");

        error = clFinish(queue);
        test_error_fail(error, "clFinish failed\n");

        for (int i = 0; i < BUF_SIZE; ++i)
        {
            if (BufferHost[i] != i + 1)
            {
                test_fail("PinnedBufferHost[%i] expected "
                          "to be: %i, but is: %i\n",
                          i, i + 1, BufferHost[i]);
            }
        }

        if (DataIn != DataOut)
        {
            test_fail("Passing data via indirect buffers failed. "
                      "Got: %i expected: %i\n",
                      DataOut, DataIn);
        }
        return TEST_PASS;
    }

    int test_set_kernel_arg(clMemWrapper &dev_addr_buffer,
                            clMemWrapper &buffer_out_int,
                            clKernelWrapper &ptr_arith_kernel)
    {
        cl_int error = 0;
        cl_mem_device_address_ext DeviceAddrFromAPI = 0;
        int DataOut = -1;
        int DataIn = 0x12348765;

        clSetKernelArgDevicePointerEXT_fn clSetKernelArgDevicePointer =
            (clSetKernelArgDevicePointerEXT_fn)
                clGetExtensionFunctionAddressForPlatform(
                    getPlatformFromDevice(device),
                    "clSetKernelArgDevicePointerEXT");
        if (clSetKernelArgDevicePointer == nullptr)
            test_error_fail(
                error, "Cannot get address of clSetKernelArgDevicePointerEXT");

        error = clEnqueueWriteBuffer(queue, dev_addr_buffer,
                                     CL_FALSE, // block
                                     0, sizeof(cl_int), &DataIn, 0, NULL, NULL);
        test_error_fail(error,
                        "clEnqueueWriteBuffer of dev_addr_buffer failed\n");

        error =
            check_device_address_from_api(dev_addr_buffer, DeviceAddrFromAPI);
        test_error_fail(error, "dev_addr_buffer does not have device address");

        cl_mem_device_address_ext DeviceAddrFromAPIP2 =
            (cl_mem_device_address_ext)(((cl_uint *)DeviceAddrFromAPI) + 2);
        error = clSetKernelArgDevicePointer(ptr_arith_kernel, 0,
                                            DeviceAddrFromAPIP2);
        test_error_fail(error, "clSetKernelArgDevicePointer failed\n");
        error = clSetKernelArg(ptr_arith_kernel, 1, sizeof(cl_mem),
                               &buffer_out_int);
        test_error_fail(error, "clSetKernelArg 1 failed\n");

        error = clEnqueueNDRangeKernel(queue, ptr_arith_kernel, 1, NULL,
                                       global_size_one, NULL, 0, NULL, NULL);
        test_error_fail(error, "clNDRangeKernel failed\n");

        error = clEnqueueReadBuffer(queue, buffer_out_int, CL_FALSE, 0,
                                    sizeof(cl_int), &DataOut, 0, NULL, NULL);
        test_error_fail(error, "clEnqueueReadBuffer of buffer failed\n");

        error = clFinish(queue);
        test_error_fail(error, "clFinish failed\n");

        if (DataIn != DataOut)
        {
            test_fail("Negative offsetting from passed in pointer failed: "
                      "got: %i expected: %i",
                      DataOut, DataIn);
        }
        return TEST_PASS;
    }
};

}

REGISTER_TEST(private_address)
{
    BufferDeviceAddressTest test_fixture = BufferDeviceAddressTest(
        device, context, queue, CL_MEM_DEVICE_PRIVATE_ADDRESS_EXT);

    if (test_fixture.Skip())
    {
        log_info("Test fixture skip\n");
        return TEST_SKIPPED_ITSELF;
    }

    cl_int error = test_fixture.RunTest();
    test_error_ret(error, "Test Failed", TEST_FAIL);

    return TEST_PASS;
}
