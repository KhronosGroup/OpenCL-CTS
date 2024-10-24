//
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

#include <extensionHelpers.h>
#include "mutable_command_basic.h"

#include <CL/cl.h>
#include <CL/cl_ext.h>

#include <vector>

namespace {

////////////////////////////////////////////////////////////////////////////////
// command buffer with all available mutable dispatch tests which handle cases:
// CL_MUTABLE_DISPATCH_GLOBAL_OFFSET_KHR
// CL_MUTABLE_DISPATCH_GLOBAL_SIZE_KHR
// CL_MUTABLE_DISPATCH_LOCAL_SIZE_KHR
// CL_MUTABLE_DISPATCH_ARGUMENTS_KHR
// CL_MUTABLE_DISPATCH_EXEC_INFO_KHR

struct MutableCommandFullDispatch : InfoMutableCommandBufferTest
{
    using InfoMutableCommandBufferTest::InfoMutableCommandBufferTest;

    MutableCommandFullDispatch(cl_device_id device, cl_context context,
                               cl_command_queue queue)
        : InfoMutableCommandBufferTest(device, context, queue),
          svm_buffers(context), group_size(0), available_caps(0)
    {}

    bool Skip() override
    {
        cl_mutable_dispatch_fields_khr requested =
            CL_MUTABLE_DISPATCH_GLOBAL_OFFSET_KHR
            | CL_MUTABLE_DISPATCH_GLOBAL_SIZE_KHR
            | CL_MUTABLE_DISPATCH_LOCAL_SIZE_KHR
            | CL_MUTABLE_DISPATCH_ARGUMENTS_KHR
            | CL_MUTABLE_DISPATCH_EXEC_INFO_KHR;


        cl_int error =
            clGetDeviceInfo(device, CL_DEVICE_MUTABLE_DISPATCH_CAPABILITIES_KHR,
                            sizeof(available_caps), &available_caps, nullptr);
        test_error(error, "clGetDeviceInfo failed");

        available_caps &= requested;

        cl_device_svm_capabilities svm_caps;
        bool svm_capabilities =
            !clGetDeviceInfo(device, CL_DEVICE_SVM_CAPABILITIES,
                             sizeof(svm_caps), &svm_caps, NULL)
            && svm_caps != 0;

        if (!svm_capabilities)
            available_caps &= ~CL_MUTABLE_DISPATCH_EXEC_INFO_KHR;

        // require at least one mutable capabillity
        return (available_caps == 0) || InfoMutableCommandBufferTest::Skip();
    }

    // setup kernel program specific for command buffer with full mutable
    // dispatch test
    cl_int SetUpKernel() override
    {
        const char *kernel_str_svm =
            R"(typedef struct {
                global int* ptr;
            } wrapper;
            __kernel void full_dispatch(__global int *src, __global wrapper *dst)
            {
                size_t gid = get_global_id(0) % get_global_size(0);
                size_t lid = gid % get_local_size(0);
                dst->ptr[gid] = src[lid];
            })";

        const char *kernel_str_no_svm =
            R"(
            __kernel void full_dispatch(__global int *src, __global int *dst)
            {
                size_t gid = get_global_id(0) % get_global_size(0);
                size_t lid = gid % get_local_size(0);
                dst[gid] = src[lid];
            })";

        cl_int error = CL_SUCCESS;

        if ((available_caps & CL_MUTABLE_DISPATCH_EXEC_INFO_KHR) == 0)
        {
            error = create_single_kernel_helper(context, &program, &kernel, 1,
                                                &kernel_str_no_svm,
                                                "full_dispatch");
        }
        else
        {
            error =
                create_single_kernel_helper(context, &program, &kernel, 1,
                                            &kernel_str_svm, "full_dispatch");
        }
        test_error(error, "Failed to create program with source");

        return CL_SUCCESS;
    }

    // setup kernel arguments specific for command buffer with full mutable
    // dispatch test
    cl_int SetUpKernelArgs() override
    {
        // query max work-group size needed for allocation size of input buffers
        size_t workgroupinfo_size = 0;
        cl_int error = clGetKernelWorkGroupInfo(
            kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(group_size),
            &workgroupinfo_size, NULL);
        test_error(error, "clGetKernelWorkGroupInfo failed");

        group_size = std::min(num_elements, workgroupinfo_size);
        const size_t size_to_allocate_src = group_size * sizeof(cl_int);

        // create and initialize source buffer
        MTdataHolder d(gRandomSeed);
        src_host.resize(group_size);
        for (cl_int i = 0; i < src_host.size(); i++)
        {
            src_host[i] = genrand_int32(d);
        }

        in_mem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                size_to_allocate_src, src_host.data(), &error);
        test_error(error, "Creating test array failed");

        if ((available_caps & CL_MUTABLE_DISPATCH_ARGUMENTS_KHR) != 0)
        {
            in_buf_update =
                clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                               size_to_allocate_src, src_host.data(), &error);
            test_error(error, "Creating test array failed");
        }

        // create and initialize destination buffers
        const size_t size_to_allocate_dst = num_elements * sizeof(cl_int);

        if ((available_caps & CL_MUTABLE_DISPATCH_EXEC_INFO_KHR) != 0)
        {
            svm_buffers.initWrapper = (cl_int *)clSVMAlloc(
                context, CL_MEM_READ_WRITE, sizeof(cl_int *), 0);
            svm_buffers.initBuffer = (cl_int *)clSVMAlloc(
                context, CL_MEM_READ_WRITE, size_to_allocate_dst, 0);
            test_assert_error(svm_buffers.initWrapper != nullptr
                                  && svm_buffers.initBuffer != nullptr,
                              "clSVMAlloc failed for initial execution");

            error = clEnqueueSVMMemcpy(queue, CL_TRUE, svm_buffers.initWrapper,
                                       &svm_buffers.initBuffer,
                                       sizeof(cl_int *), 0, nullptr, nullptr);
            test_error(error, "clEnqueueSVMMemcpy failed for initWrapper");

            const cl_int zero = 0;
            error = clEnqueueSVMMemFill(queue, svm_buffers.initBuffer, &zero,
                                        sizeof(zero), size_to_allocate_dst, 0,
                                        nullptr, nullptr);
            test_error(error, "clEnqueueSVMMemFill failed for initBuffer");

            // Allocate and initialize SVM for modified execution
            svm_buffers.newWrapper = (cl_int *)clSVMAlloc(
                context, CL_MEM_READ_WRITE, sizeof(cl_int *), 0);
            svm_buffers.newBuffer = (cl_int *)clSVMAlloc(
                context, CL_MEM_READ_WRITE, size_to_allocate_dst, 0);
            test_assert_error(svm_buffers.newWrapper != nullptr
                                  && svm_buffers.newBuffer != nullptr,
                              "clSVMAlloc failed for modified execution");

            error = clEnqueueSVMMemcpy(queue, CL_TRUE, svm_buffers.newWrapper,
                                       &svm_buffers.newBuffer, sizeof(cl_int *),
                                       0, nullptr, nullptr);
            test_error(error, "clEnqueueSVMMemcpy failed for newWrapper");

            error = clEnqueueSVMMemFill(queue, svm_buffers.newBuffer, &zero,
                                        sizeof(zero), size_to_allocate_dst, 0,
                                        nullptr, nullptr);
            test_error(error, "clEnqueueSVMMemFill failed for newB");

            error =
                clSetKernelArgSVMPointer(kernel, 1, svm_buffers.initWrapper);
            test_error(error, "clSetKernelArg failed for initWrapper");

            error = clSetKernelExecInfo(kernel, CL_KERNEL_EXEC_INFO_SVM_PTRS,
                                        sizeof(svm_buffers.initBuffer),
                                        &svm_buffers.initBuffer);
            test_error(error, "clSetKernelExecInfo failed for initBuffer");
        }
        else
        {
            out_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                     size_to_allocate_dst, nullptr, &error);
            test_error(error, "Creating test array failed");

            const cl_int pattern = 0;
            error =
                clEnqueueFillBuffer(queue, out_mem, &pattern, sizeof(cl_int), 0,
                                    size_to_allocate_dst, 0, nullptr, nullptr);
            test_error(error, "clEnqueueFillBuffer failed");

            error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem);
            test_error(error, "Unable to set indexed kernel arguments");
        }

        error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem);
        test_error(error, "Unable to set indexed kernel arguments");

        return CL_SUCCESS;
    }

    // Check the results of command buffer execution with svm target
    bool verify_result_svm(int *const buf, const size_t work_size,
                           const size_t offset)
    {
        cl_int error =
            clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_READ, buf,
                            num_elements * sizeof(cl_int), 0, nullptr, nullptr);
        test_error_ret(error, "clEnqueueSVMMap failed for svm buffer", false);

        bool res = compare_result(buf, work_size, offset);

        error = clEnqueueSVMUnmap(queue, buf, 0, nullptr, nullptr);
        test_error(error, "clEnqueueSVMUnmap failed for svm buffer");

        error = clFinish(queue);
        test_error(error, "clFinish failed");

        return res;
    }

    // Check the results of command buffer execution without svm target
    bool verify_result_no_svm(const size_t work_size, const size_t offset)
    {
        cl_int error = CL_SUCCESS;
        const size_t out_buf_size = num_elements * sizeof(cl_int);
        std::vector<cl_int> data(num_elements);
        error = clEnqueueReadBuffer(queue, out_mem, CL_TRUE, 0, out_buf_size,
                                    data.data(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        return compare_result(data.data(), work_size, offset);
    }

    // compare expected values and results of command buffer execution
    bool compare_result(const int *const buf, const size_t work_size,
                        const size_t offset)
    {
        for (size_t i = 0; i < num_elements; i++)
        {
            size_t gid = (offset + i) % num_elements;
            size_t lid = gid % work_size;

            if (buf[gid] != src_host[lid])
            {
                log_error("Modified verification failed at index %zu: Got %d, "
                          "wanted %d\n",
                          i, buf[i], src_host[lid]);
                return false;
            }
        }
        return true;
    }

    // verify the result
    bool verify_result(int *const buf, const size_t work_size,
                       const size_t offset)
    {
        if (buf != nullptr)
        {
            if (!verify_result_svm(buf, group_size, offset)) return false;
        }
        else
        {
            if (!verify_result_no_svm(group_size, offset)) return false;
        }
        return true;
    }

    // run command buffer with full mutable dispatch test
    cl_int Run() override
    {
        cl_command_properties_khr props[] = {
            CL_MUTABLE_DISPATCH_UPDATABLE_FIELDS_KHR, available_caps, 0
        };

        size_t work_offset = 0;
        /* Round the global work size up to nearest multiple of the local work
         * size to ensure work group uniformity. */
        num_elements =
            ((num_elements + group_size - 1) / group_size) * group_size;

        cl_int error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, props, kernel, 1, &work_offset,
            &num_elements, &group_size, 0, nullptr, nullptr, &command);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error = clFinish(queue);
        test_error(error, "clFinish failed");

        // Check the results of the initial execution
        if (!verify_result(
                ((available_caps & CL_MUTABLE_DISPATCH_EXEC_INFO_KHR) != 0)
                    ? svm_buffers.initBuffer
                    : nullptr,
                group_size, work_offset))
            return TEST_FAIL;

        if ((available_caps & CL_MUTABLE_DISPATCH_EXEC_INFO_KHR) == 0)
        {
            // clear output buffer before applying mutable dispatch
            const size_t size_to_allocate_dst = num_elements * sizeof(cl_int);
            const cl_int pattern = 0;
            error =
                clEnqueueFillBuffer(queue, out_mem, &pattern, sizeof(cl_int), 0,
                                    size_to_allocate_dst, 0, nullptr, nullptr);
            test_error(error, "clEnqueueFillBuffer failed");
        }

        // Modify and execute the command buffer
        cl_mutable_dispatch_config_khr dispatch_config{
            command,
            0 /* num_args */,
            0 /* num_svm_arg */,
            0 /* num_exec_infos */,
            0 /* work_dim - 0 means no change to dimensions */,
            nullptr /* arg_list */,
            nullptr /* arg_svm_list - nullptr means no change*/,
            nullptr /* exec_info_list */,
            nullptr /* global_work_offset */,
            nullptr /* global_work_size */,
            nullptr /* local_work_size */
        };

        cl_mutable_dispatch_arg_khr arg0{ 0 };
        cl_mutable_dispatch_arg_khr arg1{ 0 };
        cl_mutable_dispatch_exec_info_khr exec_info{ 0 };

        if ((available_caps & CL_MUTABLE_DISPATCH_ARGUMENTS_KHR) != 0)
        {
            arg0 = { 0, sizeof(cl_mem), &in_buf_update };
            dispatch_config.num_args = 1;
            dispatch_config.arg_list = &arg0;
        }

        if ((available_caps & CL_MUTABLE_DISPATCH_EXEC_INFO_KHR) != 0)
        {
            arg1 = { 1, sizeof(svm_buffers.newWrapper),
                     svm_buffers.newWrapper };

            exec_info.param_name = CL_KERNEL_EXEC_INFO_SVM_PTRS;
            exec_info.param_value_size = sizeof(svm_buffers.newBuffer);
            exec_info.param_value = &svm_buffers.newBuffer;

            dispatch_config.num_svm_args = 1;
            dispatch_config.arg_svm_list = &arg1;
            dispatch_config.num_exec_infos = 1;
            dispatch_config.exec_info_list = &exec_info;
        }

        if ((available_caps & CL_MUTABLE_DISPATCH_GLOBAL_OFFSET_KHR) != 0)
        {
            work_offset = 42;
            dispatch_config.global_work_offset = &work_offset;
        }

        if ((available_caps & CL_MUTABLE_DISPATCH_LOCAL_SIZE_KHR) != 0)
        {
            group_size /= 2;
            dispatch_config.local_work_size = &group_size;
        }

        if ((available_caps & CL_MUTABLE_DISPATCH_GLOBAL_SIZE_KHR) != 0)
        {
            num_elements /= 2;
            /* Round the global work size up to nearest multiple of the local
             * work size to ensure work group uniformity. */
            num_elements =
                ((num_elements + group_size - 1) / group_size) * group_size;

            dispatch_config.global_work_size = &num_elements;
        }

        cl_uint num_configs = 1;
        cl_command_buffer_update_type_khr config_types[1] = {
            CL_STRUCTURE_TYPE_MUTABLE_DISPATCH_CONFIG_KHR
        };
        const void *configs[1] = { &dispatch_config };
        error = clUpdateMutableCommandsKHR(command_buffer, num_configs,
                                           config_types, configs);
        test_error(error, "clUpdateMutableCommandsKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error = clFinish(queue);
        test_error(error, "clFinish failed");

        // Check the results of the modified execution
        auto check_info_result = [&](const cl_uint param, const size_t test) {
            size_t info_res = 0;
            error = clGetMutableCommandInfoKHR(command, param, sizeof(info_res),
                                               &info_res, nullptr);
            test_error_ret(error, "clGetMutableCommandInfoKHR failed", false);

            if (info_res != test)
            {
                log_error("ERROR: Wrong value returned from "
                          "clGetMutableCommandInfoKHR.");
                return false;
            }
            return true;
        };

        if ((available_caps & CL_MUTABLE_DISPATCH_GLOBAL_SIZE_KHR) != 0
            && !check_info_result(CL_MUTABLE_DISPATCH_GLOBAL_WORK_SIZE_KHR,
                                  num_elements))
            return TEST_FAIL;

        if ((available_caps & CL_MUTABLE_DISPATCH_GLOBAL_OFFSET_KHR) != 0
            && !check_info_result(CL_MUTABLE_DISPATCH_GLOBAL_WORK_OFFSET_KHR,
                                  work_offset))
            return TEST_FAIL;

        if ((available_caps & CL_MUTABLE_DISPATCH_LOCAL_SIZE_KHR) != 0
            && !check_info_result(CL_MUTABLE_DISPATCH_LOCAL_WORK_SIZE_KHR,
                                  group_size))
            return TEST_FAIL;

        if (!verify_result(
                ((available_caps & CL_MUTABLE_DISPATCH_EXEC_INFO_KHR) != 0)
                    ? svm_buffers.newBuffer
                    : nullptr,
                group_size, work_offset))
            return TEST_FAIL;

        return TEST_PASS;
    }

    // all available command mutable dispatch test attributes
    cl_mutable_command_khr command;
    clMemWrapper in_buf_update;

    struct ScopeGuard
    {
        ScopeGuard(const cl_context &c)
            : context(c), initWrapper(nullptr), initBuffer(nullptr),
              newWrapper(nullptr), newBuffer(nullptr)
        {}
        ~ScopeGuard()
        {
            if (initWrapper != nullptr) clSVMFree(context, initWrapper);
            if (initBuffer != nullptr) clSVMFree(context, initBuffer);
            if (newWrapper != nullptr) clSVMFree(context, newWrapper);
            if (newBuffer != nullptr) clSVMFree(context, newBuffer);
        }

        cl_context context;
        cl_int *initWrapper;
        cl_int *initBuffer;
        cl_int *newWrapper;
        cl_int *newBuffer;
    };

    ScopeGuard svm_buffers;
    std::vector<cl_int> src_host;
    size_t group_size;
    cl_mutable_dispatch_fields_khr available_caps;
};

}

int test_mutable_command_full_dispatch(cl_device_id device, cl_context context,
                                       cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<MutableCommandFullDispatch>(device, context, queue,
                                                      num_elements);
}
