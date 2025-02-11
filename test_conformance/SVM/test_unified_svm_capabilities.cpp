//
// Copyright (c) 2025 The Khronos Group Inc.
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

#include "unified_svm_fixture.h"
#include <cinttypes>
#include <memory>

struct UnifiedSVMCapabilities : UnifiedSVMBase
{
    UnifiedSVMCapabilities(cl_context context, cl_device_id device,
                           cl_command_queue queue, int num_elements)
        : UnifiedSVMBase(context, device, queue, num_elements)
    {}

    cl_int test_CL_SVM_CAPABILITY_SINGLE_ADDRESS_SPACE_KHR(cl_uint typeIndex)
    {
        cl_int err;

        if (!kernel_StorePointer)
        {
            err = createStorePointerKernel();
            test_error(err, "could not create StorePointer kernel");
        }

        auto mem = get_usvm_wrapper<cl_int>(typeIndex);
        err = mem->allocate(1);
        test_error(err, "could not allocate source memory");

        clMemWrapper out = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                          sizeof(cl_int*), nullptr, &err);
        test_error(err, "could not create destination buffer");

        err |= clSetKernelArgSVMPointer(kernel_StorePointer, 0, mem->get_ptr());
        err |= clSetKernelArg(kernel_StorePointer, 1, sizeof(out), &out);
        test_error(err, "could not set kernel arguments");

        size_t global_work_size = 1;
        err = clEnqueueNDRangeKernel(queue, kernel_StorePointer, 1, nullptr,
                                     &global_work_size, nullptr, 0, nullptr,
                                     nullptr);
        test_error(err, "clEnqueueNDRangeKernel failed");

        err = clFinish(queue);
        test_error(err, "clFinish failed");

        cl_int* check = nullptr;
        err = clEnqueueReadBuffer(queue, out, CL_TRUE, 0, sizeof(cl_int*),
                                  &check, 0, nullptr, nullptr);
        test_error(err, "could not read output buffer");

        test_assert_error(check == mem->get_ptr(),
                          "stored pointer does not match input pointer");

        return CL_SUCCESS;
    }

    cl_int test_CL_SVM_CAPABILITY_DEVICE_UNASSOCIATED_KHR(cl_uint typeIndex)
    {
        const auto caps = deviceUSVMCaps[typeIndex];
        if (caps & CL_SVM_CAPABILITY_SYSTEM_ALLOCATED_KHR)
        {
            return CL_SUCCESS;
        }

        cl_int err;

        void* ptr;

        ptr = clSVMAllocWithPropertiesKHR(context, nullptr, typeIndex, 1, &err);
        test_error(err, "allocating without associated device failed");

        err = clSVMFreeWithPropertiesKHR(context, nullptr, 0, ptr);
        test_error(err, "freeing without associated device failed");

        cl_svm_alloc_properties_khr props[] = {
            CL_SVM_ALLOC_ASSOCIATED_DEVICE_HANDLE_KHR,
            reinterpret_cast<cl_svm_alloc_properties_khr>(device), 0
        };
        ptr = clSVMAllocWithPropertiesKHR(context, props, typeIndex, 1, &err);
        test_error(err, "allocating with associated device failed");

        err = clSVMFreeWithPropertiesKHR(context, nullptr, 0, ptr);
        test_error(err, "freeing with associated device failed");

        return CL_SUCCESS;
    }

    cl_int test_CL_SVM_CAPABILITY_HOST_READ_KHR(cl_uint typeIndex)
    {
        const auto caps = deviceUSVMCaps[typeIndex];
        cl_int err;

        auto mem = get_usvm_wrapper<cl_int>(typeIndex);
        err = mem->allocate(1);
        test_error(err, "could not allocate usvm memory");

        cl_int value = genrand_int32(d);
        err = mem->write(value);
        test_error(err, "could not write to usvm memory");

        cl_int check = mem->get_ptr()[0];
        test_assert_error(check == value, "read value does not match");

        if (caps & CL_SVM_CAPABILITY_DEVICE_WRITE_KHR)
        {
            value = genrand_int32(d);
            err = clEnqueueSVMMemcpy(queue, CL_TRUE, mem->get_ptr(), &value,
                                     sizeof(value), 0, nullptr, nullptr);
            test_error(err, "could not write to usvm memory on the device");

            check = mem->get_ptr()[0];
            test_assert_error(check == value, "read value does not match");
        }

        return CL_SUCCESS;
    }

    cl_int test_CL_SVM_CAPABILITY_HOST_WRITE_KHR(cl_uint typeIndex)
    {
        const auto caps = deviceUSVMCaps[typeIndex];
        cl_int err;

        auto mem = get_usvm_wrapper<cl_int>(typeIndex);
        err = mem->allocate(1);
        test_error(err, "could not allocate usvm memory");

        cl_int value = genrand_int32(d);
        mem->get_ptr()[0] = value;

        cl_int check;
        err = mem->read(check);
        test_error(err, "could not read from usvm memory");
        test_assert_error(check == value, "read value does not match");

        if (caps & CL_SVM_CAPABILITY_DEVICE_READ_KHR)
        {
            value = genrand_int32(d);
            mem->get_ptr()[0] = value;

            err = clEnqueueSVMMemcpy(queue, CL_TRUE, &check, mem->get_ptr(),
                                     sizeof(value), 0, nullptr, nullptr);
            test_error(err, "could not read from usvm memory on the device");
            test_assert_error(check == value, "read value does not match");
        }

        return CL_SUCCESS;
    }

    cl_int test_CL_SVM_CAPABILITY_HOST_MAP_KHR(cl_uint typeIndex)
    {
        const auto caps = deviceUSVMCaps[typeIndex];
        cl_int err;

        auto mem = get_usvm_wrapper<cl_int>(typeIndex);
        err = mem->allocate(1);
        test_error(err, "could not allocate usvm memory");

        // map for writing, then map for reading
        cl_int value = genrand_int32(d);
        err =
            clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION,
                            mem->get_ptr(), sizeof(value), 0, nullptr, nullptr);
        test_error(err, "could not map usvm memory for writing");

        mem->get_ptr()[0] = value;
        err = clEnqueueSVMUnmap(queue, mem->get_ptr(), 0, nullptr, nullptr);
        test_error(err, "could not unmap usvm memory");

        err = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_READ, mem->get_ptr(),
                              sizeof(value), 0, nullptr, nullptr);
        test_error(err, "could not map usvm memory for reading");

        cl_int check = mem->get_ptr()[0];
        err = clEnqueueSVMUnmap(queue, mem->get_ptr(), 0, nullptr, nullptr);
        test_error(err, "could not unmap usvm memory");

        test_assert_error(check == value, "read value does not match");

        // write directly on the host, map for reading on the host
        if (caps & CL_SVM_CAPABILITY_HOST_WRITE_KHR)
        {
            value = genrand_int32(d);
            mem->get_ptr()[0] = value;

            err = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_READ, mem->get_ptr(),
                                  sizeof(value), 0, nullptr, nullptr);
            test_error(err, "could not map usvm memory for reading");

            check = mem->get_ptr()[0];
            err = clEnqueueSVMUnmap(queue, mem->get_ptr(), 0, nullptr, nullptr);
            test_error(err, "could not unmap usvm memory");

            test_assert_error(check == value, "read value does not match");
        }

        // map for writing on the host, read directly on the host
        if (caps & CL_SVM_CAPABILITY_HOST_READ_KHR)
        {
            value = genrand_int32(d);
            err = clEnqueueSVMMap(
                queue, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, mem->get_ptr(),
                sizeof(value), 0, nullptr, nullptr);
            test_error(err, "could not map usvm memory for writing");

            mem->get_ptr()[0] = value;
            err = clEnqueueSVMUnmap(queue, mem->get_ptr(), 0, nullptr, nullptr);
            test_error(err, "could not unmap usvm memory");

            err = clFinish(queue);
            test_error(err, "clFinish failed");

            check = mem->get_ptr()[0];
            test_assert_error(check == value, "read value does not match");
        }

        // write on the device, map for reading on the host
        if (caps & CL_SVM_CAPABILITY_DEVICE_WRITE_KHR)
        {
            value = genrand_int32(d);
            err = clEnqueueSVMMemcpy(queue, CL_TRUE, mem->get_ptr(), &value,
                                     sizeof(value), 0, nullptr, nullptr);
            test_error(err, "could not write to usvm memory on the device");

            err = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_READ, mem->get_ptr(),
                                  sizeof(value), 0, nullptr, nullptr);
            test_error(err, "could not map usvm memory for reading");

            check = mem->get_ptr()[0];
            err = clEnqueueSVMUnmap(queue, mem->get_ptr(), 0, nullptr, nullptr);
            test_error(err, "could not unmap usvm memory");

            test_assert_error(check == value, "read value does not match");
        }

        // map for writing on the host, read on the device
        if (caps & CL_SVM_CAPABILITY_DEVICE_READ_KHR)
        {
            cl_int value = genrand_int32(d);
            err = clEnqueueSVMMap(
                queue, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, mem->get_ptr(),
                sizeof(value), 0, nullptr, nullptr);
            test_error(err, "could not map usvm memory for writing");

            mem->get_ptr()[0] = value;

            err = clEnqueueSVMUnmap(queue, mem->get_ptr(), 0, nullptr, nullptr);
            test_error(err, "could not unmap usvm memory");

            cl_int check;
            err = clEnqueueSVMMemcpy(queue, CL_TRUE, &check, mem->get_ptr(),
                                     sizeof(value), 0, nullptr, nullptr);
            test_error(err, "could not read from usvm memory on the device");

            test_assert_error(check == value, "read value does not match");
        }

        return CL_SUCCESS;
    }

    cl_int test_CL_SVM_CAPABILITY_DEVICE_READ_KHR(cl_uint typeIndex)
    {
        cl_int err;

        // setup
        auto mem = get_usvm_wrapper<cl_int>(typeIndex);
        err = mem->allocate(1);
        test_error(err, "could not allocate usvm memory");

        if (!kernel_CopyMemory)
        {
            err = createCopyMemoryKernel();
            test_error(err, "could not create CopyMemory kernel");
        }

        // test reading via memcpy:
        cl_int value = genrand_int32(d);
        err = mem->write(value);
        test_error(err, "could not write to usvm memory");

        cl_int check;
        err = clEnqueueSVMMemcpy(queue, CL_TRUE, &check, mem->get_ptr(),
                                 sizeof(value), 0, nullptr, nullptr);
        test_error(err, "could not read from usvm memory with memcpy");

        test_assert_error(check == value,
                          "read value with memcpy does not match");

        // test reading via kernel
        value = genrand_int32(d);
        err = mem->write(value);
        test_error(err, "could not write to usvm memory");

        clMemWrapper out = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                          sizeof(cl_int), nullptr, &err);
        test_error(err, "could not create output buffer");

        err |= clSetKernelArgSVMPointer(kernel_CopyMemory, 0, mem->get_ptr());
        err |= clSetKernelArg(kernel_CopyMemory, 1, sizeof(out), &out);
        test_error(err, "could not set kernel arguments");

        size_t global_work_size = 1;
        err = clEnqueueNDRangeKernel(queue, kernel_CopyMemory, 1, nullptr,
                                     &global_work_size, nullptr, 0, nullptr,
                                     nullptr);
        test_error(err, "clEnqueueNDRangeKernel failed");

        err = clFinish(queue);
        test_error(err, "clFinish failed");

        err = clEnqueueReadBuffer(queue, out, CL_TRUE, 0, sizeof(cl_int),
                                  &check, 0, nullptr, nullptr);
        test_error(err, "could not read output buffer");

        test_assert_error(check == value,
                          "read value with kernel does not match");

        return CL_SUCCESS;
    }

    cl_int test_CL_SVM_CAPABILITY_DEVICE_WRITE_KHR(cl_uint typeIndex)
    {
        cl_int err;

        // setup
        auto mem = get_usvm_wrapper<cl_int>(typeIndex);
        err = mem->allocate(1);
        test_error(err, "could not allocate usvm memory");

        if (!kernel_CopyMemory)
        {
            err = createCopyMemoryKernel();
            test_error(err, "could not create CopyMemory kernel");
        }

        // test writing via memfill
        cl_int value = genrand_int32(d);
        err = clEnqueueSVMMemFill(queue, mem->get_ptr(), &value, sizeof(value),
                                  sizeof(value), 0, nullptr, nullptr);
        test_error(err, "could not write to usvm memory with memfill");

        cl_int check;
        err = mem->read(check);
        test_error(err, "could not read from usvm memory");

        test_assert_error(check == value,
                          "read value with memfill does not match");

        // test writing via memcpy
        value = genrand_int32(d);
        err = clEnqueueSVMMemcpy(queue, CL_TRUE, mem->get_ptr(), &value,
                                 sizeof(value), 0, nullptr, nullptr);
        test_error(err, "could not write to usvm memory with memcpy");

        err = mem->read(check);
        test_error(err, "could not read from usvm memory");

        test_assert_error(check == value,
                          "read value with memcpy does not match");

        // test writing via kernel
        value = genrand_int32(d);
        clMemWrapper in =
            clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           sizeof(cl_int), &value, &err);
        test_error(err, "could not create input buffer");

        err |= clSetKernelArg(kernel_CopyMemory, 0, sizeof(in), &in);
        err |= clSetKernelArgSVMPointer(kernel_CopyMemory, 1, mem->get_ptr());
        test_error(err, "could not set kernel arguments");

        size_t global_work_size = 1;
        err = clEnqueueNDRangeKernel(queue, kernel_CopyMemory, 1, nullptr,
                                     &global_work_size, nullptr, 0, nullptr,
                                     nullptr);
        test_error(err, "clEnqueueNDRangeKernel failed");

        err = clFinish(queue);
        test_error(err, "clFinish failed");

        err = mem->read(check);
        test_error(err, "could not read from usvm memory");

        test_assert_error(check == value,
                          "read value with kernel does not match");

        return CL_SUCCESS;
    }

    cl_int test_CL_SVM_CAPABILITY_DEVICE_ATOMIC_ACCESS_KHR(cl_uint typeIndex)
    {
        cl_int err;

        // setup
        auto mem = get_usvm_wrapper<cl_int>(typeIndex);
        err = mem->allocate(1);
        test_error(err, "could not allocate usvm memory");

        if (!kernel_AtomicIncrement)
        {
            err = createAtomicIncrementKernel();
            test_error(err, "could not create AtomicIncrement kernel");
        }

        err = mem->write(0);
        test_error(err, "could not write to usvm memory");

        err =
            clSetKernelArgSVMPointer(kernel_AtomicIncrement, 0, mem->get_ptr());
        test_error(err, "could not set kernel arguments");

        size_t global_work_size = num_elements;
        err = clEnqueueNDRangeKernel(queue, kernel_AtomicIncrement, 1, nullptr,
                                     &global_work_size, nullptr, 0, nullptr,
                                     nullptr);
        test_error(err, "clEnqueueNDRangeKernel failed");

        err = clFinish(queue);
        test_error(err, "clFinish failed");

        cl_int check;
        err = mem->read(check);
        test_error(err, "could not read from usvm memory");

        test_assert_error(check == num_elements,
                          "read value does not match expected value");

        return CL_SUCCESS;
    }

    cl_int test_CL_SVM_CAPABILITY_INDIRECT_ACCESS_KHR(cl_uint typeIndex)
    {
        cl_int err;

        // setup
        auto mem = get_usvm_wrapper<cl_int>(typeIndex);
        err = mem->allocate(1);
        test_error(err, "could not allocate usvm memory");

        if (!kernel_IndirectAccessRead)
        {
            err = createIndirectAccessKernel();
            test_error(err, "could not create IndirectAccess kernel");
        }

        // test reading indirectly
        cl_int value = genrand_int32(d);
        err = mem->write(value);
        test_error(err, "could not write to usvm memory");

        auto ptr = mem->get_ptr();
        clMemWrapper indirect =
            clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           sizeof(ptr), &ptr, &err);
        test_error(err, "could not create indirect buffer");

        clMemWrapper direct = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                             sizeof(cl_int), nullptr, &err);
        test_error(err, "could not create direct buffer");

        err |= clSetKernelArg(kernel_IndirectAccessRead, 0, sizeof(indirect),
                              &indirect);
        err |= clSetKernelArg(kernel_IndirectAccessRead, 1, sizeof(direct),
                              &direct);
        test_error(err, "could not set kernel arguments");

        cl_bool enable = CL_TRUE;
        err = clSetKernelExecInfo(kernel_IndirectAccessRead,
                                  CL_KERNEL_EXEC_INFO_SVM_INDIRECT_ACCESS_KHR,
                                  sizeof(enable), &enable);
        test_error(err, "could not enable indirect access");

        size_t global_work_size = 1;
        err = clEnqueueNDRangeKernel(queue, kernel_IndirectAccessRead, 1,
                                     nullptr, &global_work_size, nullptr, 0,
                                     nullptr, nullptr);
        test_error(err, "clEnqueueNDRangeKernel failed");

        err = clFinish(queue);
        test_error(err, "clFinish failed");

        cl_int check;
        err = clEnqueueReadBuffer(queue, direct, CL_TRUE, 0, sizeof(cl_int),
                                  &check, 0, nullptr, nullptr);
        test_error(err, "could not read direct buffer");

        test_assert_error(check == value, "read value does not match");

        // test writing indirectly
        value = genrand_int32(d);
        err = clEnqueueWriteBuffer(queue, direct, CL_TRUE, 0, sizeof(cl_int),
                                   &value, 0, nullptr, nullptr);
        test_error(err, "could not write to direct buffer");

        err |= clSetKernelArg(kernel_IndirectAccessWrite, 0, sizeof(indirect),
                              &indirect);
        err |= clSetKernelArg(kernel_IndirectAccessWrite, 1, sizeof(direct),
                              &direct);
        test_error(err, "could not set kernel arguments");

        err = clSetKernelExecInfo(kernel_IndirectAccessWrite,
                                  CL_KERNEL_EXEC_INFO_SVM_INDIRECT_ACCESS_KHR,
                                  sizeof(enable), &enable);
        test_error(err, "could not enable indirect access");

        err = clEnqueueNDRangeKernel(queue, kernel_IndirectAccessWrite, 1,
                                     nullptr, &global_work_size, nullptr, 0,
                                     nullptr, nullptr);
        test_error(err, "clEnqueueNDRangeKernel failed");

        err = clFinish(queue);
        test_error(err, "clFinish failed");

        err = mem->read(check);
        test_error(err, "could not read from usvm memory");

        test_assert_error(check == value, "read value does not match");

        return CL_SUCCESS;
    }

    cl_int run() override
    {
        cl_int err;
        for (cl_uint ti = 0; ti < static_cast<cl_uint>(deviceUSVMCaps.size());
             ti++)
        {
            const auto caps = deviceUSVMCaps[ti];
            log_info("   testing SVM type %u\n", ti);

            if (caps & CL_SVM_CAPABILITY_SINGLE_ADDRESS_SPACE_KHR)
            {
                log_info(
                    "     testing CL_SVM_CAPABILITY_SINGLE_ADDRESS_SPACE\n");
                err = test_CL_SVM_CAPABILITY_SINGLE_ADDRESS_SPACE_KHR(ti);
                test_error(err,
                           "CL_SVM_CAPABILITY_SINGLE_ADDRESS_SPACE failed");
            }
            // CL_SVM_CAPABILITY_SYSTEM_ALLOCATED_KHR
            // CL_SVM_CAPABILITY_DEVICE_OWNED_KHR
            if (caps & CL_SVM_CAPABILITY_DEVICE_UNASSOCIATED_KHR)
            {
                log_info(
                    "     testing CL_SVM_CAPABILITY_DEVICE_UNASSOCIATED\n");
                err = test_CL_SVM_CAPABILITY_DEVICE_UNASSOCIATED_KHR(ti);
                test_error(err, "CL_SVM_CAPABILITY_DEVICE_UNASSOCIATED failed");
            }
            // CL_SVM_CAPABILITY_CONTEXT_ACCESS_KHR
            // CL_SVM_CAPABILITY_HOST_OWNED_KHR
            if (caps & CL_SVM_CAPABILITY_HOST_READ_KHR)
            {
                log_info("     testing CL_SVM_CAPABILITY_HOST_READ\n");
                err = test_CL_SVM_CAPABILITY_HOST_READ_KHR(ti);
                test_error(err, "CL_SVM_CAPABILITY_HOST_READ failed");
            }
            if (caps & CL_SVM_CAPABILITY_HOST_WRITE_KHR)
            {
                log_info("     testing CL_SVM_CAPABILITY_HOST_WRITE\n");
                err = test_CL_SVM_CAPABILITY_HOST_WRITE_KHR(ti);
                test_error(err, "CL_SVM_CAPABILITY_HOST_WRITE failed");
            }
            if (caps & CL_SVM_CAPABILITY_HOST_MAP_KHR)
            {
                log_info("     testing CL_SVM_CAPABILITY_HOST_MAP\n");
                err = test_CL_SVM_CAPABILITY_HOST_MAP_KHR(ti);
                test_error(err, "CL_SVM_CAPABILITY_HOST_MAP failed");
            }
            if (caps & CL_SVM_CAPABILITY_DEVICE_READ_KHR)
            {
                log_info("     testing CL_SVM_CAPABILITY_DEVICE_READ\n");
                err = test_CL_SVM_CAPABILITY_DEVICE_READ_KHR(ti);
                test_error(err, "CL_SVM_CAPABILITY_DEVICE_READ failed");
            }
            if (caps & CL_SVM_CAPABILITY_DEVICE_WRITE_KHR)
            {
                log_info("     testing CL_SVM_CAPABILITY_DEVICE_WRITE\n");
                err = test_CL_SVM_CAPABILITY_DEVICE_READ_KHR(ti);
                test_error(err, "CL_SVM_CAPABILITY_DEVICE_READ failed");
            }
            if (caps & CL_SVM_CAPABILITY_DEVICE_ATOMIC_ACCESS_KHR)
            {
                log_info(
                    "     testing CL_SVM_CAPABILITY_DEVICE_ATOMIC_ACCESS\n");
                err = test_CL_SVM_CAPABILITY_DEVICE_ATOMIC_ACCESS_KHR(ti);
                test_error(err,
                           "CL_SVM_CAPABILITY_DEVICE_ATOMIC_ACCESS failed");
            }
            // CL_SVM_CAPABILITY_CONCURRENT_ACCESS_KHR
            // CL_SVM_CAPABILITY_CONCURRENT_ATOMIC_ACCESS_KHR
            if (caps & CL_SVM_CAPABILITY_INDIRECT_ACCESS_KHR)
            {
                log_info("     testing CL_SVM_CAPABILITY_INDIRECT_ACCESS\n");
                err = test_CL_SVM_CAPABILITY_INDIRECT_ACCESS_KHR(ti);
                test_error(err, "CL_SVM_CAPABILITY_INDIRECT_ACCESS failed");
            }
        }
        return CL_SUCCESS;
    }

    cl_int createStorePointerKernel()
    {
        cl_int err;

        const char* programString = R"(
            // workaround for error: kernel parameter cannot be declared as a pointer to a pointer
            struct s { const global int* ptr; }; 
            kernel void test_StorePointer(const global int* ptr, global struct s* dst)
            {
                dst[get_global_id(0)].ptr = ptr;
            }
        )";

        clProgramWrapper program;
        err =
            create_single_kernel_helper(context, &program, &kernel_StorePointer,
                                        1, &programString, "test_StorePointer");
        test_error(err, "could not create StorePointer kernel");

        return CL_SUCCESS;
    }

    cl_int createCopyMemoryKernel()
    {
        cl_int err;

        const char* programString = R"(
            kernel void test_CopyMemory(const global int* src, global int* dst)
            {
                dst[get_global_id(0)] = src[get_global_id(0)];
            }
        )";

        clProgramWrapper program;
        err = create_single_kernel_helper(context, &program, &kernel_CopyMemory,
                                          1, &programString, "test_CopyMemory");
        test_error(err, "could not create CopyMemory kernel");

        return CL_SUCCESS;
    }

    cl_int createAtomicIncrementKernel()
    {
        cl_int err;

        const char* programString = R"(
            kernel void test_AtomicIncrement(global int* ptr)
            {
                atomic_inc(ptr);
            }
        )";

        clProgramWrapper program;
        err = create_single_kernel_helper(
            context, &program, &kernel_AtomicIncrement, 1, &programString,
            "test_AtomicIncrement");
        test_error(err, "could not create AtomicIncrement kernel");

        return CL_SUCCESS;
    }

    cl_int createIndirectAccessKernel()
    {
        cl_int err;

        const char* programString = R"(
            struct s { const global int* ptr; };
            kernel void test_IndirectAccessRead(const global struct s* src, global int* dst)
            {
                dst[get_global_id(0)] = src->ptr[get_global_id(0)];
            }

            struct d { global int* ptr; };
            kernel void test_IndirectAccessWrite(global struct d* dst, const global int* src)
            {
                dst->ptr[get_global_id(0)] = src[get_global_id(0)];
            }
        )";

        clProgramWrapper program;
        err = create_single_kernel_helper(
            context, &program, &kernel_IndirectAccessRead, 1, &programString,
            "test_IndirectAccessRead");
        test_error(err, "could not create IndirectAccessRead kernel");

        kernel_IndirectAccessWrite =
            clCreateKernel(program, "test_IndirectAccessWrite", &err);
        test_error(err, "could not create IndirectAccessWrite kernel");

        return CL_SUCCESS;
    }

    clKernelWrapper kernel_StorePointer;
    clKernelWrapper kernel_CopyMemory;
    clKernelWrapper kernel_AtomicIncrement;
    clKernelWrapper kernel_IndirectAccessRead;
    clKernelWrapper kernel_IndirectAccessWrite;
};

REGISTER_TEST(unified_svm_capabilities)
{
    if (!is_extension_available(device, "cl_khr_unified_svm"))
    {
        log_info("cl_khr_unified_svm is not supported, skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    cl_int err;

    clContextWrapper contextWrapper;
    clCommandQueueWrapper queueWrapper;

    // For now: create a new context and queue.
    // If we switch to a new test executable and run the tests without
    // forceNoContextCreation then this can be removed, and we can just use the
    // context and the queue from the harness.
    if (context == nullptr)
    {
        contextWrapper =
            clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        test_error(err, "clCreateContext failed");
        context = contextWrapper;
    }

    if (queue == nullptr)
    {
        queueWrapper = clCreateCommandQueue(context, device, 0, &err);
        test_error(err, "clCreateCommandQueue failed");
        queue = queueWrapper;
    }

    UnifiedSVMCapabilities Test(context, device, queue, num_elements);
    err = Test.setup();
    test_error(err, "test setup failed");

    err = Test.run();
    test_error(err, "test failed");

    return TEST_PASS;
}
