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

#include "common.h"

#include <memory>

template <typename T> class USVMWrapper {
public:
    USVMWrapper(cl_context context_, cl_device_id device_,
                cl_command_queue queue_, cl_uint typeIndex_,
                cl_svm_capabilities_khr caps_,
                clSVMAllocWithPropertiesKHR_fn clSVMAllocWithPropertiesKHR_,
                clSVMFreeWithPropertiesKHR_fn clSVMFreeWithPropertiesKHR_,
                clGetSVMPointerInfoKHR_fn clGetSVMPointerInfoKHR_,
                clGetSVMSuggestedTypeIndexKHR_fn clGetSVMSuggestedTypeIndexKHR_)
        : context(context_), device(device_), queue(queue_),
          typeIndex(typeIndex_), caps(caps_),
          clSVMAllocWithPropertiesKHR(clSVMAllocWithPropertiesKHR_),
          clSVMFreeWithPropertiesKHR(clSVMFreeWithPropertiesKHR_),
          clGetSVMPointerInfoKHR(clGetSVMPointerInfoKHR_),
          clGetSVMSuggestedTypeIndexKHR(clGetSVMSuggestedTypeIndexKHR_)
    {}

    ~USVMWrapper() { free(); }

    cl_int allocate(const size_t count,
                    const std::vector<cl_svm_alloc_properties_khr> props_ = {})
    {
        if (data != nullptr)
        {
            free();
        }

        if (caps & CL_SVM_CAPABILITY_SYSTEM_ALLOCATED_KHR)
        {
            data = new T[count];
            test_assert_error_ret(data != nullptr, "Failed to allocate memory",
                                  CL_OUT_OF_RESOURCES);
        }
        else
        {
            std::vector<cl_svm_alloc_properties_khr> props = props_;
            if (!props.empty())
            {
                props.pop_back();
            }
            if (!(caps & CL_SVM_CAPABILITY_DEVICE_UNASSOCIATED_KHR))
            {
                props.push_back(CL_SVM_ALLOC_ASSOCIATED_DEVICE_HANDLE_KHR);
                props.push_back(
                    reinterpret_cast<cl_svm_alloc_properties_khr>(device));
            }
            if (!props.empty() || !props_.empty())
            {
                props.push_back(0);
            }

            cl_int err;
            data = (T*)clSVMAllocWithPropertiesKHR(
                context, props.empty() ? nullptr : props.data(), typeIndex,
                count * sizeof(T), &err);
            test_error(err, "clSVMAllocWithPropertiesKHR failed");
        }

        return CL_SUCCESS;
    }

    cl_int free()
    {
        if (data)
        {
            if (caps & CL_SVM_CAPABILITY_SYSTEM_ALLOCATED_KHR)
            {
                delete[] data;
            }
            else
            {
                cl_int err;
                err = clSVMFreeWithPropertiesKHR(context, nullptr, 0, data);
                test_error(err, "clSVMFreeWithPropertiesKHR failed");
            }

            data = nullptr;
        }

        return CL_SUCCESS;
    }

    cl_int write(const T* source, size_t size, size_t offset = 0)
    {
        if (data == nullptr)
        {
            return CL_INVALID_OPERATION;
        }

        cl_int err;

        if (caps & CL_SVM_CAPABILITY_HOST_WRITE_KHR)
        {
            std::copy(source, source + size, data + offset);
        }
        else if (caps & CL_SVM_CAPABILITY_HOST_MAP_KHR)
        {
            err = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_WRITE,
                                  data, size * sizeof(T), 0, nullptr, nullptr);
            test_error(err, "clEnqueueSVMMap failed");

            std::copy(source, source + size, data + offset);

            err = clEnqueueSVMUnmap(queue, data, 0, nullptr, nullptr);
            test_error(err, "clEnqueueSVMUnmap failed");
        }
        else if (caps & CL_SVM_CAPABILITY_DEVICE_WRITE_KHR)
        {
            err = clEnqueueSVMMemcpy(queue, CL_TRUE, data + offset, source,
                                     size * sizeof(T), 0, nullptr, nullptr);
            test_error(err, "clEnqueueSVMMemcpy failed");
        }
        else
        {
            log_error("Not sure how to write to SVM type index %u!\n", typeIndex);
            return CL_INVALID_OPERATION;
        }

        return CL_SUCCESS;
    }

    cl_int write(const std::vector<T>& source, size_t offset = 0)
    {
        return write(source.data(), source.size(), offset);
    }

    cl_int write(T source, size_t offset = 0)
    {
        return write(&source, sizeof(T), offset);
    }

    cl_int read(T* dst, size_t size, size_t offset = 0)
    {
        if (data == nullptr)
        {
            return CL_INVALID_OPERATION;
        }

        cl_int err;

        if (caps & CL_SVM_CAPABILITY_HOST_READ_KHR)
        {
            std::copy(data + offset, data + offset + size, dst);
        }
        else if (caps & CL_SVM_CAPABILITY_HOST_MAP_KHR)
        {
            err = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_READ, data,
                                  size * sizeof(T), 0, nullptr, nullptr);
            test_error(err, "clEnqueueSVMMap failed");

            std::copy(data + offset, data + offset + size, dst);

            err = clEnqueueSVMUnmap(queue, data, 0, nullptr, nullptr);
            test_error(err, "clEnqueueSVMUnmap failed");
        }
        else if (caps & CL_SVM_CAPABILITY_DEVICE_READ_KHR)
        {
            err = clEnqueueSVMMemcpy(queue, CL_TRUE, dst, data + offset,
                                     size * sizeof(T), 0, nullptr, nullptr);
            test_error(err, "clEnqueueSVMMemcpy failed");
        }
        else
        {
            log_error("Not sure how to read from SVM type index %u!\n", typeIndex);
            return CL_INVALID_OPERATION;
        }

        return CL_SUCCESS;
    }

    cl_int read(std::vector<T>& dst, size_t offset = 0)
    {
        return read(dst.data(), dst.size(), offset);
    }

    cl_int read(T& dst, size_t offset = 0)
    {
        return read(&dst, sizeof(T), offset);
    }

    T* get_ptr() { return data; }

private:
    cl_context context = nullptr;
    cl_device_id device = nullptr;
    cl_command_queue queue = nullptr;
    cl_uint typeIndex = 0;
    cl_svm_capabilities_khr caps = 0;

    clSVMAllocWithPropertiesKHR_fn clSVMAllocWithPropertiesKHR = nullptr;
    clSVMFreeWithPropertiesKHR_fn clSVMFreeWithPropertiesKHR = nullptr;
    clGetSVMPointerInfoKHR_fn clGetSVMPointerInfoKHR = nullptr;
    clGetSVMSuggestedTypeIndexKHR_fn clGetSVMSuggestedTypeIndexKHR = nullptr;

    T* data = nullptr;
};

struct UnifiedSVMBase
{
    UnifiedSVMBase(cl_context context_, cl_device_id device_,
                   cl_command_queue queue_)
        : context(context_), device(device_), queue(queue_)
    {}

    virtual cl_int setup()
    {
        cl_int err;

        size_t sz{};
        err = clGetDeviceInfo(device, CL_DEVICE_SVM_TYPE_CAPABILITIES_KHR, 0,
                              nullptr, &sz);
        test_error(
            err,
            "clGetDeviceInfo failed for CL_DEVICE_SVM_CAPABILITIES_KHR size");

        deviceUSVMCaps.resize(sz / sizeof(cl_svm_capabilities_khr));
        err = clGetDeviceInfo(device, CL_DEVICE_SVM_TYPE_CAPABILITIES_KHR, sz,
                              deviceUSVMCaps.data(), nullptr);
        test_error(
            err,
            "clGetDeviceInfo failed for CL_DEVICE_SVM_CAPABILITIES_KHR data");

        cl_platform_id platform{};
        err = clGetDeviceInfo(device, CL_DEVICE_PLATFORM,
                              sizeof(cl_platform_id), &platform, nullptr);
        test_error(err, "clGetDeviceInfo failed for CL_DEVICE_PLATFORM");

        clSVMAllocWithPropertiesKHR = (clSVMAllocWithPropertiesKHR_fn)
            clGetExtensionFunctionAddressForPlatform(
                platform, "clSVMAllocWithPropertiesKHR");
        test_assert_error_ret(clSVMAllocWithPropertiesKHR != nullptr,
                              "clSVMAllocWithPropertiesKHR not found",
                              CL_INVALID_OPERATION);

        clSVMFreeWithPropertiesKHR = (clSVMFreeWithPropertiesKHR_fn)
            clGetExtensionFunctionAddressForPlatform(
                platform, "clSVMFreeWithPropertiesKHR");
        test_assert_error_ret(clSVMFreeWithPropertiesKHR != nullptr,
                              "clSVMFreeWithPropertiesKHR not found",
                              CL_INVALID_OPERATION);

        clGetSVMPointerInfoKHR =
            (clGetSVMPointerInfoKHR_fn)clGetExtensionFunctionAddressForPlatform(
                platform, "clGetSVMPointerInfoKHR");
        test_assert_error_ret(clGetSVMPointerInfoKHR != nullptr,
                              "clGetSVMPointerInfoKHR not found",
                              CL_INVALID_OPERATION);

        clGetSVMSuggestedTypeIndexKHR = (clGetSVMSuggestedTypeIndexKHR_fn)
            clGetExtensionFunctionAddressForPlatform(
                platform, "clGetSVMSuggestedTypeIndexKHR");
        test_assert_error_ret(clGetSVMSuggestedTypeIndexKHR != nullptr,
                              "clGetSVMSuggestedTypeIndexKHR not found",
                              CL_INVALID_OPERATION);

        return CL_SUCCESS;
    }

    virtual cl_int run() = 0;

    template <typename T>
    std::unique_ptr<USVMWrapper<T>> get_usvm_wrapper(cl_uint typeIndex)
    {
        return std::unique_ptr<USVMWrapper<T>>(new USVMWrapper<T>(
            context, device, queue, typeIndex, deviceUSVMCaps[typeIndex],
            clSVMAllocWithPropertiesKHR, clSVMFreeWithPropertiesKHR,
            clGetSVMPointerInfoKHR, clGetSVMSuggestedTypeIndexKHR));
    }

    cl_context context = nullptr;
    cl_device_id device = nullptr;
    cl_command_queue queue = nullptr;

    std::vector<cl_svm_capabilities_khr> deviceUSVMCaps;

    clSVMAllocWithPropertiesKHR_fn clSVMAllocWithPropertiesKHR = nullptr;
    clSVMFreeWithPropertiesKHR_fn clSVMFreeWithPropertiesKHR = nullptr;
    clGetSVMPointerInfoKHR_fn clGetSVMPointerInfoKHR = nullptr;
    clGetSVMSuggestedTypeIndexKHR_fn clGetSVMSuggestedTypeIndexKHR = nullptr;
};
