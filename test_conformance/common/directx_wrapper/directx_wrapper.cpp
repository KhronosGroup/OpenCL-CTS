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

#include "directx_wrapper.hpp"

DirectXWrapper::DirectXWrapper()
{

    HRESULT hr = D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0,
                                   IID_PPV_ARGS(&dx_device));
    if (FAILED(hr))
    {
        throw std::runtime_error("Failed to create DirectX 12 device");
    }

    D3D12_COMMAND_QUEUE_DESC desc{};
    desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    hr = dx_device->CreateCommandQueue(&desc, IID_PPV_ARGS(&dx_command_queue));
    if (FAILED(hr))
    {
        throw std::runtime_error("Failed to create DirectX 12 command queue");
    }

    hr = dx_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                           IID_PPV_ARGS(&dx_command_allocator));
    if (FAILED(hr))
    {
        throw std::runtime_error(
            "Failed to create DirectX 12 command allocator");
    }
}

DirectXWrapper::~DirectXWrapper()
{
    if (dx_command_allocator)
    {
        dx_command_allocator->Release();
    }
    if (dx_command_queue)
    {
        dx_command_queue->Release();
    }
    if (dx_device)
    {
        dx_device->Release();
    }
}

ID3D12Device* DirectXWrapper::getDXDevice() const { return dx_device; }

ID3D12CommandQueue* DirectXWrapper::getDXCommandQueue() const
{
    return dx_command_queue;
}
ID3D12CommandAllocator* DirectXWrapper::getDXCommandAllocator() const
{
    return dx_command_allocator;
}

DirectXFenceWrapper::DirectXFenceWrapper(ID3D12Device* dx_device)
    : dx_device(dx_device)
{
    if (!dx_device)
    {
        throw std::runtime_error("ID3D12Device is not valid");
    }
    const HRESULT hr = dx_device->CreateFence(0, D3D12_FENCE_FLAG_SHARED,
                                              IID_PPV_ARGS(&dx_fence));
    if (FAILED(hr))
    {
        throw std::runtime_error("Failed to create the DirectX fence");
    }
}

DirectXFenceWrapper::~DirectXFenceWrapper()
{
    if (dx_fence)
    {
        dx_fence->Release();
    }
}
