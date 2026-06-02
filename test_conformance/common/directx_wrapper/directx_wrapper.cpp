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

#include <stdexcept>

#include "directx_wrapper.hpp"

#if D3D12_IS_SUPPORTED
DirectX12Wrapper::DirectX12Wrapper()
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

ID3D12Device* DirectX12Wrapper::getDXDevice() const { return dx_device.Get(); }

ID3D12CommandQueue* DirectX12Wrapper::getDXCommandQueue() const
{
    return dx_command_queue.Get();
}
ID3D12CommandAllocator* DirectX12Wrapper::getDXCommandAllocator() const
{
    return dx_command_allocator.Get();
}

DirectX12FenceWrapper::DirectX12FenceWrapper(ID3D12Device* dx_device)
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
#endif

#if D3D11_IS_SUPPORTED
DirectX11Wrapper::DirectX11Wrapper()
{
    ComPtr<IDXGIFactory> factory;
    HRESULT hr = CreateDXGIFactory(IID_PPV_ARGS(factory.GetAddressOf()));
    if (FAILED(hr))
    {
        throw std::runtime_error("Failed to create DXGI factory");
    }

    UINT i = 0;
    ComPtr<IDXGIAdapter> adapter;
    while (factory->EnumAdapters(i, adapter.GetAddressOf())
           != DXGI_ERROR_NOT_FOUND)
    {
        ++i;

        ComPtr<ID3D11Device> device;
        hr = D3D11CreateDevice(adapter.Get(), D3D_DRIVER_TYPE_HARDWARE, nullptr,
                               0, nullptr, 0, D3D11_SDK_VERSION,
                               device.GetAddressOf(), nullptr, nullptr);
        if (FAILED(hr))
        {
            throw std::runtime_error("Failed to create DirectX 10 device");
        }

        devices.push_back({ adapter, device });
    }
}
#endif

#if D3D10_IS_SUPPORTED
DirectX10Wrapper::DirectX10Wrapper()
{
    ComPtr<IDXGIFactory> factory;
    HRESULT hr = CreateDXGIFactory(IID_PPV_ARGS(factory.GetAddressOf()));
    if (FAILED(hr))
    {
        throw std::runtime_error("Failed to create DXGI factory");
    }

    UINT i = 0;
    ComPtr<IDXGIAdapter> adapter;
    while (factory->EnumAdapters(i, adapter.GetAddressOf())
           != DXGI_ERROR_NOT_FOUND)
    {
        ++i;

        ComPtr<ID3D10Device> device;
        hr = D3D10CreateDevice(adapter.Get(), D3D10_DRIVER_TYPE_HARDWARE,
                               nullptr, 0, D3D10_SDK_VERSION,
                               device.GetAddressOf());
        if (FAILED(hr))
        {
            throw std::runtime_error("Failed to create DirectX 10 device");
        }

        devices.push_back({ adapter, device });
    }
}
#endif
