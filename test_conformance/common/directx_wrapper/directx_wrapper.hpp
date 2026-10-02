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

#pragma once

#if D3D12_IS_SUPPORTED
#include <d3d12.h>
#endif
#if D3D11_IS_SUPPORTED
#include <cl/cl_d3d11.h>
#endif
#if D3D10_IS_SUPPORTED
#include <cl/cl_d3d10.h>
#endif

#include <vector>
#include <wrl/client.h>

using namespace Microsoft::WRL;

#if D3D12_IS_SUPPORTED
class DirectX12Wrapper {
public:
    DirectX12Wrapper();

    [[nodiscard]] ID3D12Device* getDXDevice() const;
    [[nodiscard]] ID3D12CommandQueue* getDXCommandQueue() const;
    [[nodiscard]] ID3D12CommandAllocator* getDXCommandAllocator() const;

protected:
    ComPtr<ID3D12Device> dx_device = nullptr;
    ComPtr<ID3D12CommandQueue> dx_command_queue = nullptr;
    ComPtr<ID3D12CommandAllocator> dx_command_allocator = nullptr;
};

class DirectX12FenceWrapper {
public:
    DirectX12FenceWrapper(ID3D12Device* dx_device);
    [[nodiscard]] ID3D12Fence* get() const { return dx_fence.Get(); }

private:
    ComPtr<ID3D12Fence> dx_fence = nullptr;
    ComPtr<ID3D12Device> dx_device = nullptr;
};
#endif

#if D3D11_IS_SUPPORTED
struct DirectX11Wrapper
{
    struct DeviceEntry
    {
        ComPtr<IDXGIAdapter> dx_adapter;
        ComPtr<ID3D11Device> dx_device;
    };

    DirectX11Wrapper();

    std::vector<DeviceEntry> devices;
};
#endif

#if D3D10_IS_SUPPORTED
struct DirectX10Wrapper
{
    struct DeviceEntry
    {
        ComPtr<IDXGIAdapter> dx_adapter;
        ComPtr<ID3D10Device> dx_device;
    };

    DirectX10Wrapper();

    std::vector<DeviceEntry> devices;
};
#endif
