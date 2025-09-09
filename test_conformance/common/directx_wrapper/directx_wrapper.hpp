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

#include <d3d12.h>
#include <wrl/client.h>
#include <stdexcept>

using namespace Microsoft::WRL;

class DirectXWrapper {
public:
    DirectXWrapper();

    ID3D12Device* getDXDevice() const;
    ID3D12CommandQueue* getDXCommandQueue() const;
    ID3D12CommandAllocator* getDXCommandAllocator() const;

protected:
    ComPtr<ID3D12Device> dx_device = nullptr;
    ComPtr<ID3D12CommandQueue> dx_command_queue = nullptr;
    ComPtr<ID3D12CommandAllocator> dx_command_allocator = nullptr;
};

class DirectXFenceWrapper {
public:
    DirectXFenceWrapper(ID3D12Device* dx_device);
    ID3D12Fence* operator*() const { return dx_fence.Get(); }

private:
    ComPtr<ID3D12Fence> dx_fence = nullptr;
    ComPtr<ID3D12Device> dx_device = nullptr;
};