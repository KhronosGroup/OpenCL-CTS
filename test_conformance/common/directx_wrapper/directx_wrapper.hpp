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
#include <stdexcept>

class DirectXWrapper
{
public:
    DirectXWrapper();
    ~DirectXWrapper();

    ID3D12Device* getDXDevice() const;
    ID3D12CommandQueue* getDXCommandQueue() const;
    ID3D12CommandAllocator* getDXCommandAllocator() const;

protected:
    ID3D12Device* dx_device = nullptr;
    ID3D12CommandQueue* dx_command_queue = nullptr;
    ID3D12CommandAllocator* dx_command_allocator = nullptr;
};

class DirectXFenceWrapper
{
public:
    DirectXFenceWrapper(ID3D12Device* dx_device);
    ~DirectXFenceWrapper();
    ID3D12Fence* operator *() const { return dx_fence; }

private:
    ID3D12Fence* dx_fence = nullptr;
    ID3D12Device* dx_device = nullptr;
};