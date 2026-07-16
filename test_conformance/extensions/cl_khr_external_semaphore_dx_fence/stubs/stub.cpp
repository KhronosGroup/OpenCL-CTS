#include "directx_wrapper.hpp"
#include "harness/errorHelpers.h"

#ifndef _WIN32
BOOL CloseHandle(HANDLE hObject)
{
    log_info("Stub CloseHandle called\n");
    return 1;
}

HANDLE CreateEventEx(void* lpEventAttributes, const void* lpName,
                     unsigned long dwFlags, unsigned long dwDesiredAccess)
{
    log_info("Stub CreateEventEx called\n");
    return nullptr;
}

unsigned long WaitForSingleObject(HANDLE hHandle, unsigned long dwMilliseconds)
{
    log_info("Stub WaitForSingleObject called\n");
    return 0;
}
#endif

HRESULT ID3D12Fence::Signal(unsigned long long Value)
{
    log_error("Stub ID3D12Fence::Signal called\n");
    return E_FAIL;
}

unsigned long ID3D12Fence::Release()
{
    log_info("Stub ID3D12Fence::Release called\n");
    return 0;
}

unsigned long long ID3D12Fence::GetCompletedValue()
{
    log_error("Stub ID3D12Fence::GetCompletedValue called\n");
    return 0;
}

HRESULT ID3D12Fence::SetEventOnCompletion(unsigned long long Value,
                                          HANDLE hEvent)
{
    log_error("Stub ID3D12Fence::SetEventOnCompletion called\n");
    return E_FAIL;
}

HRESULT ID3D12Device::CreateSharedHandle(void* pObject, void* pAttributes,
                                         unsigned int Access, const void* Name,
                                         HANDLE* pHandle)
{
    log_error("Stub ID3D12Device::CreateSharedHandle called\n");
    return E_FAIL;
}

HRESULT ID3D12Device::OpenSharedHandle(HANDLE NTHandle, const void* riid,
                                       void** ppvObj)
{
    log_error("Stub ID3D12Device::OpenSharedHandle called\n");
    return E_FAIL;
}

HRESULT ID3D12CommandQueue::Wait(ID3D12Fence* pFence, unsigned long long Value)
{
    log_error("Stub ID3D12CommandQueue::Wait called\n");
    return E_FAIL;
}

HRESULT ID3D12CommandQueue::Signal(ID3D12Fence* pFence,
                                   unsigned long long Value)
{
    log_error("Stub ID3D12CommandQueue::Signal called\n");
    return E_FAIL;
}


DirectX12Wrapper::DirectX12Wrapper()
{
    log_info("Stub DirectX12Wrapper constructor called\n");
}

ID3D12Device* DirectX12Wrapper::getDXDevice() const
{
    log_error("Stub DirectX12Wrapper::getDXDevice called\n");
    static ID3D12Device dummy_device;
    return &dummy_device;
}

ID3D12CommandQueue* DirectX12Wrapper::getDXCommandQueue() const
{
    log_error("Stub DirectX12Wrapper::getDXCommandQueue called\n");
    static ID3D12CommandQueue dummy_queue;
    return &dummy_queue;
}

void* DirectX12Wrapper::getDXCommandAllocator() const
{
    log_error("Stub DirectX12Wrapper::getDXCommandAllocator called\n");
    return nullptr;
}

DirectX12FenceWrapper::DirectX12FenceWrapper(ID3D12Device* dx_device)
{
    log_error("Stub DirectX12FenceWrapper constructor called\n");
}

ID3D12Fence* DirectX12FenceWrapper::get() const
{
    log_error("Stub DirectX12FenceWrapper::get called\n");
    return nullptr;
}
