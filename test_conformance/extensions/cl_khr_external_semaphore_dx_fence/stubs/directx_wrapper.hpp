#ifndef _directx_wrapper_hpp_
#define _directx_wrapper_hpp_

#include <vector>
#include <algorithm> // For std::find

#ifdef _WIN32
#include <windows.h>
#else
// Stub Windows types
typedef void* HANDLE;
typedef int BOOL;
typedef long HRESULT;

#define S_OK ((HRESULT)0L)
#define E_FAIL ((HRESULT)0x80004005L)
#define FAILED(hr) (((HRESULT)(hr)) < 0)
#define SUCCEEDED(hr) (((HRESULT)(hr)) >= 0)
#define GENERIC_ALL (0x10000000L)
#define EVENT_ALL_ACCESS (0x1F0003L)
#define INFINITE (0xFFFFFFFFL)

#endif

#ifdef IID_PPV_ARGS
#undef IID_PPV_ARGS
#endif
#define IID_PPV_ARGS(ppType) nullptr, reinterpret_cast<void**>(ppType)

#ifndef _WIN32
BOOL CloseHandle(HANDLE hObject);
HANDLE CreateEventEx(void* lpEventAttributes, const void* lpName,
                     unsigned long dwFlags, unsigned long dwDesiredAccess);
unsigned long WaitForSingleObject(HANDLE hHandle, unsigned long dwMilliseconds);
#endif

// Dummy D3D12 types
struct ID3D12Fence
{
    HRESULT Signal(unsigned long long Value);
    unsigned long Release();
    unsigned long long GetCompletedValue();
    HRESULT SetEventOnCompletion(unsigned long long Value, HANDLE hEvent);
};

struct ID3D12Device
{
    HRESULT CreateSharedHandle(void* pObject, void* pAttributes,
                               unsigned int Access, const void* Name,
                               HANDLE* pHandle);
    HRESULT OpenSharedHandle(HANDLE NTHandle, const void* riid, void** ppvObj);
};

struct ID3D12CommandQueue
{
    HRESULT Wait(ID3D12Fence* pFence, unsigned long long Value);
    HRESULT Signal(ID3D12Fence* pFence, unsigned long long Value);
};

class DirectX12Wrapper {
public:
    DirectX12Wrapper();
    ID3D12Device* getDXDevice() const;
    ID3D12CommandQueue* getDXCommandQueue() const;
    void* getDXCommandAllocator() const;
};

class DirectX12FenceWrapper {
public:
    DirectX12FenceWrapper(ID3D12Device* dx_device);
    ID3D12Fence* get() const;
};

#endif // _directx_wrapper_hpp_
