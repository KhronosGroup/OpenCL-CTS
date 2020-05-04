//
// Copyright (c) 2017 The Khronos Group Inc.
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
#define INITGUID
#include "harness.h"

#include <tchar.h>
#include <string>
#include <vector>

/*
 * OpenCL state
 */

clGetDeviceIDsFromD3D11KHR_fn      clGetDeviceIDsFromD3D11KHR      = NULL;
clCreateFromD3D11BufferKHR_fn      clCreateFromD3D11BufferKHR      = NULL;
clCreateFromD3D11Texture2DKHR_fn   clCreateFromD3D11Texture2DKHR   = NULL;
clCreateFromD3D11Texture3DKHR_fn   clCreateFromD3D11Texture3DKHR   = NULL;
clEnqueueAcquireD3D11ObjectsKHR_fn clEnqueueAcquireD3D11ObjectsKHR = NULL;
clEnqueueReleaseD3D11ObjectsKHR_fn clEnqueueReleaseD3D11ObjectsKHR = NULL;

#define INITPFN(x) \
    x = (x ## _fn)clGetExtensionFunctionAddressForPlatform(platform, #x); NonTestRequire(x, "Failed to get function pointer for %s", #x);

void
HarnessD3D11_ExtensionCheck()
{
    cl_int result = CL_SUCCESS;
    cl_platform_id platform = NULL;

    HarnessD3D11_TestBegin("Extension query");

    bool platform_d3d11 = false; // Does platform support the extension?
    {
        std::vector< char > buffer;
        size_t size = 0;
        result = clGetPlatformIDs( 1, &platform, NULL );
            NonTestRequire( result == CL_SUCCESS, "Failed to get any platforms." );
        result = clGetPlatformInfo( platform, CL_PLATFORM_EXTENSIONS, 0, NULL, & size );
            NonTestRequire( result == CL_SUCCESS, "Failed to get size of extension string." );
        buffer.resize( size );
        result = clGetPlatformInfo( platform, CL_PLATFORM_EXTENSIONS, buffer.size(), & buffer.front(), & size );
            NonTestRequire( result == CL_SUCCESS, "Failed to get extension string." );
        std::string extensions = std::string( " " ) + & buffer.front() + " ";
        platform_d3d11 = ( extensions.find( " cl_khr_d3d11_sharing " ) != std::string::npos );
    }

    // Platform is required to report the extension only if all devices support it,
    // so let us iterate through all the devices and count devices supporting the extension.

    // Get list of all devices.
    std::vector< cl_device_id > devices;
    cl_uint num_devices = 0;
    result = clGetDeviceIDs( platform, CL_DEVICE_TYPE_ALL, 0, NULL, & num_devices );
        NonTestRequire( result == CL_SUCCESS, "Failed to get number of devices." );
    devices.resize( num_devices );
    result = clGetDeviceIDs( platform, CL_DEVICE_TYPE_ALL, devices.size(), & devices.front(), & num_devices );
        NonTestRequire( result == CL_SUCCESS, "Failed to get list of device ids." );
        NonTestRequire( num_devices == devices.size(), "Failed to get list of device ids." );

    // Iterate through the devices and count devices supporting the extension.
    cl_uint num_devices_d3d11 = 0; // Number of devices supporting cl_khr_d3d11_sharing.
    for ( cl_uint i = 0; i < devices.size(); ++ i )
    {
        if (is_extension_available(devices[i], "cl_khr_d3d11_sharing"))
        {
            ++ num_devices_d3d11;
        }
    }

    OSVERSIONINFO osvi;
    osvi.dwOSVersionInfoSize = sizeof(osvi);
    GetVersionEx(&osvi);
    if (osvi.dwMajorVersion <= 5)
    {
        // Neither platform nor devices should declare support.
        TestRequire( ! platform_d3d11, "Platform should not declare extension on Windows < 6" );
        TestRequire( num_devices_d3d11 == 0, "Devices should not declare extension on Windows < 6" );
    }
    else
    {
        if ( num_devices_d3d11 == num_devices )
        {
            // All the devices declare support, so platform must declare support as well.
            TestRequire( platform_d3d11, "Extension should be exported on Windows >= 6" );
        }
        else
        {
            // Not all the devices support th eextension => platform should not declare it.
            TestRequire( ! platform_d3d11, "Extension should not be exported on Windows >= 6" );
        }
    }

Cleanup:
    HarnessD3D11_TestEnd();

    // early-out of the extension is not present
    if ( num_devices_d3d11 == 0 )
    {
        HarnessD3D11_TestStats();
    }
}

void
HarnessD3D11_Initialize(cl_platform_id platform)
{
    HarnessD3D11_ExtensionCheck();

    // extract function pointers for exported functions
    INITPFN(clGetDeviceIDsFromD3D11KHR);
    INITPFN(clCreateFromD3D11BufferKHR);
    INITPFN(clCreateFromD3D11Texture2DKHR);
    INITPFN(clCreateFromD3D11Texture3DKHR);
    INITPFN(clEnqueueAcquireD3D11ObjectsKHR);
    INITPFN(clEnqueueReleaseD3D11ObjectsKHR);
}

/*
 * Window management
 */

static IDXGISwapChain*       HarnessD3D11_pSwapChain = NULL;
static ID3D11Device*         HarnessD3D11_pDevice = NULL;
static ID3D11DeviceContext*  HarnessD3D11_pDC = NULL;
static HWND                  HarnessD3D11_hWnd = NULL;

static LRESULT WINAPI HarnessD3D11_Proc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch(msg)
    {
        case WM_KEYDOWN:
            return 0;
            break;
        case WM_DESTROY:
            HarnessD3D11_hWnd = NULL;
            PostQuitMessage(0);
            return 0;
        case WM_PAINT:
            ValidateRect(hWnd, NULL);
            return 0;
    }
    return DefWindowProc(hWnd, msg, wParam, lParam);
}

static void HarnessD3D11_InteractiveLoop()
{
    MSG msg;
    while(PeekMessage(&msg,HarnessD3D11_hWnd,0,0,PM_REMOVE))
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
}

cl_int HarnessD3D11_CreateDevice(
    IDXGIAdapter* pAdapter,
    ID3D11Device **ppDevice,
    ID3D11DeviceContext** ppDC)
{
    HRESULT hr = S_OK;
    unsigned int cuStatus = 1;

    *ppDevice = NULL;

    // create window
    static WNDCLASSEX wc =
    {
        sizeof(WNDCLASSEX),
        CS_CLASSDC,
        HarnessD3D11_Proc,
        0L,
        0L,
        GetModuleHandle(NULL),
        NULL,
        NULL,
        NULL,
        NULL,
        _T( "cl_khr_d3d11_sharing_conformance" ),
        NULL
    };
    RegisterClassEx(&wc);
    HarnessD3D11_hWnd = CreateWindow(
        _T( "cl_khr_d3d11_sharing_conformance" ),
        _T( "cl_khr_d3d11_sharing_conformance" ),
        WS_OVERLAPPEDWINDOW,
        0, 0, 256, 256,
        NULL,
        NULL,
        wc.hInstance,
        NULL);
    NonTestRequire(0 != HarnessD3D11_hWnd, "Failed to create window");

    ShowWindow(HarnessD3D11_hWnd,SW_SHOWDEFAULT);
    UpdateWindow(HarnessD3D11_hWnd);

    RECT rc;
    GetClientRect(HarnessD3D11_hWnd, &rc);
    UINT width = rc.right - rc.left;
    UINT height = rc.bottom - rc.top;

    // Create device and swapchain
    DXGI_SWAP_CHAIN_DESC sd;
    ZeroMemory( &sd, sizeof(sd) );
    sd.BufferCount = 1;
    sd.BufferDesc.Width = width;
    sd.BufferDesc.Height = height;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator = 60;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = HarnessD3D11_hWnd;
    sd.SampleDesc.Count = 1;
    sd.SampleDesc.Quality = 0;
    sd.Windowed = TRUE;
    D3D_FEATURE_LEVEL requestedFeatureLevels[] = {D3D_FEATURE_LEVEL_10_0};
    D3D_FEATURE_LEVEL featureLevel = D3D_FEATURE_LEVEL_11_0;
    hr = D3D11CreateDeviceAndSwapChain(
        NULL, // pAdapter,
        D3D_DRIVER_TYPE_HARDWARE,
        NULL,
        0,
        requestedFeatureLevels,
        1,
        D3D11_SDK_VERSION,
        &sd,
        &HarnessD3D11_pSwapChain,
        &HarnessD3D11_pDevice,
        &featureLevel,
        &HarnessD3D11_pDC);
    if (FAILED(hr) ) {
        return CL_DEVICE_NOT_FOUND;
    }

    *ppDevice = HarnessD3D11_pDevice;
    *ppDC = HarnessD3D11_pDC;
    return CL_SUCCESS;
}

void HarnessD3D11_DestroyDevice()
{
    HarnessD3D11_pSwapChain->Release();
    HarnessD3D11_pDevice->Release();
    HarnessD3D11_pDC->Release();

    if (HarnessD3D11_hWnd) DestroyWindow(HarnessD3D11_hWnd);
    HarnessD3D11_hWnd = 0;
}

/*
 *
 * texture formats
 *
 */

#define ADD_TEXTURE_FORMAT(x,y,z,a,b,g) { x, y, z, a*b/8, g, #x, #y, #z, }
TextureFormat formats[] =
{
    ADD_TEXTURE_FORMAT( DXGI_FORMAT_R32G32B32A32_FLOAT , CL_RGBA , CL_FLOAT           , 32, 4, TextureFormat::GENERIC_FLOAT ),
    ADD_TEXTURE_FORMAT( DXGI_FORMAT_R32G32B32A32_UINT  , CL_RGBA , CL_UNSIGNED_INT32  , 32, 4, TextureFormat::GENERIC_UINT  ),
    ADD_TEXTURE_FORMAT( DXGI_FORMAT_R32G32B32A32_SINT  , CL_RGBA , CL_SIGNED_INT32    , 32, 4, TextureFormat::GENERIC_SINT  ),
    ADD_TEXTURE_FORMAT( DXGI_FORMAT_R16G16B16A16_FLOAT , CL_RGBA , CL_HALF_FLOAT      , 16, 4, TextureFormat::GENERIC_FLOAT ),
    ADD_TEXTURE_FORMAT( DXGI_FORMAT_R16G16B16A16_UNORM , CL_RGBA , CL_UNORM_INT16     , 16, 4, TextureFormat::GENERIC_FLOAT ),
    ADD_TEXTURE_FORMAT( DXGI_FORMAT_R16G16B16A16_UINT  , CL_RGBA , CL_UNSIGNED_INT16  , 16, 4, TextureFormat::GENERIC_UINT  ),
    ADD_TEXTURE_FORMAT( DXGI_FORMAT_R16G16B16A16_SNORM , CL_RGBA , CL_SNORM_INT16     , 16, 4, TextureFormat::GENERIC_FLOAT ),
    ADD_TEXTURE_FORMAT( DXGI_FORMAT_R16G16B16A16_SINT  , CL_RGBA , CL_SIGNED_INT16    , 16, 4, TextureFormat::GENERIC_SINT  ),
    ADD_TEXTURE_FORMAT( DXGI_FORMAT_R8G8B8A8_UNORM     , CL_RGBA , CL_UNORM_INT8      ,  8, 4, TextureFormat::GENERIC_FLOAT ),
    ADD_TEXTURE_FORMAT( DXGI_FORMAT_R8G8B8A8_UINT      , CL_RGBA , CL_UNSIGNED_INT8   ,  8, 4, TextureFormat::GENERIC_UINT  ),
    ADD_TEXTURE_FORMAT( DXGI_FORMAT_R8G8B8A8_SNORM     , CL_RGBA , CL_SNORM_INT8      ,  8, 4, TextureFormat::GENERIC_FLOAT ),
    ADD_TEXTURE_FORMAT( DXGI_FORMAT_R8G8B8A8_SINT      , CL_RGBA , CL_SIGNED_INT8     ,  8, 4, TextureFormat::GENERIC_SINT  ),

    ADD_TEXTURE_FORMAT( DXGI_FORMAT_R32G32_FLOAT       , CL_RG   , CL_FLOAT           , 32, 2, TextureFormat::GENERIC_FLOAT ),
    ADD_TEXTURE_FORMAT( DXGI_FORMAT_R32G32_UINT        , CL_RG   , CL_UNSIGNED_INT32  , 32, 2, TextureFormat::GENERIC_UINT  ),
    ADD_TEXTURE_FORMAT( DXGI_FORMAT_R32G32_SINT        , CL_RG   , CL_SIGNED_INT32    , 32, 2, TextureFormat::GENERIC_SINT  ),
    ADD_TEXTURE_FORMAT( DXGI_FORMAT_R16G16_FLOAT       , CL_RG   , CL_HALF_FLOAT      , 16, 2, TextureFormat::GENERIC_FLOAT ),
    ADD_TEXTURE_FORMAT( DXGI_FORMAT_R16G16_UNORM       , CL_RG   , CL_UNORM_INT16     , 16, 2, TextureFormat::GENERIC_FLOAT ),
    ADD_TEXTURE_FORMAT( DXGI_FORMAT_R16G16_UINT        , CL_RG   , CL_UNSIGNED_INT16  , 16, 2, TextureFormat::GENERIC_UINT  ),
    ADD_TEXTURE_FORMAT( DXGI_FORMAT_R16G16_SNORM       , CL_RG   , CL_SNORM_INT16     , 16, 2, TextureFormat::GENERIC_FLOAT ),
    ADD_TEXTURE_FORMAT( DXGI_FORMAT_R16G16_SINT        , CL_RG   , CL_SIGNED_INT16    , 16, 2, TextureFormat::GENERIC_SINT  ),
    ADD_TEXTURE_FORMAT( DXGI_FORMAT_R8G8_UNORM         , CL_RG   , CL_UNORM_INT8      ,  8, 2, TextureFormat::GENERIC_FLOAT ),
    ADD_TEXTURE_FORMAT( DXGI_FORMAT_R8G8_UINT          , CL_RG   , CL_UNSIGNED_INT8   ,  8, 2, TextureFormat::GENERIC_UINT  ),
    ADD_TEXTURE_FORMAT( DXGI_FORMAT_R8G8_SNORM         , CL_RG   , CL_SNORM_INT8      ,  8, 2, TextureFormat::GENERIC_FLOAT ),
    ADD_TEXTURE_FORMAT( DXGI_FORMAT_R8G8_SINT          , CL_RG   , CL_SIGNED_INT8     ,  8, 2, TextureFormat::GENERIC_SINT  ),

    ADD_TEXTURE_FORMAT( DXGI_FORMAT_R32_FLOAT          , CL_R    , CL_FLOAT           , 32, 1, TextureFormat::GENERIC_FLOAT ),
    ADD_TEXTURE_FORMAT( DXGI_FORMAT_R32_UINT           , CL_R    , CL_UNSIGNED_INT32  , 32, 1, TextureFormat::GENERIC_UINT  ),
    ADD_TEXTURE_FORMAT( DXGI_FORMAT_R32_SINT           , CL_R    , CL_SIGNED_INT32    , 32, 1, TextureFormat::GENERIC_SINT  ),
    ADD_TEXTURE_FORMAT( DXGI_FORMAT_R16_FLOAT          , CL_R    , CL_HALF_FLOAT      , 16, 1, TextureFormat::GENERIC_FLOAT ),
    ADD_TEXTURE_FORMAT( DXGI_FORMAT_R16_UNORM          , CL_R    , CL_UNORM_INT16     , 16, 1, TextureFormat::GENERIC_FLOAT ),
    ADD_TEXTURE_FORMAT( DXGI_FORMAT_R16_UINT           , CL_R    , CL_UNSIGNED_INT16  , 16, 1, TextureFormat::GENERIC_UINT  ),
    ADD_TEXTURE_FORMAT( DXGI_FORMAT_R16_SNORM          , CL_R    , CL_SNORM_INT16     , 16, 1, TextureFormat::GENERIC_FLOAT ),
    ADD_TEXTURE_FORMAT( DXGI_FORMAT_R16_SINT           , CL_R    , CL_SIGNED_INT16    , 16, 1, TextureFormat::GENERIC_SINT  ),
    ADD_TEXTURE_FORMAT( DXGI_FORMAT_R8_UNORM           , CL_R    , CL_UNORM_INT8      ,  8, 1, TextureFormat::GENERIC_FLOAT ),
    ADD_TEXTURE_FORMAT( DXGI_FORMAT_R8_UINT            , CL_R    , CL_UNSIGNED_INT8   ,  8, 1, TextureFormat::GENERIC_UINT  ),
    ADD_TEXTURE_FORMAT( DXGI_FORMAT_R8_SNORM           , CL_R    , CL_SNORM_INT8      ,  8, 1, TextureFormat::GENERIC_FLOAT ),
    ADD_TEXTURE_FORMAT( DXGI_FORMAT_R8_SINT            , CL_R    , CL_SIGNED_INT8     ,  8, 1, TextureFormat::GENERIC_SINT  ),
};
UINT formatCount = sizeof(formats)/sizeof(formats[0]);

/*
 *
 * Logging and error reporting
 *
 */

static struct
{
    cl_int testCount;
    cl_int passCount;

    cl_int nonTestFailures;
    cl_int inTest;
    cl_int currentTestPass;

    char currentTestName[1024];
} HarnessD3D11_testStats = {0};

void HarnessD3D11_TestBegin(const char* fmt, ...)
{
    va_list ap;

    va_start(ap, fmt);
    vsprintf(HarnessD3D11_testStats.currentTestName, fmt, ap);
    va_end(ap);

    printf("[%s] ... ", HarnessD3D11_testStats.currentTestName);

    HarnessD3D11_testStats.inTest = 1;
    HarnessD3D11_testStats.currentTestPass = 1;
}

void HarnessD3D11_TestFail()
{
    if (HarnessD3D11_testStats.inTest)
    {
        HarnessD3D11_testStats.currentTestPass = 0;
    }
    else
    {
        ++HarnessD3D11_testStats.nonTestFailures;
    }
}

void HarnessD3D11_TestEnd()
{
    HarnessD3D11_testStats.inTest = 0;

    HarnessD3D11_testStats.testCount += 1;
    HarnessD3D11_testStats.passCount += HarnessD3D11_testStats.currentTestPass;

    TestPrint("%s\n",
        HarnessD3D11_testStats.currentTestPass ? "PASSED" : "FAILED");
}

void HarnessD3D11_TestStats()
{
    TestPrint("PASSED %d of %d tests.\n", HarnessD3D11_testStats.passCount, HarnessD3D11_testStats.testCount);
    if (HarnessD3D11_testStats.testCount > HarnessD3D11_testStats.passCount)
    {
        TestPrint("***FAILED***\n");
        exit(1);
    }
    else
    {
        TestPrint("&&&& cl_khr_d3d11_sharing test PASSED\n");
    }
    exit(0);
}

/*
 *
 * Helper function
 *
 */

cl_int HarnessD3D11_CreateKernelFromSource(
    cl_kernel *outKernel,
    cl_device_id device,
    cl_context context,
    const char *source,
    const char *entrypoint)
{
    cl_int status;
    cl_kernel kernel = NULL;

    // compile program
    cl_program program = NULL;
    {
        const char *sourceTexts[] = {source};
        size_t sourceLengths[] = {strlen(source) };

        status = create_single_kernel_helper_create_program(context, &program, 1, &sourceTexts[0]);
        TestRequire(
            CL_SUCCESS == status,
            "clCreateProgramWithSource failed");
    }
    status = clBuildProgram(
        program,
        0,
        NULL,
        NULL,
        NULL,
        NULL);
    if (CL_SUCCESS != status)
    {
        char log[2048] = {0};
        status = clGetProgramBuildInfo(
            program,
            device,
            CL_PROGRAM_BUILD_LOG,
            sizeof(log),
            log,
            NULL);
        TestPrint("error: %s\n", log);
        TestRequire(
            CL_SUCCESS == status,
            "Compilation error log:\n%s\n", log);
    }

    kernel = clCreateKernel(
        program,
        entrypoint,
        &status);
    TestRequire(
        CL_SUCCESS == status,
        "clCreateKernel failed");

    clReleaseProgram(program);
    *outKernel = kernel;

Cleanup:

    return CL_SUCCESS;
}



