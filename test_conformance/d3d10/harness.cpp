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
#include <vector>

#include <tchar.h>

/*
 * OpenCL state
 */

clGetDeviceIDsFromD3D10KHR_fn      clGetDeviceIDsFromD3D10KHR      = NULL;
clCreateFromD3D10BufferKHR_fn      clCreateFromD3D10BufferKHR      = NULL;
clCreateFromD3D10Texture2DKHR_fn   clCreateFromD3D10Texture2DKHR   = NULL;
clCreateFromD3D10Texture3DKHR_fn   clCreateFromD3D10Texture3DKHR   = NULL;
clEnqueueAcquireD3D10ObjectsKHR_fn clEnqueueAcquireD3D10ObjectsKHR = NULL;
clEnqueueReleaseD3D10ObjectsKHR_fn clEnqueueReleaseD3D10ObjectsKHR = NULL;

#define INITPFN(x) \
    x = (x ## _fn)clGetExtensionFunctionAddressForPlatform(platform, #x); NonTestRequire(x, "Failed to get function pointer for %s", #x);

void
HarnessD3D10_ExtensionCheck()
{
    bool extensionPresent = false;
    cl_int result = CL_SUCCESS;
    cl_platform_id platform = NULL;
    size_t set_size;

    HarnessD3D10_TestBegin("Extension query");

    result = clGetPlatformIDs(1, &platform, NULL);
    NonTestRequire(result == CL_SUCCESS, "Failed to get any platforms.");
    result = clGetPlatformInfo(platform, CL_PLATFORM_EXTENSIONS, 0, NULL, &set_size);
    NonTestRequire(result == CL_SUCCESS, "Failed to get size of extensions string.");
    std::vector<char> extensions(set_size);
    result = clGetPlatformInfo(platform, CL_PLATFORM_EXTENSIONS, extensions.size(), extensions.data(), NULL);
    NonTestRequire(result == CL_SUCCESS, "Failed to list extensions.");
    extensionPresent = strstr(extensions.data(), "cl_khr_d3d10_sharing") ? true : false;

    if (!extensionPresent) {
      // platform is required to report the extension only if all devices support it
      cl_uint devicesCount;
      result = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &devicesCount);
      NonTestRequire(result == CL_SUCCESS, "Failed to get devices count.");
      std::vector<cl_device_id> devices(devicesCount);
      result = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, &devices[0], NULL);
      NonTestRequire(result == CL_SUCCESS, "Failed to get devices count.");

      for (cl_uint i = 0; i < devicesCount; i++) {
        if (is_extension_available(devices[i], "cl_khr_d3d10_sharing")) {
          extensionPresent = true;
          break;
        }
      }
    }

    OSVERSIONINFO osvi;
    osvi.dwOSVersionInfoSize = sizeof(osvi);
    GetVersionEx(&osvi);
    if (osvi.dwMajorVersion <= 5)
    {
        TestRequire(!extensionPresent, "Extension should not be exported on Windows < 6");
    }
    else
    {
        TestRequire(extensionPresent, "Extension should be exported on Windows >= 6");
    }

Cleanup:
    HarnessD3D10_TestEnd();

    // early-out of the extension is not present
    if (!extensionPresent)
    {
        HarnessD3D10_TestStats();
    }
}

void
HarnessD3D10_Initialize(cl_platform_id platform)
{
    HarnessD3D10_ExtensionCheck();

    // extract function pointers for exported functions
    INITPFN(clGetDeviceIDsFromD3D10KHR);
    INITPFN(clCreateFromD3D10BufferKHR);
    INITPFN(clCreateFromD3D10Texture2DKHR);
    INITPFN(clCreateFromD3D10Texture3DKHR);
    INITPFN(clEnqueueAcquireD3D10ObjectsKHR);
    INITPFN(clEnqueueReleaseD3D10ObjectsKHR);
}

/*
 * Window management
 */

static IDXGISwapChain*  HarnessD3D10_pSwapChain = NULL;
static ID3D10Device*    HarnessD3D10_pDevice = NULL;
static HWND             HarnessD3D10_hWnd = NULL;

static LRESULT WINAPI HarnessD3D10_Proc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch(msg)
    {
        case WM_KEYDOWN:
            return 0;
            break;
        case WM_DESTROY:
            HarnessD3D10_hWnd = NULL;
            PostQuitMessage(0);
            return 0;
        case WM_PAINT:
            ValidateRect(hWnd, NULL);
            return 0;
    }
    return DefWindowProc(hWnd, msg, wParam, lParam);
}

static void HarnessD3D10_InteractiveLoop()
{
    MSG msg;
    while(PeekMessage(&msg,HarnessD3D10_hWnd,0,0,PM_REMOVE))
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
}

cl_int HarnessD3D10_CreateDevice(IDXGIAdapter* pAdapter, ID3D10Device **ppDevice)
{
    HRESULT hr = S_OK;
    unsigned int cuStatus = 1;

    *ppDevice = NULL;

    // create window
    static WNDCLASSEX wc =
    {
        sizeof(WNDCLASSEX),
        CS_CLASSDC,
        HarnessD3D10_Proc,
        0L,
        0L,
        GetModuleHandle(NULL),
        NULL,
        NULL,
        NULL,
        NULL,
        _T( "cl_khr_d3d10_sharing_conformance" ),
        NULL
    };
    RegisterClassEx(&wc);
    HarnessD3D10_hWnd = CreateWindow(
        _T( "cl_khr_d3d10_sharing_conformance" ),
        _T( "cl_khr_d3d10_sharing_conformance" ),
        WS_OVERLAPPEDWINDOW,
        0, 0, 256, 256,
        NULL,
        NULL,
        wc.hInstance,
        NULL);
    NonTestRequire(0 != HarnessD3D10_hWnd, "Failed to create window");

    ShowWindow(HarnessD3D10_hWnd,SW_SHOWDEFAULT);
    UpdateWindow(HarnessD3D10_hWnd);

    RECT rc;
    GetClientRect(HarnessD3D10_hWnd, &rc);
    UINT width = rc.right - rc.left;
    UINT height = rc.bottom - rc.top;

    // Create device and swapchain
    DXGI_SWAP_CHAIN_DESC sd;
    ZeroMemory(&sd, sizeof(sd));
    sd.BufferCount = 1;
    sd.BufferDesc.Width = width;
    sd.BufferDesc.Height = height;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator = 60;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = HarnessD3D10_hWnd;
    sd.SampleDesc.Count = 1;
    sd.SampleDesc.Quality = 0;
    sd.Windowed = TRUE;
    hr = D3D10CreateDeviceAndSwapChain(
        pAdapter,
        D3D10_DRIVER_TYPE_HARDWARE,
        NULL,
        0,
        D3D10_SDK_VERSION,
        &sd,
        &HarnessD3D10_pSwapChain,
        &HarnessD3D10_pDevice);

    if (FAILED(hr) ) {
        return CL_DEVICE_NOT_FOUND;
    }

    *ppDevice = HarnessD3D10_pDevice;
    return CL_SUCCESS;
}

void HarnessD3D10_DestroyDevice()
{
    HarnessD3D10_pSwapChain->Release();
    HarnessD3D10_pDevice->Release();

    if (HarnessD3D10_hWnd) DestroyWindow(HarnessD3D10_hWnd);
    HarnessD3D10_hWnd = 0;
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
} HarnessD3D10_testStats = {0};

void HarnessD3D10_TestBegin(const char* fmt, ...)
{
    va_list ap;

    va_start(ap, fmt);
    vsprintf(HarnessD3D10_testStats.currentTestName, fmt, ap);
    va_end(ap);

    TestPrint("[%s] ... ", HarnessD3D10_testStats.currentTestName);

    HarnessD3D10_testStats.inTest = 1;
    HarnessD3D10_testStats.currentTestPass = 1;
}

void HarnessD3D10_TestFail()
{
    if (HarnessD3D10_testStats.inTest)
    {
        HarnessD3D10_testStats.currentTestPass = 0;
    }
    else
    {
        ++HarnessD3D10_testStats.nonTestFailures;
    }
}

void HarnessD3D10_TestEnd()
{
    HarnessD3D10_testStats.inTest = 0;

    HarnessD3D10_testStats.testCount += 1;
    HarnessD3D10_testStats.passCount += HarnessD3D10_testStats.currentTestPass;

    TestPrint("%s\n",
        HarnessD3D10_testStats.currentTestPass ? "PASSED" : "FAILED");
}

void HarnessD3D10_TestStats()
{
    TestPrint("PASSED %d of %d tests.\n", HarnessD3D10_testStats.passCount, HarnessD3D10_testStats.testCount);
    if (HarnessD3D10_testStats.testCount > HarnessD3D10_testStats.passCount)
    {
        TestPrint("***FAILED***\n");
        exit(1);
    }
    else
    {
        TestPrint("&&&& cl_khr_d3d10_sharing test PASSED\n");
    }
    exit(0);
}

/*
 *
 * Helper function
 *
 */

cl_int HarnessD3D10_CreateKernelFromSource(
    cl_kernel *outKernel,
    cl_device_id device,
    cl_context context,
    const char *source,
    const char *entrypoint)
{
    cl_int status;
    cl_program program = NULL;
    cl_kernel kernel = NULL;

    // compile program
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



