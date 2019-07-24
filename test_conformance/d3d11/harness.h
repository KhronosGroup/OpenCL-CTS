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
#ifndef _HARNESS_H_
#define _HARNESS_H_

#define _CRT_SECURE_NO_WARNINGS

#if defined (__MINGW32__)
#include <rpcsal.h>
typedef unsigned char UINT8;
#define __out
#define __in
#define __inout
#define __out_bcount(size)
#define __out_bcount_opt(size)
#define __in_opt
#define __in_ecount(size)
#define __in_ecount_opt(size)
#define __out_opt
#define __out_ecount(size)
#define __out_ecount_opt(size)
#define __in_bcount_opt(size)
#define __inout_opt
#endif

#include <CL/cl.h>
#include <CL/cl_platform.h>
#include <CL/cl_d3d11.h>
#include <stdio.h>
#include "errorHelpers.h"
#include "kernelHelpers.h"

// #define log_info(...) printf(__VA_ARGS__)
// #define log_error(...) printf(__VA_ARGS__)

#define NonTestRequire(x, ...) \
do \
{ \
    if (!(x) ) \
    { \
        log_info("\n[assertion failed: %s at %s:%d]\n", #x, __FILE__, __LINE__); \
        log_info("CATASTROPHIC NON-TEST ERROR: "); \
        log_error(__VA_ARGS__); \
        log_info("\n"); \
        log_info("***FAILED***\n"); \
        exit(1); \
    } \
} while (0)

#define TestRequire(x, ...) \
    do \
    { \
        if (!(x) ) \
        { \
            log_info("\n[assertion failed: %s at %s:%d]\n", #x, __FILE__, __LINE__); \
            log_info("ERROR: "); \
            log_error(__VA_ARGS__); \
            log_info("\n"); \
            HarnessD3D11_TestFail(); \
            goto Cleanup; \
        } \
    } while (0)

#define TestPrint(...) \
    do \
    { \
        log_error(__VA_ARGS__); \
    } while (0)

struct TextureFormat
{
    DXGI_FORMAT format;
    cl_channel_order channel_order;
    cl_channel_type  channel_type;
    UINT bytesPerPixel;
    enum
    {
        GENERIC_FLOAT = 0,
        GENERIC_UINT  = 1,
        GENERIC_SINT  = 2,
    } generic;

    const char *name_format;
    const char *name_channel_order;
    const char *name_channel_type;
};
extern TextureFormat formats[];
extern UINT formatCount;


#define MAX_REGISTERED_SUBRESOURCES 4 // limit to just make life easier

struct BufferProperties
{
    UINT ByteWidth;
    UINT BindFlags;
    D3D11_USAGE Usage;
    UINT CPUAccess;
    const char* name_BindFlags;
    const char* name_Usage;
    const char* name_CPUAccess;
};

struct Texture2DSize
{
    UINT Width;
    UINT Height;
    UINT MipLevels;
    UINT ArraySize;
    UINT SubResourceCount;
    struct
    {
        UINT MipLevel;
        UINT ArraySlice;
    } subResources[MAX_REGISTERED_SUBRESOURCES];
    UINT MiscFlags;
};
struct Texture3DSize
{
    UINT Width;
    UINT Height;
    UINT Depth;
    UINT MipLevels;
    UINT SubResourceCount;
    struct
    {
        UINT MipLevel;
    } subResources[MAX_REGISTERED_SUBRESOURCES];
    UINT MiscFlags;
};

void HarnessD3D11_Initialize(cl_platform_id platform);
cl_int HarnessD3D11_CreateDevice(
    IDXGIAdapter* pAdapter,
    ID3D11Device **ppDevice,
    ID3D11DeviceContext** ppDC);
void HarnessD3D11_DestroyDevice();

void HarnessD3D11_TestBegin(const char* fmt, ...);
void HarnessD3D11_TestFail();
void HarnessD3D11_TestEnd();
void HarnessD3D11_TestStats();

void TestAdapterEnumeration(
    cl_platform_id platform,
    IDXGIAdapter* pAdapter,
    ID3D11Device* pDevice,
    cl_uint* num_devices);

void TestAdapterDevices(
    cl_platform_id platform,
    IDXGIAdapter* pAdapter,
    ID3D11Device* pDevice,
    ID3D11DeviceContext* pDC,
    cl_uint num_devices);

void TestDevice(
    cl_device_id device,
    ID3D11Device* pDevice,
    ID3D11DeviceContext* pDC);

bool TestDeviceContextCreate(
    cl_device_id device,
    ID3D11Device* pDevice,
    cl_context* out_context,
    cl_command_queue* out_command_queue);

void TestDeviceBuffer(
    cl_context context,
    cl_command_queue command_queue,
    ID3D11Device* pDevice,
    ID3D11DeviceContext* pDC);

void TestDeviceTexture2D(
    cl_device_id device,
    cl_context context,
    cl_command_queue command_queue,
    ID3D11Device* pDevice,
    ID3D11DeviceContext* pDC);

void TestDeviceTexture3D(
    cl_device_id device,
    cl_context context,
    cl_command_queue command_queue,
    ID3D11Device* pDevice,
    ID3D11DeviceContext* pDC);

void TestDeviceMisc(
    cl_device_id device,
    cl_context context,
    cl_command_queue command_queue,
    ID3D11Device* pDevice);

cl_int HarnessD3D11_CreateKernelFromSource(
    cl_kernel *outKernel,
    cl_device_id device,
    cl_context context,
    const char *source,
    const char *entrypoint);

extern clGetDeviceIDsFromD3D11KHR_fn      clGetDeviceIDsFromD3D11KHR;
extern clCreateFromD3D11BufferKHR_fn      clCreateFromD3D11BufferKHR;
extern clCreateFromD3D11Texture2DKHR_fn   clCreateFromD3D11Texture2DKHR;
extern clCreateFromD3D11Texture3DKHR_fn   clCreateFromD3D11Texture3DKHR;
extern clEnqueueAcquireD3D11ObjectsKHR_fn clEnqueueAcquireD3D11ObjectsKHR;
extern clEnqueueReleaseD3D11ObjectsKHR_fn clEnqueueReleaseD3D11ObjectsKHR;

#endif
