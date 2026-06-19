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
#include "harness.h"

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
HarnessD3D11_Initialize(cl_platform_id platform)
{
    // extract function pointers for exported functions
    INITPFN(clGetDeviceIDsFromD3D11KHR);
    INITPFN(clCreateFromD3D11BufferKHR);
    INITPFN(clCreateFromD3D11Texture2DKHR);
    INITPFN(clCreateFromD3D11Texture3DKHR);
    INITPFN(clEnqueueAcquireD3D11ObjectsKHR);
    INITPFN(clEnqueueReleaseD3D11ObjectsKHR);
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

        status = create_single_kernel_helper(context, &program, &kernel, 1,
                                             &sourceTexts[0], entrypoint);
        TestRequire(CL_SUCCESS == status, "Kernel creation failed");
    }

    clReleaseProgram(program);
    *outKernel = kernel;

Cleanup:

    return CL_SUCCESS;
}
