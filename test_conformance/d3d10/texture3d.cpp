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
#define _CRT_SECURE_NO_WARNINGS
#include "harness.h"

Texture3DSize texture3DSizes[] =
{
    {
        4, // Width
        4, // Height
        4, // Depth
        1, // MipLevels
        1, // SubResourceCount
        {  // SubResources
            { 0 }, // MipLevel
            { 0 }, // MipLevel
            { 0 }, // MipLevel
            { 0 }, // MipLevel
        },
        0, // MiscFlags
    },
    {
        127, // Width
        25, // Height
        33, // Depth
        1, // MipLevels
        1, // SubResourceCount
        {  // SubResources
            { 0 }, // MipLevel
            { 0 }, // MipLevel
            { 0 }, // MipLevel
            { 0 }, // MipLevel
        },
        0, // MiscFlags
    },
    {
        128, // Width
        256, // Height
        64, // Depth
        4, // MipLevels
        3, // SubResourceCount
        {  // SubResources
            { 2 }, // MipLevel
            { 1 }, // MipLevel
            { 0 }, // MipLevel
            { 0 }, // MipLevel
        },
        0, // MiscFlags
    },
    {
        512, // Width
         64, // Height
         32, // Depth
        3, // MipLevels
        1, // SubResourceCount
        {  // SubResources
            { 2 }, // MipLevel
            { 0 }, // MipLevel
            { 0 }, // MipLevel
            { 0 }, // MipLevel
        },
        0, // MiscFlags
    },
};
UINT texture3DSizeCount = sizeof(texture3DSizes)/sizeof(texture3DSizes[0]);

const char *
texture3DPatterns[2][2][2] =
{
    {
        {"PlaceTheCasseroleDis", "hInAColdOvenPlaceACh"},
        {"airFacingTheOvenAndS", "itInItForeverThinkAb"},
    },
    {
        {"outHowHungryYouAreWh", "enNightFallsDoNotTur"},
        {"nOnTheLightMyEyeBeca", "meInflamedIHateCamus"},
    },
};

void SubTestTexture3D(
    cl_context context,
    cl_command_queue command_queue,
    ID3D10Device* pDevice,
    const TextureFormat* format,
    const Texture3DSize* size)
{
    ID3D10Texture3D* pTexture = NULL;
    HRESULT hr = S_OK;

    cl_int result = CL_SUCCESS;

    HarnessD3D10_TestBegin("3D Texture: Format=%s, Width=%d, Height=%d, Depth=%d, MipLevels=%d",
        format->name_format,
        size->Width,
        size->Height,
        size->Depth,
        size->MipLevels);

    struct
    {
        cl_mem mem;
        UINT subResource;
        UINT width;
        UINT height;
        UINT depth;
    }
    subResourceInfo[4];

    // create the D3D10 resources
    {
        D3D10_TEXTURE3D_DESC desc;
        memset(&desc, 0, sizeof(desc) );
        desc.Width      = size->Width;
        desc.Height     = size->Height;
        desc.Depth      = size->Depth;
        desc.MipLevels  = size->MipLevels;
        desc.Format     = format->format;
        desc.Usage = D3D10_USAGE_DEFAULT;
        desc.BindFlags = D3D10_BIND_SHADER_RESOURCE;
        desc.CPUAccessFlags = 0;
        desc.MiscFlags = 0;

        hr = pDevice->CreateTexture3D(&desc, NULL, &pTexture);
        TestRequire(SUCCEEDED(hr), "CreateTexture3D failed.");
    }

    // initialize some useful variables
    for (UINT i = 0; i < size->SubResourceCount; ++i)
    {
        // compute the expected values for the subresource
        subResourceInfo[i].subResource = size->subResources[i].MipLevel;
        subResourceInfo[i].width = size->Width;
        subResourceInfo[i].height = size->Height;
        subResourceInfo[i].depth = size->Depth;
        for (UINT j = 0; j < size->subResources[i].MipLevel; ++j) {
            subResourceInfo[i].width /= 2;
            subResourceInfo[i].height /= 2;
            subResourceInfo[i].depth /= 2;
        }
    }

    // copy a pattern into the corners of the image, coordinates
    for (UINT i = 0; i < size->SubResourceCount; ++i)
    for (UINT x = 0; x < 2; ++x)
    for (UINT y = 0; y < 2; ++y)
    for (UINT z = 0; z < 2; ++z)
    {
        // create the staging buffer
        ID3D10Texture3D* pStagingBuffer = NULL;
        {
            D3D10_TEXTURE3D_DESC desc = {0};
            desc.Width      = 1;
            desc.Height     = 1;
            desc.Depth      = 1;
            desc.MipLevels  = 1;
            desc.Format     = format->format;
            desc.Usage = D3D10_USAGE_STAGING;
            desc.BindFlags = 0;
            desc.CPUAccessFlags = D3D10_CPU_ACCESS_READ | D3D10_CPU_ACCESS_WRITE;
            desc.MiscFlags = 0;
            hr = pDevice->CreateTexture3D(&desc, NULL, &pStagingBuffer);
            TestRequire(SUCCEEDED(hr), "CreateTexture3D failed.");
        }

        // write the data to the staging buffer
        {
            D3D10_MAPPED_TEXTURE3D mappedTexture;
            hr = pStagingBuffer->Map(
                0,
                D3D10_MAP_READ_WRITE,
                0,
                &mappedTexture);
            memcpy(mappedTexture.pData, texture3DPatterns[x][y][z], format->bytesPerPixel);
            pStagingBuffer->Unmap(0);
        }

        // copy the data to to the texture
        {
            D3D10_BOX box = {0};
            box.front   = 0; box.back    = 1;
            box.top     = 0; box.bottom  = 1;
            box.left    = 0; box.right   = 1;
            pDevice->CopySubresourceRegion(
                pTexture,
                subResourceInfo[i].subResource,
                x ? subResourceInfo[i].width  - 1 : 0,
                y ? subResourceInfo[i].height - 1 : 0,
                z ? subResourceInfo[i].depth  - 1 : 0,
                pStagingBuffer,
                0,
                &box);
        }

        pStagingBuffer->Release();
    }

    // create the cl_mem objects for the resources and verify its sanity
    for (UINT i = 0; i < size->SubResourceCount; ++i)
    {
        // create a cl_mem for the resource
        subResourceInfo[i].mem = clCreateFromD3D10Texture3DKHR(
            context,
            0,
            pTexture,
            subResourceInfo[i].subResource,
            &result);
        TestRequire(result == CL_SUCCESS, "clCreateFromD3D10Texture3DKHR failed");

        // query resource pointer and verify
        ID3D10Resource* clResource = NULL;
        result = clGetMemObjectInfo(
            subResourceInfo[i].mem,
            CL_MEM_D3D10_RESOURCE_KHR,
            sizeof(clResource),
            &clResource,
            NULL);
        TestRequire(result == CL_SUCCESS, "clGetMemObjectInfo for CL_MEM_D3D10_RESOURCE_KHR failed.");
        TestRequire(clResource == pTexture, "clGetMemObjectInfo for CL_MEM_D3D10_RESOURCE_KHR returned incorrect value.");

        // query subresource and verify
        UINT clSubResource;
        result = clGetImageInfo(
            subResourceInfo[i].mem,
            CL_IMAGE_D3D10_SUBRESOURCE_KHR,
            sizeof(clSubResource),
            &clSubResource,
            NULL);
        TestRequire(result == CL_SUCCESS, "clGetImageInfo for CL_IMAGE_D3D10_SUBRESOURCE_KHR failed");
        TestRequire(clSubResource == subResourceInfo[i].subResource, "clGetImageInfo for CL_IMAGE_D3D10_SUBRESOURCE_KHR returned incorrect value.");

        // query format and verify
        cl_image_format clFormat;
        result = clGetImageInfo(
            subResourceInfo[i].mem,
            CL_IMAGE_FORMAT,
            sizeof(clFormat),
            &clFormat,
            NULL);
        TestRequire(result == CL_SUCCESS, "clGetImageInfo for CL_IMAGE_FORMAT failed");
        TestRequire(clFormat.image_channel_order == format->channel_order, "clGetImageInfo for CL_IMAGE_FORMAT returned incorrect channel order.");
        TestRequire(clFormat.image_channel_data_type == format->channel_type, "clGetImageInfo for CL_IMAGE_FORMAT returned incorrect channel data type.");

        // query width
        size_t width;
        result = clGetImageInfo(
            subResourceInfo[i].mem,
            CL_IMAGE_WIDTH,
            sizeof(width),
            &width,
            NULL);
        TestRequire(result == CL_SUCCESS, "clGetImageInfo for CL_IMAGE_WIDTH failed");
        TestRequire(width == subResourceInfo[i].width, "clGetImageInfo for CL_IMAGE_HEIGHT returned incorrect value.");

        // query height
        size_t height;
        result = clGetImageInfo(
            subResourceInfo[i].mem,
            CL_IMAGE_HEIGHT,
            sizeof(height),
            &height,
            NULL);
        TestRequire(result == CL_SUCCESS, "clGetImageInfo for CL_IMAGE_HEIGHT failed");
        TestRequire(height == subResourceInfo[i].height, "clGetImageInfo for CL_IMAGE_HEIGHT returned incorrect value.");

        // query depth
        size_t depth;
        result = clGetImageInfo(
            subResourceInfo[i].mem,
            CL_IMAGE_DEPTH,
            sizeof(depth),
            &depth,
            NULL);
        TestRequire(result == CL_SUCCESS, "clGetImageInfo for CL_IMAGE_DEPTH failed");
        TestRequire(depth == subResourceInfo[i].depth, "clGetImageInfo for CL_IMAGE_DEPTH returned incorrect value.");

    }

    // acquire the resources for OpenCL
    {
        cl_mem memToAcquire[MAX_REGISTERED_SUBRESOURCES];

        // cut the registered sub-resources into two sets and send the acquire calls for them separately
        for(UINT i = 0; i < size->SubResourceCount; ++i)
        {
            memToAcquire[i] = subResourceInfo[i].mem;
        }

        // do the acquire
        result = clEnqueueAcquireD3D10ObjectsKHR(
            command_queue,
            size->SubResourceCount,
            memToAcquire,
            0,
            NULL,
            NULL);
        TestRequire(result == CL_SUCCESS, "clEnqueueAcquireD3D10ObjectsKHR failed.");
    }

    // download the data using OpenCL & compare with the expected results
    // copy the corners of the image into the image
    for (UINT i = 0; i < size->SubResourceCount; ++i)
    for (UINT x = 0; x < 2; ++x)
    for (UINT y = 0; y < 2; ++y)
    for (UINT z = 0; z < 2; ++z)
    {
        if (x == y && y == z && 0)
        {
            continue;
        }
        size_t src[3] =
        {
            x ? subResourceInfo[i].width  - 1 : 0,
            y ? subResourceInfo[i].height - 1 : 0,
            z ? subResourceInfo[i].depth  - 1 : 0,
        };
        size_t dst[3] =
        {
            x ? subResourceInfo[i].width  - 2 : 1,
            y ? subResourceInfo[i].height - 2 : 1,
            z ? subResourceInfo[i].depth  - 2 : 1,
        };
        size_t region[3] =
        {
            1,
            1,
            1,
        };
        result = clEnqueueCopyImage(
            command_queue,
            subResourceInfo[i].mem,
            subResourceInfo[i].mem,
            src,
            dst,
            region,
            0,
            NULL,
            NULL);
        TestRequire(result == CL_SUCCESS, "clEnqueueCopyImage failed.");
    }

    // release the resource from OpenCL
    {
        cl_mem memToAcquire[MAX_REGISTERED_SUBRESOURCES];
        for(UINT i = 0; i < size->SubResourceCount; ++i)
        {
            memToAcquire[i] = subResourceInfo[i].mem;
        }

        // do the release
        result = clEnqueueReleaseD3D10ObjectsKHR(
            command_queue,
            size->SubResourceCount,
            memToAcquire,
            0,
            NULL,
            NULL);
        TestRequire(result == CL_SUCCESS, "clEnqueueReleaseD3D10ObjectsKHR failed.");
    }

    for (UINT i = 0; i < size->SubResourceCount; ++i)
    for (UINT x = 0; x < 2; ++x)
    for (UINT y = 0; y < 2; ++y)
    for (UINT z = 0; z < 2; ++z)
    {
        // create the staging buffer
        ID3D10Texture3D* pStagingBuffer = NULL;
        {
            D3D10_TEXTURE3D_DESC desc = {0};
            desc.Width      = 1;
            desc.Height     = 1;
            desc.Depth      = 1;
            desc.MipLevels  = 1;
            desc.Format     = format->format;
            desc.Usage = D3D10_USAGE_STAGING;
            desc.BindFlags = 0;
            desc.CPUAccessFlags = D3D10_CPU_ACCESS_READ | D3D10_CPU_ACCESS_WRITE;
            desc.MiscFlags = 0;
            hr = pDevice->CreateTexture3D(&desc, NULL, &pStagingBuffer);
            TestRequire(SUCCEEDED(hr), "Failed to create staging buffer.");
        }

        // wipe out the staging buffer to make sure we don't get stale values
        {
            D3D10_MAPPED_TEXTURE3D mappedTexture;
            hr = pStagingBuffer->Map(
                0,
                D3D10_MAP_READ_WRITE,
                0,
                &mappedTexture);
            TestRequire(SUCCEEDED(hr), "Failed to map staging buffer");
            memset(mappedTexture.pData, 0, format->bytesPerPixel);
            pStagingBuffer->Unmap(0);
        }

        // copy the pixel to the staging buffer
        {
            D3D10_BOX box = {0};
            box.left    = x ? subResourceInfo[i].width  - 2 : 1; box.right  = box.left  + 1;
            box.top     = y ? subResourceInfo[i].height - 2 : 1; box.bottom = box.top   + 1;
            box.front   = z ? subResourceInfo[i].depth  - 2 : 1; box.back   = box.front + 1;
            pDevice->CopySubresourceRegion(
                pStagingBuffer,
                0,
                0,
                0,
                0,
                pTexture,
                subResourceInfo[i].subResource,
                &box);
        }

        // make sure we read back what was written next door
        {
            D3D10_MAPPED_TEXTURE3D mappedTexture;
            hr = pStagingBuffer->Map(
                0,
                D3D10_MAP_READ_WRITE,
                0,
                &mappedTexture);
            TestRequire(SUCCEEDED(hr), "Failed to map staging buffer");

            /*
            // This can be helpful in debugging...
            printf("\n");
            for (UINT k = 0; k < format->bytesPerPixel; ++k)
            {
                printf("[%c %c]\n",
                    texture2DPatterns[x][y][k],
                    ( (char *)mappedTexture.pData )[k]);
            }
            */

            TestRequire(
                !memcmp(mappedTexture.pData, texture3DPatterns[x][y][z], format->bytesPerPixel),
                "Failed to map staging buffer");

            pStagingBuffer->Unmap(0);
        }

        pStagingBuffer->Release();
    }


Cleanup:

    if (pTexture)
    {
        pTexture->Release();
    }
    for (UINT i = 0; i < size->SubResourceCount; ++i)
    {
        clReleaseMemObject(subResourceInfo[i].mem);
    }

    HarnessD3D10_TestEnd();
}


void TestDeviceTexture3D(
    cl_device_id device,
    cl_context context,
    cl_command_queue command_queue,
    ID3D10Device* pDevice)
{
    cl_int result = CL_SUCCESS;


    for (UINT format = 0, size = 0; format < formatCount; ++size, ++format)
    {
        SubTestTexture3D(
            context,
            command_queue,
            pDevice,
            &formats[format],
            &texture3DSizes[size % texture3DSizeCount]);
    }
}

