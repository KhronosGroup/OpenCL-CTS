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
#include "testBase.h"
#include "common.h"

#if defined(__APPLE__)
#include <OpenGL/glu.h>
#else
#include <GL/glu.h>
#include <CL/cl_gl.h>
#endif
#include <algorithm>

void calc_3D_size_descriptors(sizevec_t* sizes, size_t nsizes)
{
    // Need to limit array size according to GL device properties
    GLint maxTextureSize = 2048;
    glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &maxTextureSize);

    RandomSeed seed(gRandomSeed);

    // Generate some random sizes (within reasonable ranges)
    for (size_t i = 0; i < nsizes; i++)
    {
        sizes[i].width =
            random_in_range(2, std::min(maxTextureSize, 1 << (i + 4)), seed);
        sizes[i].height =
            random_in_range(2, std::min(maxTextureSize, 1 << (i + 4)), seed);
        sizes[i].depth =
            random_in_range(2, std::min(maxTextureSize, 1 << (i + 4)), seed);
    }
}

int test_images_read_3D(cl_device_id device, cl_context context,
                        cl_command_queue queue, int numElements)
{
    GLenum targets[] = { GL_TEXTURE_3D };
    size_t ntargets = 1;

    size_t nformats = sizeof(common_formats) / sizeof(common_formats[0]);

    const size_t nsizes = 6;
    sizevec_t sizes[nsizes];
    calc_3D_size_descriptors(sizes, nsizes);

    return test_images_read_common(device, context, queue, common_formats,
                                   nformats, targets, ntargets, sizes, nsizes);
}

#pragma mark -
#pragma marm _3D write test

int test_images_write_3D(cl_device_id device, cl_context context,
                         cl_command_queue queue, int numElements)
{
    // TODO: Perhaps the expected behavior is to FAIL if 3D images are
    //       unsupported?

    if (!is_extension_available(device, "cl_khr_3d_image_writes"))
    {
        log_info(
            "This device does not support 3D image writes.  Skipping test.\n");
        return 0;
    }

    GLenum targets[] = { GL_TEXTURE_3D };
    size_t ntargets = sizeof(targets) / sizeof(targets[0]);
    size_t nformats = sizeof(common_formats) / sizeof(common_formats[0]);


    const size_t nsizes = 6;
    sizevec_t sizes[nsizes];
    calc_3D_size_descriptors(sizes, nsizes);

    return test_images_write_common(device, context, queue, common_formats,
                                    nformats, targets, ntargets, sizes, nsizes);
}

#pragma mark -
#pragma mark _3D get info test

int test_images_3D_getinfo(cl_device_id device, cl_context context,
                           cl_command_queue queue, int numElements)
{
    GLenum targets[] = { GL_TEXTURE_3D };
    size_t ntargets = 1;

    size_t nformats = sizeof(common_formats) / sizeof(common_formats[0]);

    const size_t nsizes = 6;
    sizevec_t sizes[nsizes];
    calc_3D_size_descriptors(sizes, nsizes);

    return test_images_get_info_common(device, context, queue, common_formats,
                                       nformats, targets, ntargets, sizes,
                                       nsizes);
}
