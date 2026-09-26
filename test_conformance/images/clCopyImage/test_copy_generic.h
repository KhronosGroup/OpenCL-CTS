//
// Copyright (c) 2026 The Khronos Group Inc.
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
#ifndef IMAGES_COPY_GENERIC_H
#define IMAGES_COPY_GENERIC_H

#include "../testBase.h"
#include "typeWrappers.h"
#include "testHarness.h"

struct copy_image_buffers_t
{
    clMemWrapper srcImage, dstImage;
    BufferOwningPtr<char> srcData, dstData;
};
struct copy_image_env_t
{
    cl_context context;
    cl_command_queue queue;
    MTdata d;
    const image_test_context_t &ctx;
};

int test_copy_init_images(copy_image_env_t &env, image_descriptor *srcImageInfo,
                          image_descriptor *dstImageInfo,
                          copy_image_buffers_t &buffers);

int test_copy_image_generic(copy_image_env_t &env,
                            image_descriptor *srcImageInfo,
                            image_descriptor *dstImageInfo,
                            copy_image_buffers_t &buffers,
                            const size_t sourcePos[], const size_t destPos[],
                            const size_t regionSize[]);

#endif
