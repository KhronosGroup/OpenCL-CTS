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

int test_copy_init_images(cl_context context, cl_command_queue queue,
                          image_descriptor *srcImageInfo,
                          image_descriptor *dstImageInfo,
                          clMemWrapper &srcImage, clMemWrapper &dstImage,
                          BufferOwningPtr<char> &srcData,
                          BufferOwningPtr<char> &dstData, MTdata d,
                          const context_t &ctx);

int test_copy_image_generic(cl_context context, cl_command_queue queue,
                            image_descriptor *srcImageInfo,
                            image_descriptor *dstImageInfo,
                            clMemWrapper &srcImage, clMemWrapper &dstImage,
                            BufferOwningPtr<char> &srcData,
                            BufferOwningPtr<char> &dstHost,
                            const size_t sourcePos[], const size_t destPos[],
                            const size_t regionSize[], MTdata d,
                            const context_t &ctx);

#endif
