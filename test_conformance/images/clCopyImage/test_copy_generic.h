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
#include "imageHelpers.h"

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

int test_image_type(cl_device_id device, cl_context context,
                    cl_command_queue queue, const struct TestConfigs &config,
                    const image_test_context_t &ctx);

using test_function_t = int (*)(cl_device_id, cl_context, cl_command_queue,
                                cl_mem_flags, cl_mem_object_type, cl_mem_flags,
                                cl_mem_object_type, cl_image_format *,
                                const image_test_context_t &);

struct TestConfigs
{
    std::string name;
    cl_mem_object_type src_type;
    cl_mem_flags src_flags;
    cl_mem_object_type dst_type;
    cl_mem_flags dst_flags;
    test_function_t func;
    cl_channel_type channel_type;

    TestConfigs(const char *name_, cl_mem_object_type src_type_,
                cl_mem_flags src_flags_, cl_mem_object_type dst_type_,
                cl_mem_flags dst_flags_, test_function_t func_,
                cl_channel_type channel_type_)
        : src_type(src_type_), src_flags(src_flags_), dst_type(dst_type_),
          dst_flags(dst_flags_), func(func_), channel_type(channel_type_)
    {
        name += name_;
        name += "_";
        name += cl_channel_type_to_string(channel_type);
    }
};

int test_copy_image_set_1D(cl_device_id device, cl_context context,
                           cl_command_queue queue, cl_mem_flags src_flags,
                           cl_mem_object_type src_type, cl_mem_flags dst_flags,
                           cl_mem_object_type dst_type, cl_image_format *format,
                           const image_test_context_t &ctx);
int test_copy_image_set_2D(cl_device_id device, cl_context context,
                           cl_command_queue queue, cl_mem_flags src_flags,
                           cl_mem_object_type src_type, cl_mem_flags dst_flags,
                           cl_mem_object_type dst_type, cl_image_format *format,
                           const image_test_context_t &ctx);
int test_copy_image_set_3D(cl_device_id device, cl_context context,
                           cl_command_queue queue, cl_mem_flags src_flags,
                           cl_mem_object_type src_type, cl_mem_flags dst_flags,
                           cl_mem_object_type dst_type, cl_image_format *format,
                           const image_test_context_t &ctx);
int test_copy_image_set_1D_array(cl_device_id device, cl_context context,
                                 cl_command_queue queue, cl_mem_flags src_flags,
                                 cl_mem_object_type src_type,
                                 cl_mem_flags dst_flags,
                                 cl_mem_object_type dst_type,
                                 cl_image_format *format,
                                 const image_test_context_t &ctx);
int test_copy_image_set_2D_array(cl_device_id device, cl_context context,
                                 cl_command_queue queue, cl_mem_flags src_flags,
                                 cl_mem_object_type src_type,
                                 cl_mem_flags dst_flags,
                                 cl_mem_object_type dst_type,
                                 cl_image_format *format,
                                 const image_test_context_t &ctx);
int test_copy_image_set_2D_3D(cl_device_id device, cl_context context,
                              cl_command_queue queue, cl_mem_flags src_flags,
                              cl_mem_object_type src_type,
                              cl_mem_flags dst_flags,
                              cl_mem_object_type dst_type,
                              cl_image_format *format,
                              const image_test_context_t &ctx);
int test_copy_image_set_2D_2D_array(
    cl_device_id device, cl_context context, cl_command_queue queue,
    cl_mem_flags src_flags, cl_mem_object_type src_type, cl_mem_flags dst_flags,
    cl_mem_object_type dst_type, cl_image_format *format,
    const image_test_context_t &ctx);
int test_copy_image_set_3D_2D_array(
    cl_device_id device, cl_context context, cl_command_queue queue,
    cl_mem_flags src_flags, cl_mem_object_type src_type, cl_mem_flags dst_flags,
    cl_mem_object_type dst_type, cl_image_format *format,
    const image_test_context_t &ctx);
int test_copy_image_set_1D_buffer(
    cl_device_id device, cl_context context, cl_command_queue queue,
    cl_mem_flags src_flags, cl_mem_object_type src_type, cl_mem_flags dst_flags,
    cl_mem_object_type dst_type, cl_image_format *format,
    const image_test_context_t &ctx);
int test_copy_image_set_1D_1D_buffer(
    cl_device_id device, cl_context context, cl_command_queue queue,
    cl_mem_flags src_flags, cl_mem_object_type src_type, cl_mem_flags dst_flags,
    cl_mem_object_type dst_type, cl_image_format *format,
    const image_test_context_t &ctx);

#endif
