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
#include "../testBase.h"
#include "../common.h"
#include <algorithm>

extern int
test_copy_image_set_1D(cl_device_id device, cl_context context,
                       cl_command_queue queue, cl_mem_flags src_flags,
                       cl_mem_object_type src_type, cl_mem_flags dst_flags,
                       cl_mem_object_type dst_type, cl_image_format *format);
extern int
test_copy_image_set_2D(cl_device_id device, cl_context context,
                       cl_command_queue queue, cl_mem_flags src_flags,
                       cl_mem_object_type src_type, cl_mem_flags dst_flags,
                       cl_mem_object_type dst_type, cl_image_format *format);
extern int
test_copy_image_set_3D(cl_device_id device, cl_context context,
                       cl_command_queue queue, cl_mem_flags src_flags,
                       cl_mem_object_type src_type, cl_mem_flags dst_flags,
                       cl_mem_object_type dst_type, cl_image_format *format);
extern int test_copy_image_set_1D_array(
    cl_device_id device, cl_context context, cl_command_queue queue,
    cl_mem_flags src_flags, cl_mem_object_type src_type, cl_mem_flags dst_flags,
    cl_mem_object_type dst_type, cl_image_format *format);
extern int test_copy_image_set_2D_array(
    cl_device_id device, cl_context context, cl_command_queue queue,
    cl_mem_flags src_flags, cl_mem_object_type src_type, cl_mem_flags dst_flags,
    cl_mem_object_type dst_type, cl_image_format *format);
extern int
test_copy_image_set_2D_3D(cl_device_id device, cl_context context,
                          cl_command_queue queue, cl_mem_flags src_flags,
                          cl_mem_object_type src_type, cl_mem_flags dst_flags,
                          cl_mem_object_type dst_type, cl_image_format *format);
extern int test_copy_image_set_2D_2D_array(
    cl_device_id device, cl_context context, cl_command_queue queue,
    cl_mem_flags src_flags, cl_mem_object_type src_type, cl_mem_flags dst_flags,
    cl_mem_object_type dst_type, cl_image_format *format);
extern int test_copy_image_set_3D_2D_array(
    cl_device_id device, cl_context context, cl_command_queue queue,
    cl_mem_flags src_flags, cl_mem_object_type src_type, cl_mem_flags dst_flags,
    cl_mem_object_type dst_type, cl_image_format *format);
extern int test_copy_image_set_1D_buffer(
    cl_device_id device, cl_context context, cl_command_queue queue,
    cl_mem_flags src_flags, cl_mem_object_type src_type, cl_mem_flags dst_flags,
    cl_mem_object_type dst_type, cl_image_format *format);
extern int test_copy_image_set_1D_1D_buffer(
    cl_device_id device, cl_context context, cl_command_queue queue,
    cl_mem_flags src_flags, cl_mem_object_type src_type, cl_mem_flags dst_flags,
    cl_mem_object_type dst_type, cl_image_format *format);

using test_function_t = int (*)(cl_device_id, cl_context, cl_command_queue,
                                cl_mem_flags, cl_mem_object_type, cl_mem_flags,
                                cl_mem_object_type, cl_image_format *);

struct TestConfigs
{
    const char *name;
    cl_mem_object_type src_type;
    cl_mem_flags src_flags;
    cl_mem_object_type dst_type;
    cl_mem_flags dst_flags;

    TestConfigs(const char *name_, cl_mem_object_type src_type_,
                cl_mem_flags src_flags_, cl_mem_object_type dst_type_,
                cl_mem_flags dst_flags_)
        : name(name_), src_type(src_type_), src_flags(src_flags_),
          dst_type(dst_type_), dst_flags(dst_flags_)
    {}
};

int test_image_type(cl_device_id device, cl_context context,
                    cl_command_queue queue, MethodsToTest testMethod)
{
    test_function_t test_fn = nullptr;

    if ( gTestMipmaps )
    {
        if ( 0 == is_extension_available( device, "cl_khr_mipmap_image" ))
        {
            log_info( "-----------------------------------------------------\n" );
            log_info( "This device does not support cl_khr_mipmap_image.\nSkipping mipmapped image test. \n" );
            log_info( "-----------------------------------------------------\n\n" );
            return TEST_SKIPPED_ITSELF;
        }
    }

    std::vector<TestConfigs> test_configs;
    switch (testMethod)
    {
        case k1D:
            test_configs.emplace_back(
                "1D -> 1D", CL_MEM_OBJECT_IMAGE1D,
                gEnablePitch ? CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY
                             : CL_MEM_READ_ONLY,
                CL_MEM_OBJECT_IMAGE1D,
                gEnablePitch ? CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY
                             : CL_MEM_READ_ONLY);
            test_fn = test_copy_image_set_1D;
            break;
        case k2D:
            test_configs.emplace_back(
                "2D -> 2D", CL_MEM_OBJECT_IMAGE2D,
                gEnablePitch ? CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY
                             : CL_MEM_READ_ONLY,
                CL_MEM_OBJECT_IMAGE2D,
                gEnablePitch ? CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY
                             : CL_MEM_READ_ONLY);
            test_fn = test_copy_image_set_2D;
            break;
        case k3D:
            test_configs.emplace_back(
                "3D -> 3D", CL_MEM_OBJECT_IMAGE3D,
                gEnablePitch ? CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY
                             : CL_MEM_READ_ONLY,
                CL_MEM_OBJECT_IMAGE3D,
                gEnablePitch ? CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY
                             : CL_MEM_READ_ONLY);
            test_fn = test_copy_image_set_3D;
            break;
        case k1DArray:
            test_configs.emplace_back(
                "1D array -> 1D array", CL_MEM_OBJECT_IMAGE1D_ARRAY,
                gEnablePitch ? CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY
                             : CL_MEM_READ_ONLY,
                CL_MEM_OBJECT_IMAGE1D_ARRAY,
                gEnablePitch ? CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY
                             : CL_MEM_READ_ONLY);
            test_fn = test_copy_image_set_1D_array;
            break;
        case k2DArray:
            test_configs.emplace_back(
                "2D array -> 2D array", CL_MEM_OBJECT_IMAGE2D_ARRAY,
                gEnablePitch ? CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY
                             : CL_MEM_READ_ONLY,
                CL_MEM_OBJECT_IMAGE2D_ARRAY,
                gEnablePitch ? CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY
                             : CL_MEM_READ_ONLY);
            test_fn = test_copy_image_set_2D_array;
            break;
        case k2DTo3D:
            test_configs.emplace_back(
                "2D -> 3D", CL_MEM_OBJECT_IMAGE2D,
                gEnablePitch ? CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY
                             : CL_MEM_READ_ONLY,
                CL_MEM_OBJECT_IMAGE3D,
                gEnablePitch ? CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY
                             : CL_MEM_READ_ONLY);
            test_fn = test_copy_image_set_2D_3D;
            break;
        case k3DTo2D:
            test_configs.emplace_back(
                "3D -> 2D", CL_MEM_OBJECT_IMAGE3D,
                gEnablePitch ? CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY
                             : CL_MEM_READ_ONLY,
                CL_MEM_OBJECT_IMAGE2D,
                gEnablePitch ? CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY
                             : CL_MEM_READ_ONLY);
            test_fn = test_copy_image_set_2D_3D;
            break;
        case k2DArrayTo2D:
            test_configs.emplace_back(
                "2D array -> 2D", CL_MEM_OBJECT_IMAGE2D_ARRAY,
                gEnablePitch ? CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY
                             : CL_MEM_READ_ONLY,
                CL_MEM_OBJECT_IMAGE2D,
                gEnablePitch ? CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY
                             : CL_MEM_READ_ONLY);
            test_fn = test_copy_image_set_2D_2D_array;
            break;
        case k2DTo2DArray:
            test_configs.emplace_back(
                "2D -> 2D array", CL_MEM_OBJECT_IMAGE2D,
                gEnablePitch ? CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY
                             : CL_MEM_READ_ONLY,
                CL_MEM_OBJECT_IMAGE2D_ARRAY,
                gEnablePitch ? CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY
                             : CL_MEM_READ_ONLY);
            test_fn = test_copy_image_set_2D_2D_array;
            break;
        case k2DArrayTo3D:
            test_configs.emplace_back(
                "2D array -> 3D", CL_MEM_OBJECT_IMAGE2D_ARRAY,
                gEnablePitch ? CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY
                             : CL_MEM_READ_ONLY,
                CL_MEM_OBJECT_IMAGE3D,
                gEnablePitch ? CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY
                             : CL_MEM_READ_ONLY);
            test_fn = test_copy_image_set_3D_2D_array;
            break;
        case k3DTo2DArray:
            test_configs.emplace_back(
                "3D -> 2D array", CL_MEM_OBJECT_IMAGE3D,
                gEnablePitch ? CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY
                             : CL_MEM_READ_ONLY,
                CL_MEM_OBJECT_IMAGE2D_ARRAY,
                gEnablePitch ? CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY
                             : CL_MEM_READ_ONLY);
            test_fn = test_copy_image_set_3D_2D_array;
            break;
        case k1DBuffer:
            test_configs.emplace_back(
                "1D buffer -> 1D buffer", CL_MEM_OBJECT_IMAGE1D_BUFFER,
                CL_MEM_READ_ONLY, CL_MEM_OBJECT_IMAGE1D_BUFFER,
                CL_MEM_READ_ONLY);
            test_fn = test_copy_image_set_1D_buffer;
            break;
        case k1DTo1DBuffer:
            test_configs.emplace_back(
                "1D -> 1D buffer", CL_MEM_OBJECT_IMAGE1D,
                gEnablePitch ? CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY
                             : CL_MEM_READ_ONLY,
                CL_MEM_OBJECT_IMAGE1D_BUFFER, CL_MEM_READ_ONLY);
            test_fn = test_copy_image_set_1D_1D_buffer;
            break;
        case k1DBufferTo1D:
            test_configs.emplace_back(
                "1D buffer -> 1D", CL_MEM_OBJECT_IMAGE1D_BUFFER,
                CL_MEM_READ_ONLY, CL_MEM_OBJECT_IMAGE1D,
                gEnablePitch ? CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY
                             : CL_MEM_READ_ONLY);
            test_fn = test_copy_image_set_1D_1D_buffer;
            break;
    }


    int ret = 0;
    for (const auto &test_config : test_configs)
    {
        if (gTestMipmaps)
            log_info("Running mipmapped %s tests...\n", test_config.name);
        else
            log_info("Running %s tests...\n", test_config.name);

        // Grab the list of supported image formats for integer reads
        std::vector<cl_image_format> formatList;
        {
            std::vector<cl_image_format> srcFormatList;
            std::vector<cl_image_format> dstFormatList;
            if (get_format_list(context, test_config.src_type, srcFormatList,
                                test_config.src_flags))
                return TEST_FAIL;
            if (get_format_list(context, test_config.dst_type, dstFormatList,
                                test_config.dst_flags))
                return TEST_FAIL;

            for (auto src_format : srcFormatList)
            {
                const bool src_format_in_dst =
                    std::find_if(dstFormatList.begin(), dstFormatList.end(),
                                 [src_format](cl_image_format fmt) {
                                     return src_format.image_channel_data_type
                                         == fmt.image_channel_data_type
                                         && src_format.image_channel_order
                                         == fmt.image_channel_order;
                                 })
                    != dstFormatList.end();
                if (src_format_in_dst)
                {
                    formatList.push_back(src_format);
                }
            }
        }

        std::vector<bool> filterFlags(formatList.size(), false);
        filter_formats(formatList, filterFlags, nullptr);

        // Run the format list
        for (unsigned int i = 0; i < formatList.size(); i++)
        {
            int test_return = 0;
            if (filterFlags[i])
            {
                continue;
            }

            print_header(&formatList[i], false);

            gTestCount++;

            test_return = test_fn(device, context, queue, test_config.src_flags,
                                  test_config.src_type, test_config.dst_flags,
                                  test_config.dst_type, &formatList[i]);

            if (test_return)
            {
                gFailCount++;
                log_error("FAILED: ");
                print_header(&formatList[i], true);
                log_info("\n");
            }

            ret += test_return;
        }
    }

    return ret;
}

int test_image_set( cl_device_id device, cl_context context, cl_command_queue queue, MethodsToTest testMethod )
{
    return test_image_type(device, context, queue, testMethod);
}




