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


extern int test_get_image_info_1D(cl_device_id device, cl_context context,
                                  cl_command_queue queue,
                                  cl_image_format *format, cl_mem_flags flags);
extern int test_get_image_info_2D(cl_device_id device, cl_context context,
                                  cl_command_queue queue,
                                  cl_image_format *format, cl_mem_flags flags);
extern int test_get_image_info_3D(cl_device_id device, cl_context context,
                                  cl_command_queue queue,
                                  cl_image_format *format, cl_mem_flags flags);
extern int test_get_image_info_1D_array(cl_device_id device, cl_context context,
                                        cl_command_queue queue,
                                        cl_image_format *format,
                                        cl_mem_flags flags);
extern int test_get_image_info_2D_array(cl_device_id device, cl_context context,
                                        cl_command_queue queue,
                                        cl_image_format *format,
                                        cl_mem_flags flags);

int test_image_type( cl_device_id device, cl_context context, cl_command_queue queue, cl_mem_object_type imageType, cl_mem_flags flags )
{
    log_info( "Running %s %s-only tests...\n", convert_image_type_to_string(imageType), flags == CL_MEM_READ_ONLY ? "read" : "write" );

    int ret = 0;

    // Grab the list of supported image formats for integer reads
    std::vector<cl_image_format> formatList;
    if (get_format_list(context, imageType, formatList, flags)) return -1;

    std::vector<bool> filterFlags(formatList.size(), false);
    filter_formats(formatList, filterFlags, nullptr);

    // Run the format list
    for (unsigned int i = 0; i < formatList.size(); i++)
    {
        int test_return = 0;
        if( filterFlags[i] )
        {
            log_info( "NOT RUNNING: " );
            print_header( &formatList[ i ], false );
            continue;
        }

        print_header( &formatList[ i ], false );

        gTestCount++;

        switch (imageType) {
            case CL_MEM_OBJECT_IMAGE1D:
                test_return = test_get_image_info_1D(device, context, queue,
                                                     &formatList[i], flags);
                break;
            case CL_MEM_OBJECT_IMAGE2D:
                test_return = test_get_image_info_2D(device, context, queue,
                                                     &formatList[i], flags);
                break;
            case CL_MEM_OBJECT_IMAGE3D:
                test_return = test_get_image_info_3D(device, context, queue,
                                                     &formatList[i], flags);
                break;
            case CL_MEM_OBJECT_IMAGE1D_ARRAY:
                test_return = test_get_image_info_1D_array(
                    device, context, queue, &formatList[i], flags);
                break;
            case CL_MEM_OBJECT_IMAGE2D_ARRAY:
                test_return = test_get_image_info_2D_array(
                    device, context, queue, &formatList[i], flags);
                break;
        }

        if (test_return) {
            gFailCount++;
            log_error( "FAILED: " );
            print_header( &formatList[ i ], true );
            log_info( "\n" );
        }

        ret += test_return;
    }

    return ret;
}

int test_image_set( cl_device_id device, cl_context context, cl_command_queue queue, cl_mem_object_type imageType )
{
    int version_check;
    auto version = get_device_cl_version(device);

    if ((version_check = (version < Version(1, 2))))
    {
        switch (imageType) {
            case CL_MEM_OBJECT_IMAGE1D:
            test_missing_feature(version_check, "image_1D");
            case CL_MEM_OBJECT_IMAGE1D_ARRAY:
            test_missing_feature(version_check, "image_1D_array");
            case CL_MEM_OBJECT_IMAGE2D_ARRAY:
            test_missing_feature(version_check, "image_2D_array");
    }
    }

  int ret = 0;
    ret += test_image_type( device, context, queue, imageType, CL_MEM_READ_ONLY );
    ret += test_image_type( device, context, queue, imageType, CL_MEM_WRITE_ONLY );

    return ret;
}
