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

extern int test_copy_image_set_1D( cl_device_id device, cl_context context, cl_command_queue queue, cl_image_format *format );
extern int test_copy_image_set_2D( cl_device_id device, cl_context context, cl_command_queue queue, cl_image_format *format );
extern int test_copy_image_set_3D( cl_device_id device, cl_context context, cl_command_queue queue, cl_image_format *format );
extern int test_copy_image_set_1D_array( cl_device_id device, cl_context context, cl_command_queue queue, cl_image_format *format );
extern int test_copy_image_set_2D_array( cl_device_id device, cl_context context, cl_command_queue queue, cl_image_format *format );
extern int test_copy_image_set_2D_3D( cl_device_id device, cl_context context, cl_command_queue queue, cl_image_format *format, bool reverse );
extern int test_copy_image_set_2D_2D_array( cl_device_id device, cl_context context, cl_command_queue queue, cl_image_format *format, bool reverse );
extern int test_copy_image_set_3D_2D_array( cl_device_id device, cl_context context, cl_command_queue queue, cl_image_format *format, bool reverse );
extern int test_copy_image_set_1D_buffer(cl_device_id device,
                                         cl_context context,
                                         cl_command_queue queue,
                                         cl_image_format *format);
extern int test_copy_image_set_1D_1D_buffer(cl_device_id device,
                                            cl_context context,
                                            cl_command_queue queue,
                                            cl_image_format *format);
extern int test_copy_image_set_1D_buffer_1D(cl_device_id device,
                                            cl_context context,
                                            cl_command_queue queue,
                                            cl_image_format *format);

int test_image_type( cl_device_id device, cl_context context, cl_command_queue queue, MethodsToTest testMethod, cl_mem_flags flags )
{
    const char *name;
    cl_mem_object_type imageType;

    if ( gTestMipmaps )
    {
        if ( 0 == is_extension_available( device, "cl_khr_mipmap_image" ))
        {
            log_info( "-----------------------------------------------------\n" );
            log_info( "This device does not support cl_khr_mipmap_image.\nSkipping mipmapped image test. \n" );
            log_info( "-----------------------------------------------------\n\n" );
            return 0;
        }
    }

    switch (testMethod)
    {
        case k1D:
            name = "1D -> 1D";
            imageType = CL_MEM_OBJECT_IMAGE1D;
            break;
        case k2D:
            name = "2D -> 2D";
            imageType = CL_MEM_OBJECT_IMAGE2D;
            break;
        case k3D:
            name = "3D -> 3D";
            imageType = CL_MEM_OBJECT_IMAGE3D;
            break;
        case k1DArray:
            name = "1D array -> 1D array";
            imageType = CL_MEM_OBJECT_IMAGE1D_ARRAY;
            break;
        case k2DArray:
            name = "2D array -> 2D array";
            imageType = CL_MEM_OBJECT_IMAGE2D_ARRAY;
            break;
        case k2DTo3D:
            name = "2D -> 3D";
            imageType = CL_MEM_OBJECT_IMAGE3D;
            break;
        case k3DTo2D:
            name = "3D -> 2D";
            imageType = CL_MEM_OBJECT_IMAGE3D;
            break;
        case k2DArrayTo2D:
            name = "2D array -> 2D";
            imageType = CL_MEM_OBJECT_IMAGE2D_ARRAY;
            break;
        case k2DTo2DArray:
            name = "2D -> 2D array";
            imageType = CL_MEM_OBJECT_IMAGE2D_ARRAY;
            break;
        case k2DArrayTo3D:
            name = "2D array -> 3D";
            imageType = CL_MEM_OBJECT_IMAGE3D;
            break;
        case k3DTo2DArray:
            name = "3D -> 2D array";
            imageType = CL_MEM_OBJECT_IMAGE3D;
            break;
        case k1DBuffer:
            name = "1D buffer -> 1D buffer";
            imageType = CL_MEM_OBJECT_IMAGE1D_BUFFER;
            break;
        case k1DTo1DBuffer:
            name = "1D -> 1D buffer";
            imageType = CL_MEM_OBJECT_IMAGE1D_BUFFER;
            break;
        case k1DBufferTo1D:
            name = "1D buffer -> 1D";
            imageType = CL_MEM_OBJECT_IMAGE1D_BUFFER;
            break;
    }

    if(gTestMipmaps)
        log_info( "Running mipmapped %s tests...\n", name );
    else
        log_info( "Running %s tests...\n", name );

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
            continue;
        }

        print_header( &formatList[ i ], false );

        gTestCount++;

        if( testMethod == k1D )
            test_return = test_copy_image_set_1D( device, context, queue, &formatList[ i ] );
        else if( testMethod == k2D )
            test_return = test_copy_image_set_2D( device, context, queue, &formatList[ i ] );
        else if( testMethod == k3D )
            test_return = test_copy_image_set_3D( device, context, queue,&formatList[ i ] );
        else if( testMethod == k1DArray )
            test_return = test_copy_image_set_1D_array( device, context, queue, &formatList[ i ] );
        else if( testMethod == k2DArray )
            test_return = test_copy_image_set_2D_array( device, context, queue, &formatList[ i ] );
        else if( testMethod == k2DTo3D )
            test_return = test_copy_image_set_2D_3D( device, context, queue, &formatList[ i ], false );
        else if( testMethod == k3DTo2D )
            test_return = test_copy_image_set_2D_3D( device, context, queue, &formatList[ i ], true );
        else if( testMethod == k2DArrayTo2D)
            test_return = test_copy_image_set_2D_2D_array( device, context, queue, &formatList[ i ], true);
        else if( testMethod == k2DTo2DArray)
            test_return = test_copy_image_set_2D_2D_array( device, context, queue, &formatList[ i ], false);
        else if( testMethod == k2DArrayTo3D)
            test_return = test_copy_image_set_3D_2D_array( device, context, queue, &formatList[ i ], true);
        else if( testMethod == k3DTo2DArray)
            test_return = test_copy_image_set_3D_2D_array( device, context, queue, &formatList[ i ], false);
        else if (testMethod == k1DBuffer)
            test_return = test_copy_image_set_1D_buffer(device, context, queue,
                                                        &formatList[i]);
        else if (testMethod == k1DBufferTo1D)
            test_return = test_copy_image_set_1D_buffer_1D(
                device, context, queue, &formatList[i]);
        else if (testMethod == k1DTo1DBuffer)
            test_return = test_copy_image_set_1D_1D_buffer(
                device, context, queue, &formatList[i]);


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

int test_image_set( cl_device_id device, cl_context context, cl_command_queue queue, MethodsToTest testMethod )
{
    int ret = 0;

    ret += test_image_type( device, context, queue, testMethod, CL_MEM_READ_ONLY );

    return ret;
}




