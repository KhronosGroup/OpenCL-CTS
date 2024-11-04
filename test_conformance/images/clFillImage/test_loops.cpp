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

extern int gTypesToTest;

extern int test_fill_image_set_1D( cl_device_id device, cl_context context, cl_command_queue queue, cl_image_format *format, ExplicitType outputType );
extern int test_fill_image_set_2D( cl_device_id device, cl_context context, cl_command_queue queue, cl_image_format *format, ExplicitType outputType );
extern int test_fill_image_set_3D( cl_device_id device, cl_context context, cl_command_queue queue, cl_image_format *format, ExplicitType outputType );
extern int test_fill_image_set_1D_array( cl_device_id device, cl_context context, cl_command_queue queue, cl_image_format *format, ExplicitType outputType );
extern int test_fill_image_set_2D_array( cl_device_id device, cl_context context, cl_command_queue queue, cl_image_format *format, ExplicitType outputType );
extern int test_fill_image_set_1D_buffer(cl_device_id device,
                                         cl_context context,
                                         cl_command_queue queue,
                                         cl_image_format *format,
                                         ExplicitType outputType);
typedef int (*test_func)(cl_device_id device, cl_context context,
                         cl_command_queue queue, cl_image_format *format,
                         ExplicitType outputType);

int test_image_type( cl_device_id device, cl_context context, cl_command_queue queue, MethodsToTest testMethod, cl_mem_flags flags )
{
    const char *name;
    cl_mem_object_type imageType;
    test_func test_fn;

    switch (testMethod)
    {
        case k1D:
            name = "1D Image Fill";
            imageType = CL_MEM_OBJECT_IMAGE1D;
            test_fn = &test_fill_image_set_1D;
            break;
        case k2D:
            name = "2D Image Fill";
            imageType = CL_MEM_OBJECT_IMAGE2D;
            test_fn = &test_fill_image_set_2D;
            break;
        case k1DArray:
            name = "1D Image Array Fill";
            imageType = CL_MEM_OBJECT_IMAGE1D_ARRAY;
            test_fn = &test_fill_image_set_1D_array;
            break;
        case k2DArray:
            name = "2D Image Array Fill";
            imageType = CL_MEM_OBJECT_IMAGE2D_ARRAY;
            test_fn = &test_fill_image_set_2D_array;
            break;
        case k3D:
            name = "3D Image Fill";
            imageType = CL_MEM_OBJECT_IMAGE3D;
            test_fn = &test_fill_image_set_3D;
            break;
        case k1DBuffer:
            name = "1D Image Buffer Fill";
            imageType = CL_MEM_OBJECT_IMAGE1D_BUFFER;
            test_fn = &test_fill_image_set_1D_buffer;
            break;
        default: log_error("Unhandled method\n"); return -1;
    }

    log_info( "Running %s tests...\n", name );

    int ret = 0;

    // Grab the list of supported image formats
    std::vector<cl_image_format> formatList;
    if (get_format_list(context, imageType, formatList, flags)) return -1;

    for (auto test : imageTestTypes)
    {
        if (gTypesToTest & test.type)
        {
            std::vector<bool> filterFlags(formatList.size(), false);
            if (filter_formats(formatList, filterFlags, test.channelTypes) == 0)
            {
                log_info("No formats supported for %s type\n", test.name);
            }
            else
            {
                // Run the format list
                for (unsigned int i = 0; i < formatList.size(); i++)
                {
                    if (filterFlags[i])
                    {
                        continue;
                    }

                    print_header(&formatList[i], false);

                    gTestCount++;

                    int test_return =
                        test_fn(device, context, queue, &formatList[i],
                                test.explicitType);
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
        }
    }

    return ret;
}


int test_image_set( cl_device_id device, cl_context context, cl_command_queue queue, MethodsToTest testMethod )
{
    int ret = 0;

    ret += test_image_type( device, context, queue, testMethod, CL_MEM_READ_ONLY );

    return ret;
}
