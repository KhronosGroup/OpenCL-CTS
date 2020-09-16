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
#include "harness/compat.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>


#include "procs.h"

static const char *rgba8888_kernel_code =
"\n"
"__kernel void test_rgba8888(read_only image2d_t srcimg, write_only image2d_t dstimg, sampler_t sampler)\n"
"{\n"
"    int    tid_x = get_global_id(0);\n"
"    int    tid_y = get_global_id(1);\n"
"    float4 color;\n"
"\n"
"    if ( (tid_x >= get_image_width(dstimg)) || (tid_y >= get_image_height(dstimg)) )\n"
"        return;\n"
"    color = read_imagef(srcimg, sampler, (int2)(tid_x, tid_y));\n"
"    write_imagef(dstimg, (int2)(tid_x, tid_y), color);\n"
"\n"
"}\n";


static unsigned char *
generate_8888_image(int w, int h, MTdata d)
{
    unsigned char   *ptr = (unsigned char*)malloc(w * h * 4);
    int             i;

    for (i=0; i<w*h*4; i++)
        ptr[i] = (unsigned char)genrand_int32(d);

    return ptr;
}

static int
verify_rgba8888_image(unsigned char *src, unsigned char *dst, int w, int h)
{
    int     i;

    for (i=0; i<w*h*4; i++)
    {
        if (dst[i] != src[i])
        {
            log_error("NPOT_IMAGE_RGBA_UNORM_INT8 test for width = %d, height = %d failed\n", w, h);
            return -1;
        }
    }

    log_info("NPOT_IMAGE_RGBA_UNORM_INT8 test for width = %d, height = %d passed\n", w, h);
    return 0;
}


int    img_width_selection[] = { 97, 111, 322, 479 };
int    img_height_selection[] = { 149, 222, 754, 385 };

int
test_imagenpot(cl_device_id device_id, cl_context context, cl_command_queue queue, int num_elements)
{
    cl_mem            streams[2];
    cl_image_format    img_format;
    unsigned char    *input_ptr, *output_ptr;
    cl_program        program;
    cl_kernel        kernel;
    size_t    global_threads[3], local_threads[3];
    size_t            local_workgroup_size;
    int                img_width;
    int                img_height;
    int                err;
    cl_uint            m;
    size_t max_local_workgroup_size[3];
    MTdata          d;

    PASSIVE_REQUIRE_IMAGE_SUPPORT( device_id )

    cl_device_type device_type;
    err = clGetDeviceInfo(device_id, CL_DEVICE_TYPE, sizeof(device_type), &device_type, NULL);
    if (err) {
        log_error("Failed to get device type: %d\n",err);
        return -1;
    }

    d = init_genrand( gRandomSeed );
    for (m=0; m<sizeof(img_width_selection)/sizeof(int); m++)
    {
        img_width = img_width_selection[m];
        img_height = img_height_selection[m];
        input_ptr = generate_8888_image(img_width, img_height, d);
        output_ptr = (unsigned char*)malloc(sizeof(unsigned char) * 4 * img_width * img_height);

        img_format.image_channel_order = CL_RGBA;
        img_format.image_channel_data_type = CL_UNORM_INT8;
        streams[0] = create_image_2d(context, (cl_mem_flags)(CL_MEM_READ_WRITE),  &img_format,
                                 img_width, img_height, 0, NULL, NULL);
        if (!streams[0])
        {
            log_error("create_image_2d failed\n");
            free_mtdata(d);
            return -1;
        }
        img_format.image_channel_order = CL_RGBA;
        img_format.image_channel_data_type = CL_UNORM_INT8;
    streams[1] = create_image_2d(context, (cl_mem_flags)(CL_MEM_READ_WRITE),  &img_format,
                                 img_width, img_height, 0, NULL, NULL);
        if (!streams[1])
        {
            log_error("create_image_2d failed\n");
            free_mtdata(d);
            return -1;
        }

        size_t origin[3] = {0,0,0}, region[3] = {img_width, img_height, 1};
        err = clEnqueueWriteImage(queue, streams[0], CL_TRUE,
                              origin, region, 0, 0,
                              input_ptr,
                              0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            log_error("clWriteImage failed\n");
            free_mtdata(d);
            return -1;
        }


        err = create_single_kernel_helper(context, &program, &kernel, 1, &rgba8888_kernel_code, "test_rgba8888" );
        if (err)
        {
            log_error("Failed to create kernel and program: %d\n", err);
            free_mtdata(d);
            return -1;
        }

        cl_sampler sampler = clCreateSampler(context, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST, &err);
        test_error(err, "clCreateSampler failed");

        err = clSetKernelArg(kernel, 0, sizeof streams[0], &streams[0]);
        err |= clSetKernelArg(kernel, 1, sizeof streams[1], &streams[1]);
        err |= clSetKernelArg(kernel, 2, sizeof sampler, &sampler);
        if (err != CL_SUCCESS)
        {
            log_error("clSetKernelArgs failed\n");
            free_mtdata(d);
            return -1;
        }

        err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local_workgroup_size), &local_workgroup_size, NULL);
        test_error(err, "clGetKernelWorkGroupInfo for CL_KERNEL_WORK_GROUP_SIZE failed");

        err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_local_workgroup_size), max_local_workgroup_size, NULL);
        test_error(err, "clGetDeviceInfo failed for CL_DEVICE_MAX_WORK_ITEM_SIZES");

        // Pick the minimum of the device and the kernel
        if (local_workgroup_size > max_local_workgroup_size[0])
            local_workgroup_size = max_local_workgroup_size[0];

        global_threads[0] = ((img_width + local_workgroup_size - 1) / local_workgroup_size) * local_workgroup_size;
        global_threads[1] = img_height;
        local_threads[0] = local_workgroup_size;
        local_threads[1] = 1;
        err = clEnqueueNDRangeKernel( queue, kernel, 2, NULL, global_threads, local_threads, 0, NULL, NULL );

        if (err != CL_SUCCESS)
        {
            log_error("%s clEnqueueNDRangeKernel failed\n", __FUNCTION__);
            free_mtdata(d);
            return -1;
        }
        err = clEnqueueReadImage(queue, streams[1], CL_TRUE,
                             origin, region, 0, 0,
                             (void *)output_ptr,
                             0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            log_error("clEnqueueReadBuffer failed\n");
            return -1;
        }

        err = verify_rgba8888_image(input_ptr, output_ptr, img_width, img_height);

        // cleanup
        clReleaseSampler(sampler);
        clReleaseMemObject(streams[0]);
        clReleaseMemObject(streams[1]);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        free(input_ptr);
        free(output_ptr);

        if (err)
            break;
    }

    free_mtdata(d);

    return err;
}





