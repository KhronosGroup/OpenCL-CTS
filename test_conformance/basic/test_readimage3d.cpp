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
#include "harness/imageHelpers.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "procs.h"

static const char *bgra8888_kernel_code =
"\n"
"__kernel void test_bgra8888(read_only image3d_t srcimg, __global float4 *dst, sampler_t sampler)\n"
"{\n"
"    int    tid_x = get_global_id(0);\n"
"    int    tid_y = get_global_id(1);\n"
"    int    tid_z = get_global_id(2);\n"
"    int    indx = (tid_z * get_image_height(srcimg) + tid_y) * get_image_width(srcimg) + tid_x;\n"
"    float4 color;\n"
"\n"
"    color = read_imagef(srcimg, sampler, (int4)(tid_x, tid_y, tid_z, 0));\n"
"    dst[indx].x = color.z;\n"
"    dst[indx].y = color.y;\n"
"    dst[indx].z = color.x;\n"
"    dst[indx].w = color.w;\n"
"\n"
"}\n";

static const char *rgba8888_kernel_code =
"\n"
"__kernel void test_rgba8888(read_only image3d_t srcimg, __global float4 *dst, sampler_t sampler)\n"
"{\n"
"    int    tid_x = get_global_id(0);\n"
"    int    tid_y = get_global_id(1);\n"
"    int    tid_z = get_global_id(2);\n"
"    int    indx = (tid_z * get_image_height(srcimg) + tid_y) * get_image_width(srcimg) + tid_x;\n"
"    float4 color;\n"
"\n"
"    color = read_imagef(srcimg, sampler, (int4)(tid_x, tid_y, tid_z, 0));\n"
"    //indx *= 4;\n"
"    dst[indx].x = color.x;\n"
"    dst[indx].y = color.y;\n"
"    dst[indx].z = color.z;\n"
"    dst[indx].w = color.w;\n"
"\n"
"}\n";

static unsigned char *
generate_3d_image8(int w, int h, int d, MTdata data)
{
    unsigned char   *ptr = (unsigned char*)malloc(w * h * d * 4);
    int             i;

    for (i=0; i<w*h*d*4; i++)
        ptr[i] = (unsigned char)genrand_int32(data);

    return ptr;
}

static int
verify_3d_image8(double *image, float *outptr, int w, int h, int d)
{
    int     i;

    for (i=0; i<w*h*d*4; i++)
    {
        if (outptr[i] != (float)image[i])
        {
            float ulps = Ulp_Error( outptr[i], image[i]);

            if(! (fabsf(ulps) < 1.5f) )
            {
                log_error( "ERROR: Data sample %d does not validate! Expected (%a), got (%a), ulp %f\n",
                    (int)i, image[i], outptr[ i ],  ulps );
                return -1;
            }
        }
    }

    return 0;
}

static double *
prepare_reference(unsigned char * input_ptr, int w, int h, int d)
{
    double   *ptr = (double*)malloc(w * h * d * 4 * sizeof(double));
    int         i;
    for (i=0; i<w*h*d*4; i++)
        ptr[i] = ((double)input_ptr[i]/255);

    return ptr;
}

int test_readimage3d(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
	cl_mem streams[2];
	cl_program program;
	cl_kernel kernel;
	cl_sampler sampler;
	struct testFormat
	{
		const char* kernelName;
		const char* kernelSourceString;
		const cl_image_format img_format;
	};

	static testFormat formatsToTest[] =
	{
		{
			"test_bgra8888",
			bgra8888_kernel_code,
			{CL_BGRA, CL_UNORM_INT8},
		},
		{
			"test_rgba8888",
			rgba8888_kernel_code,
			{CL_RGBA, CL_UNORM_INT8},
		},
	};

	unsigned char *input_ptr;
	float *output_ptr;
	double *ref_ptr;
	size_t threads[3];
	int img_width = 64;
	int img_height = 64;
	int img_depth = 64;
	int err;
	size_t origin[3] = {0, 0, 0};
	size_t region[3] = {img_width, img_height, img_depth};
	size_t length = img_width * img_height * img_depth * 4 * sizeof(float);

	PASSIVE_REQUIRE_3D_IMAGE_SUPPORT( device )

	for (uint32_t i = 0; i < ARRAY_SIZE(formatsToTest); i++)
	{
		if (!is_image_format_required(formatsToTest[i].img_format, CL_MEM_READ_ONLY, CL_MEM_OBJECT_IMAGE3D, device))
			continue;

		MTdata d = init_genrand( gRandomSeed );
		input_ptr = generate_3d_image8(img_width, img_height, img_depth, d);
		ref_ptr = prepare_reference(input_ptr, img_width, img_height, img_depth);
		output_ptr = (float*)malloc(length);

		streams[0] = create_image_3d(context, CL_MEM_READ_ONLY, &formatsToTest[i].img_format, img_width, img_height, img_depth, 0, 0, NULL, &err);
		test_error(err, "create_image_3d failed");

		streams[1] = clCreateBuffer(context, CL_MEM_READ_WRITE, length, NULL, &err);
		test_error(err, "clCreateBuffer failed");

		sampler = clCreateSampler(context, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST, &err);
		test_error(err, "clCreateSampler failed");

		err = clEnqueueWriteImage(queue, streams[0], CL_TRUE, origin, region, 0, 0, input_ptr, 0, NULL, NULL);
		test_error(err, "clEnqueueWriteImage failed");

		err = create_single_kernel_helper(context, &program, &kernel, 1, &formatsToTest[i].kernelSourceString, formatsToTest[i].kernelName);
		test_error(err, "create_single_kernel_helper failed");

		err = clSetKernelArg(kernel, 0, sizeof streams[0], &streams[0]);
		err |= clSetKernelArg(kernel, 1, sizeof streams[1], &streams[1]);
		err |= clSetKernelArg(kernel, 2, sizeof sampler, &sampler);
		test_error(err, "clSetKernelArg failed");

		threads[0] = (unsigned int)img_width;
		threads[1] = (unsigned int)img_height;
		threads[2] = (unsigned int)img_depth;

		err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, threads, NULL, 0, NULL, NULL);
		test_error(err, "clEnqueueNDRangeKernel failed");

		err = clEnqueueReadBuffer(queue, streams[1], CL_TRUE, 0, length, output_ptr, 0, NULL, NULL);
		test_error(err, "clEnqueueReadBuffer failed");

		err = verify_3d_image8(ref_ptr, output_ptr, img_width, img_height, img_depth);
		if ( err == 0 )
		{
			log_info("READ_IMAGE3D_%s_%s test passed\n",
			         GetChannelTypeName(formatsToTest[i].img_format.image_channel_data_type),
			         GetChannelOrderName(formatsToTest[i].img_format.image_channel_order));
		}

		clReleaseSampler(sampler);
		clReleaseMemObject(streams[0]);
		clReleaseMemObject(streams[1]);
		clReleaseKernel(kernel);
		clReleaseProgram(program);
		free_mtdata(d);
		d = NULL;
		free(input_ptr);
		free(ref_ptr);
		free(output_ptr);
	}

	return err;
}
