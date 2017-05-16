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
#include "procs.h"
#include <ctype.h>

static const char *linear_ids_source[1] = {
"__kernel void test_linear_ids(__global int2 *out)\n"
"{\n"
"    size_t lid, gid;\n"
"    uint d = get_work_dim();\n"
"    if (d == 1U) {\n"
"        gid = get_global_id(0) - get_global_offset(0);\n"
"        lid = get_local_id(0);\n"
"    } else if (d == 2U) {\n"
"        gid = (get_global_id(1) - get_global_offset(1)) * get_global_size(0) +\n"
"              (get_global_id(0) - get_global_offset(0));\n"
"        lid = get_local_id(1) * get_local_size(0) + get_local_id(0);\n"
"    } else {\n"
"        gid = ((get_global_id(2) - get_global_offset(2)) * get_global_size(1) +\n"
"               (get_global_id(1) - get_global_offset(1))) * get_global_size(0) +\n"
"               (get_global_id(0) - get_global_offset(0));\n"
"        lid = (get_local_id(2) * get_local_size(1) +\n"
"               get_local_id(1)) * get_local_size(0) + get_local_id(0);\n"
"    }\n"
"    out[gid].x = gid == get_global_linear_id();\n"
"    out[gid].y = lid == get_local_linear_id();\n"
"}\n"
};

#define NUM_ITER 12
#define MAX_1D 4096
#define MAX_2D 64
#define MAX_3D 16
#define MAX_OFFSET 100000

int
test_get_linear_ids(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper outbuf;
    int error, iter, i, j, k;
    size_t lws[3], gws[3], gwo[3];
    cl_uint dims;
    cl_int outmem[2*MAX_1D], *om;


    // Create the kernel
    error = create_single_kernel_helper_with_build_options(context, &program, &kernel, 1, linear_ids_source, "test_linear_ids", "-cl-std=CL2.0");
    if (error)
        return error;

    // Create the out buffer
    outbuf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(outmem), NULL, &error);
    test_error(error, "failed to create result buffer\n");

    // This will leak if there is an error, but this is what is done everywhere else
    MTdata seed = init_genrand(gRandomSeed);

    // Run some tests
    for (iter=0; iter<NUM_ITER; ++iter) {
        dims = iter % 3 + 1;

        switch (dims) {
        case 1:
            gwo[0] = random_in_range(0, MAX_OFFSET, seed);
            gws[0] = random_in_range(MAX_1D/8, MAX_1D/4, seed)*4;
            error = get_max_common_work_group_size(context, kernel, gws[0], lws);
            break;
        case 2:
            gwo[0] = random_in_range(0, MAX_OFFSET, seed);
            gwo[1] = random_in_range(0, MAX_OFFSET, seed);
            gws[0] = random_in_range(MAX_2D/8, MAX_2D/4, seed)*4;
            gws[1] = random_in_range(MAX_2D/8, MAX_2D/4, seed)*4;
            error = get_max_common_2D_work_group_size(context, kernel, gws, lws);
            break;
        case 3:
            gwo[0] = random_in_range(0, MAX_OFFSET, seed);
            gwo[1] = random_in_range(0, MAX_OFFSET, seed);
            gwo[2] = random_in_range(0, MAX_OFFSET, seed);
            gws[0] = random_in_range(MAX_3D/4, MAX_3D/2, seed)*2;
            gws[1] = random_in_range(MAX_3D/4, MAX_3D/2, seed)*2;
            gws[2] = random_in_range(MAX_3D/4, MAX_3D/2, seed)*2;
            error = get_max_common_3D_work_group_size(context, kernel, gws, lws);
            break;
        }

        test_error(error, "Failed to determine local work size\n");


        switch (dims) {
        case 1:
            log_info("  testing offset=%u global=%u local=%u...\n", gwo[0], gws[0], lws[0]);
            break;
        case 2:
            log_info("  testing offset=(%u,%u) global=(%u,%u) local=(%u,%u)...\n",
                    gwo[0], gwo[1], gws[0], gws[1], lws[0], lws[1]);
            break;
        case 3:
            log_info("  testing offset=(%u,%u,%u) global=(%u,%u,%u) local=(%u,%u,%u)...\n",
                    gwo[0], gwo[1], gwo[2], gws[0], gws[1], gws[2], lws[0], lws[1], lws[2]);
            break;
        }

        // Set up and run
        memset(outmem, 0, sizeof(outmem));

        error = clSetKernelArg(kernel, 0, sizeof(outbuf), (void *)&outbuf);
        test_error(error, "clSetKernelArg failed\n");

        error = clEnqueueWriteBuffer(queue, outbuf, CL_FALSE, 0, sizeof(outmem), (void *)outmem, 0, NULL, NULL);
        test_error(error, "clEnqueueWriteBuffer failed\n");

        error = clEnqueueNDRangeKernel(queue, kernel, dims, gwo, gws, lws, 0, NULL, NULL);
        test_error(error, "clEnqueueNDRangeKernel failed\n");

        error = clEnqueueReadBuffer(queue, outbuf, CL_FALSE, 0, sizeof(outmem), (void *)outmem, 0, NULL, NULL);
        test_error(error, "clEnqueueReadBuffer failed\n");

        error = clFinish(queue);
        test_error(error, "clFinish failed\n");

        // Check the return
        switch (dims) {
        case 1:
            for (i=0, om=outmem; i<(int)gws[0]; ++i, om+=2) {
                if (om[0] != 1) {
                    log_error("get_global_linear_id() failed at %d\n", i);
                    return -1;
                }
                if (om[1] != 1) {
                    log_error("get_local_linear_id() failed at (%d, %d)\n", i % (int)lws[0], i / (int)lws[0]);
                    return -1;
                }
            }
            break;
        case 2:
            for (j=0, om=outmem; j<gws[1]; ++j) {
                for (i=0; i<gws[0]; ++i, om+=2) {
                    if (om[0] != 1) {
                        log_error("get_global_linear_id() failed at (%d,%d)\n", i, j);
                        return -1;
                    }
                    if (om[1] != 1) {
                        log_error("get_local_linear_id() failed at (%d, %d), (%d, %d)\n",
                                i % (int)lws[0], j % (int)lws[1],
                                i / (int)lws[0], j / (int)lws[1]);
                        return -1;
                    }
                }
            }
            break;
        case 3:
            for (k=0, om=outmem; k<gws[2]; ++k) {
                for (j=0; j<gws[1]; ++j) {
                    for (i=0; i<gws[0]; ++i, om+=2) {
                        if (om[0] != 1) {
                            log_error("get_global_linear_id() failed at (%d,%d, %d)\n", i, j, k);
                            return -1;
                        }
                        if (om[1] != 1) {
                            log_error("get_local_linear_id() failed at (%d, %d), (%d, %d), (%d, %d)\n",
                                    i % (int)lws[0], j % (int)lws[1], k % (int)lws[2],
                                    i / (int)lws[0], j / (int)lws[1], k / (int)lws[2]);
                            return -1;
                        }
                    }
                }
            }
            break;
        }

    }

    free_mtdata(seed);
    return 0;
}

