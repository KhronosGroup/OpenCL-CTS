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
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "procs.h"

#ifndef M_PI
#define M_PI    3.14159265358979323846264338327950288
#endif

#define CLAMP_KERNEL( type )                        \
    const char *clamp_##type##_kernel_code =                \
    EMIT_PRAGMA_DIRECTIVE                        \
    "__kernel void test_clamp(__global " #type " *x, __global " #type " *minval, __global " #type " *maxval, __global " #type " *dst)\n" \
    "{\n"                                \
    "    int  tid = get_global_id(0);\n"                \
    "\n"                                \
    "    dst[tid] = clamp(x[tid], minval[tid], maxval[tid]);\n"    \
    "}\n";

#define CLAMP_KERNEL_V( type, size)                    \
    const char *clamp_##type##size##_kernel_code =            \
    EMIT_PRAGMA_DIRECTIVE                        \
    "__kernel void test_clamp(__global " #type #size " *x, __global " #type #size " *minval, __global " #type #size " *maxval, __global " #type #size " *dst)\n" \
    "{\n"                                \
    "    int  tid = get_global_id(0);\n"                \
    "\n"                                \
    "    dst[tid] = clamp(x[tid], minval[tid], maxval[tid]);\n"    \
    "}\n";

#define CLAMP_KERNEL_V3( type, size)                    \
    const char *clamp_##type##size##_kernel_code =            \
    EMIT_PRAGMA_DIRECTIVE                        \
    "__kernel void test_clamp(__global " #type " *x, __global " #type " *minval, __global " #type " *maxval, __global " #type " *dst)\n" \
    "{\n"                                \
    "    int  tid = get_global_id(0);\n"                \
    "\n"                                \
    "    vstore3(clamp(vload3(tid, x), vload3(tid,minval), vload3(tid,maxval)), tid, dst);\n"    \
    "}\n";

#define EMIT_PRAGMA_DIRECTIVE " "
CLAMP_KERNEL( float )
CLAMP_KERNEL_V( float, 2 )
CLAMP_KERNEL_V( float, 4 )
CLAMP_KERNEL_V( float, 8 )
CLAMP_KERNEL_V( float, 16 )
CLAMP_KERNEL_V3( float, 3)
#undef EMIT_PRAGMA_DIRECTIVE

#define EMIT_PRAGMA_DIRECTIVE "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
CLAMP_KERNEL( double )
CLAMP_KERNEL_V( double, 2 )
CLAMP_KERNEL_V( double, 4 )
CLAMP_KERNEL_V( double, 8 )
CLAMP_KERNEL_V( double, 16 )
CLAMP_KERNEL_V3( double, 3 )
#undef EMIT_PRAGMA_DIRECTIVE

const char *clamp_float_codes[] = { clamp_float_kernel_code, clamp_float2_kernel_code, clamp_float4_kernel_code, clamp_float8_kernel_code, clamp_float16_kernel_code, clamp_float3_kernel_code };
const char *clamp_double_codes[] = { clamp_double_kernel_code, clamp_double2_kernel_code, clamp_double4_kernel_code, clamp_double8_kernel_code, clamp_double16_kernel_code, clamp_double3_kernel_code };

static int verify_clamp(float *x, float *minval, float *maxval, float *outptr, int n)
{
    float       t;
    int         i;

    for (i=0; i<n; i++)
    {
        t = fminf( fmaxf( x[ i ], minval[ i ] ), maxval[ i ] );
        if (t != outptr[i])
        {
            log_error( "%d) verification error: clamp( %a, %a, %a) = *%a vs. %a\n", i, x[i], minval[i], maxval[i], t, outptr[i] );
            return -1;
        }
    }

    return 0;
}

static int verify_clamp_double(double *x, double *minval, double *maxval, double *outptr, int n)
{
    double       t;
    int         i;

    for (i=0; i<n; i++)
    {
        t = fmin( fmax( x[ i ], minval[ i ] ), maxval[ i ] );
        if (t != outptr[i])
        {
            log_error( "%d) verification error: clamp( %a, %a, %a) = *%a vs. %a\n", i, x[i], minval[i], maxval[i], t, outptr[i] );
            return -1;
        }
    }

    return 0;
}

int
test_clamp(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
    cl_mem      streams[8];
    cl_float      *input_ptr[3], *output_ptr;
    cl_double     *input_ptr_double[3], *output_ptr_double = NULL;
    cl_program  *program;
    cl_kernel   *kernel;
    size_t threads[1];
    int num_elements;
    int err;
    int i, j;
    MTdata d;

    program = (cl_program*)malloc(sizeof(cl_program)*kTotalVecCount*2);
    kernel = (cl_kernel*)malloc(sizeof(cl_kernel)*kTotalVecCount*2);

    num_elements = n_elems * (1 << (kVectorSizeCount-1));

    int test_double = 0;
    if(is_extension_available( device, "cl_khr_fp64" )) {
    log_info("Testing doubles.\n");
      test_double = 1;
    }


    // why does this go from 0 to 2?? -- Oh, I see, there are four function
    // arguments to the function, and 3 of them are inputs?
    for( i = 0; i < 3; i++ )
    {
        input_ptr[i] = (cl_float*)malloc(sizeof(cl_float) * num_elements);
        if (test_double) input_ptr_double[i] = (cl_double*)malloc(sizeof(cl_double) * num_elements);
    }
    output_ptr = (cl_float*)malloc(sizeof(cl_float) * num_elements);
    if (test_double) output_ptr_double = (cl_double*)malloc(sizeof(cl_double) * num_elements);

    // why does this go from 0 to 3?
    for( i = 0; i < 4; i++ )
    {
        streams[ i ] = clCreateBuffer( context, (cl_mem_flags)(CL_MEM_READ_WRITE),  sizeof(cl_float) * num_elements, NULL, NULL );
        if (!streams[0])
        {
            log_error("clCreateBuffer failed\n");
            return -1;
        }
    }
    if (test_double)
    for( i = 4; i < 8; i++ )
        {
        streams[ i ] = clCreateBuffer( context, (cl_mem_flags)(CL_MEM_READ_WRITE),  sizeof(cl_double) * num_elements, NULL, NULL );
        if (!streams[0])
            {
            log_error("clCreateBuffer failed\n");
            return -1;
            }
        }

    d = init_genrand( gRandomSeed );
    for( j = 0; j < num_elements; j++ )
    {
        input_ptr[0][j] = get_random_float(-0x20000000, 0x20000000, d);
        input_ptr[1][j] = get_random_float(-0x20000000, 0x20000000, d);
        input_ptr[2][j] = get_random_float(input_ptr[1][j], 0x20000000, d);

        if (test_double) {
        input_ptr_double[0][j] = get_random_double(-0x20000000, 0x20000000, d);
        input_ptr_double[1][j] = get_random_double(-0x20000000, 0x20000000, d);
        input_ptr_double[2][j] = get_random_double(input_ptr_double[1][j], 0x20000000, d);
        }
    }
    free_mtdata(d); d = NULL;

    for( i = 0; i < 3; i++ )
    {
        err = clEnqueueWriteBuffer( queue, streams[ i ], CL_TRUE, 0, sizeof( cl_float ) * num_elements, input_ptr[ i ], 0, NULL, NULL );
        test_error( err, "Unable to write input buffer" );

        if (test_double) {
        err = clEnqueueWriteBuffer( queue, streams[ 4 + i ], CL_TRUE, 0, sizeof( cl_double ) * num_elements, input_ptr_double[ i ], 0, NULL, NULL );
        test_error( err, "Unable to write input buffer" );
        }
    }

    for( i = 0; i < kTotalVecCount; i++ )
    {
        err = create_single_kernel_helper( context, &program[ i ], &kernel[ i ], 1, &clamp_float_codes[ i ], "test_clamp" );
        test_error( err, "Unable to create kernel" );

        log_info("Just made a program for float, i=%d, size=%d, in slot %d\n", i, g_arrVecSizes[i], i);
        fflush(stdout);

        if (test_double) {
        err = create_single_kernel_helper( context, &program[ kTotalVecCount + i ], &kernel[ kTotalVecCount + i ], 1, &clamp_double_codes[ i ], "test_clamp" );
        log_info("Just made a program for double, i=%d, size=%d, in slot %d\n", i, g_arrVecSizes[i], kTotalVecCount+i);
        fflush(stdout);
        test_error( err, "Unable to create kernel" );
        }
    }

    for( i = 0; i < kTotalVecCount; i++ )
    {
        for( j = 0; j < 4; j++ )
        {
            err = clSetKernelArg( kernel[ i ], j, sizeof( streams[ j ] ), &streams[ j ] );
            test_error( err, "Unable to set kernel argument" );
        }

        threads[0] = (size_t)n_elems;

        err = clEnqueueNDRangeKernel( queue, kernel[i], 1, NULL, threads, NULL, 0, NULL, NULL );
        test_error( err, "Unable to execute kernel" );

        err = clEnqueueReadBuffer( queue, streams[3], true, 0, sizeof(cl_float)*num_elements, (void *)output_ptr, 0, NULL, NULL );
        test_error( err, "Unable to read results" );

        if (verify_clamp(input_ptr[0], input_ptr[1], input_ptr[2], output_ptr, n_elems*((g_arrVecSizes[i]))))
        {
            log_error("CLAMP float%d test failed\n", ((g_arrVecSizes[i])));
            err = -1;
        }
        else
        {
            log_info("CLAMP float%d test passed\n", ((g_arrVecSizes[i])));
            err = 0;
        }



        if (err)
        break;
    }

    // If the device supports double precision then test that
    if (test_double)
    {
        for( ; i < 2*kTotalVecCount; i++ )
        {

            log_info("Start of test_double loop, i is %d\n", i);
            for( j = 0; j < 4; j++ )
            {
                err = clSetKernelArg( kernel[i], j, sizeof( streams[j+4] ), &streams[j+4] );
                test_error( err, "Unable to set kernel argument" );
            }

            threads[0] = (size_t)n_elems;

            err = clEnqueueNDRangeKernel( queue, kernel[i], 1, NULL, threads, NULL, 0, NULL, NULL );
            test_error( err, "Unable to execute kernel" );

            err = clEnqueueReadBuffer( queue, streams[7], CL_TRUE, 0, sizeof(cl_double)*num_elements, (void *)output_ptr_double, 0, NULL, NULL );
            test_error( err, "Unable to read results" );

            if (verify_clamp_double(input_ptr_double[0], input_ptr_double[1], input_ptr_double[2], output_ptr_double, n_elems*g_arrVecSizes[(i-kTotalVecCount)]))
            {
                log_error("CLAMP double%d test failed\n", g_arrVecSizes[(i-kTotalVecCount)]);
                err = -1;
            }
            else
            {
                log_info("CLAMP double%d test passed\n", g_arrVecSizes[(i-kTotalVecCount)]);
                err = 0;
            }

            if (err)
            break;
        }
    }


    for( i = 0; i < ((test_double) ? 8 : 4); i++ )
    {
        clReleaseMemObject(streams[i]);
    }
    for (i=0; i < ((test_double) ? kTotalVecCount * 2-1 : kTotalVecCount); i++)
    {
        clReleaseKernel(kernel[i]);
        clReleaseProgram(program[i]);
    }
    free(input_ptr[0]);
    free(input_ptr[1]);
    free(input_ptr[2]);
    free(output_ptr);
    free(program);
    free(kernel);
    if (test_double) {
        free(input_ptr_double[0]);
        free(input_ptr_double[1]);
        free(input_ptr_double[2]);
        free(output_ptr_double);
    }

    return err;
}


