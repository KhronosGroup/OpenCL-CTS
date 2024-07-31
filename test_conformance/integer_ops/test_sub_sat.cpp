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
#include <limits.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <algorithm>
#include <cinttypes>

#include "procs.h"

static int verify_subsat_char( const cl_char *inA, const cl_char *inB, const cl_char *outptr, int n, const char *sizeName, int vecSize )
{
    int i;
    for( i = 0; i < n; i++ )
    {
        cl_int r = (cl_int) inA[i] - (cl_int) inB[i];
        r = std::max(r, CL_CHAR_MIN);
        r = std::min(r, CL_CHAR_MAX);

        if( r != outptr[i] )
        { log_info( "\n%d) Failure for sub_sat( (char%s) 0x%2.2x, (char%s) 0x%2.2x) = *0x%2.2x vs 0x%2.2x\n", i, sizeName, inA[i], sizeName, inB[i], r, outptr[i] ); return -1; }
    }
    return 0;
}

static int verify_subsat_uchar( const cl_uchar *inA, const cl_uchar *inB, const cl_uchar *outptr, int n, const char *sizeName, int vecSize )
{
    int i;
    for( i = 0; i < n; i++ )
    {
        cl_int r = (cl_int) inA[i] - (cl_int) inB[i];
        r = std::max(r, 0);
        r = std::min(r, CL_UCHAR_MAX);
        if (r != outptr[i])
        { log_info( "\n%d) Failure for sub_sat( (uchar%s) 0x%2.2x, (uchar%s) 0x%2.2x) = *0x%2.2x vs 0x%2.2x\n", i, sizeName, inA[i], sizeName, inB[i], r, outptr[i] ); return -1; }
    }
    return 0;
}

static int verify_subsat_short( const cl_short *inA, const cl_short *inB, const cl_short *outptr, int n, const char *sizeName, int vecSize )
{
    int i;
    for( i = 0; i < n; i++ )
    {
        cl_int r = (cl_int) inA[i] - (cl_int) inB[i];
        r = std::max(r, CL_SHRT_MIN);
        r = std::min(r, CL_SHRT_MAX);

        if( r != outptr[i] )
        { log_info( "\n%d) Failure for sub_sat( (short%s) 0x%4.4x, (short%s) 0x%4.4x) = *0x%4.4x vs 0x%4.4x\n", i, sizeName, inA[i], sizeName, inB[i], r, outptr[i] ); return -1; }
    }
    return 0;
}

static int verify_subsat_ushort( const cl_ushort *inA, const cl_ushort *inB, const cl_ushort *outptr, int n, const char *sizeName , int vecSize)
{
    int i;
    for( i = 0; i < n; i++ )
    {
        cl_int r = (cl_int) inA[i] - (cl_int) inB[i];
        r = std::max(r, 0);
        r = std::min(r, CL_USHRT_MAX);

        if( r != outptr[i] )
        { log_info( "\n%d) Failure for sub_sat( (ushort%s) 0x%4.4x, (ushort%s) 0x%4.4x) = *0x%4.4x vs 0x%4.4x\n", i, sizeName, inA[i], sizeName, inB[i], r, outptr[i] ); return -1; }
    }
    return 0;
}

static int verify_subsat_int( const cl_int *inA, const cl_int *inB, const cl_int *outptr, int n, const char *sizeName , int vecSize)
{
    int i;
    for( i = 0; i < n; i++ )
    {
        cl_int r = (cl_int) ((cl_uint)inA[i] - (cl_uint)inB[i]);
        if( inB[i] < 0 )
        {
            if( r < inA[i] )
                r = CL_INT_MAX;
        }
        else
        {
            if( r > inA[i] )
                r = CL_INT_MIN;
        }


        if( r != outptr[i] )
        { log_info( "\n%d) Failure for sub_sat( (int%s) 0x%8.8x, (int%s) 0x%8.8x) = *0x%8.8x vs 0x%8.8x\n", i, sizeName, inA[i], sizeName, inB[i], r, outptr[i] ); return -1; }
    }
    return 0;
}

static int verify_subsat_uint( const cl_uint *inA, const cl_uint *inB, const cl_uint *outptr, int n, const char *sizeName , int vecSize)
{
    int i;
    for( i = 0; i < n; i++ )
    {
        cl_uint r = inA[i] - inB[i];
        if(  inA[i] < inB[i] )
            r = 0;

        if( r != outptr[i] )
        { log_info( "\n%d) Failure for sub_sat( (uint%s) 0x%8.8x, (uint%s) 0x%8.8x) = *0x%8.8x vs 0x%8.8x\n", i, sizeName, inA[i], sizeName, inB[i], r, outptr[i] ); return -1; }
    }
    return 0;
}

static int verify_subsat_long( const cl_long *inA, const cl_long *inB, const cl_long *outptr, int n, const char *sizeName , int vecSize)
{
    int i;
    for( i = 0; i < n; i++ )
    {
        cl_long r = (cl_long)((cl_ulong)inA[i] - (cl_ulong)inB[i]);
        if( inB[i] < 0 )
        {
            if( r < inA[i] )
                r = CL_LONG_MAX;
        }
        else
        {
            if( r > inA[i] )
                r = CL_LONG_MIN;
        }
        if( r != outptr[i] )
        {
            log_info("%d) Failure for sub_sat( (long%s) 0x%16.16" PRIx64
                     ", (long%s) 0x%16.16" PRIx64 ") = *0x%16.16" PRIx64
                     " vs 0x%16.16" PRIx64 "\n",
                     i, sizeName, inA[i], sizeName, inB[i], r, outptr[i]);
            return -1;
        }
    }
    return 0;
}

static int verify_subsat_ulong( const cl_ulong *inA, const cl_ulong *inB, const cl_ulong *outptr, int n, const char *sizeName , int vecSize)
{
    int i;
    for( i = 0; i < n; i++ )
    {
        cl_ulong r = inA[i] - inB[i];
        if(  inA[i] < inB[i] )
            r = 0;
        if( r != outptr[i] )
        {
            log_info("%d) Failure for sub_sat( (ulong%s) 0x%16.16" PRIx64
                     ", (ulong%s) 0x%16.16" PRIx64 ") = *0x%16.16" PRIx64
                     " vs 0x%16.16" PRIx64 "\n",
                     i, sizeName, inA[i], sizeName, inB[i], r, outptr[i]);
            return -1;
        }
    }
    return 0;
}

typedef int (*verifyFunc)( const void *, const void *, const void *, int n, const char *sizeName, int );
static const verifyFunc verify[] = {   (verifyFunc) verify_subsat_char, (verifyFunc) verify_subsat_uchar,
    (verifyFunc) verify_subsat_short, (verifyFunc) verify_subsat_ushort,
    (verifyFunc) verify_subsat_int, (verifyFunc) verify_subsat_uint,
    (verifyFunc) verify_subsat_long, (verifyFunc) verify_subsat_ulong };

static const char *test_str_names[] = { "char", "uchar", "short", "ushort", "int", "uint", "long", "ulong" };
static const int vector_sizes[] = {1, 2, 3, 4, 8, 16};
static const char *vector_size_names[] = { "", "2", "3", "4", "8", "16" };

static const size_t  kSizes[8] = { 1, 1, 2, 2, 4, 4, 8, 8 };

int test_integer_sub_sat(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
    int *input_ptr[2], *output_ptr, *p;
    int err;
    cl_uint i;
    cl_uint vectorSize;
    cl_uint type;
    MTdata d;
    int fail_count = 0;

    size_t length = sizeof(int) * 4 * n_elems;

    input_ptr[0] = (int*)malloc(length);
    input_ptr[1] = (int*)malloc(length);
    output_ptr   = (int*)malloc(length);

    d = init_genrand( gRandomSeed );
    p = input_ptr[0];
    for (i=0; i<4 * (cl_uint) n_elems; i++)
        p[i] = genrand_int32(d);
    p = input_ptr[1];
    for (i=0; i<4 * (cl_uint) n_elems; i++)
        p[i] = genrand_int32(d);
    free_mtdata(d); d = NULL;

    for( type = 0; type < sizeof( test_str_names ) / sizeof( test_str_names[0] ); type++ )
    {

        //embedded devices don't support long/ulong so skip over
        if (! gHasLong && strstr(test_str_names[type],"long"))
        {
            log_info( "WARNING: device does not support 64-bit integers. Skipping %s\n", test_str_names[type] );
            continue;
        }

        verifyFunc f = verify[ type ];
        // Note: restrict the element count here so we don't end up overrunning the output buffer if we're compensating for 32-bit writes
        size_t elementCount = length / kSizes[type];
        cl_mem streams[3];

        log_info( "%s", test_str_names[type] );
        fflush( stdout );

        // Set up data streams for the type
        streams[0] = clCreateBuffer(context, 0, length, NULL, NULL);
        if (!streams[0])
        {
            log_error("clCreateBuffer failed\n");
            return -1;
        }
        streams[1] = clCreateBuffer(context, 0, length, NULL, NULL);
        if (!streams[1])
        {
            log_error("clCreateBuffer failed\n");
            return -1;
        }
        streams[2] = clCreateBuffer(context, 0, length, NULL, NULL);
        if (!streams[2])
        {
            log_error("clCreateBuffer failed\n");
            return -1;
        }

        err = clEnqueueWriteBuffer(queue, streams[0], CL_TRUE, 0, length, input_ptr[0], 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            log_error("clEnqueueWriteBuffer failed\n");
            return -1;
        }
        err = clEnqueueWriteBuffer(queue, streams[1], CL_TRUE, 0, length, input_ptr[1], 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            log_error("clEnqueueWriteBuffer failed\n");
            return -1;
        }

        for( vectorSize = 0; vectorSize < sizeof( vector_size_names ) / sizeof( vector_size_names[0] ); vectorSize++ )
        {
            cl_program program = NULL;
            cl_kernel kernel = NULL;

            const char *source[] = {
                "__kernel void test_sub_sat_", test_str_names[type], vector_size_names[vectorSize],
                "(__global ", test_str_names[type], vector_size_names[vectorSize],
                " *srcA, __global ", test_str_names[type], vector_size_names[vectorSize],
                " *srcB, __global ", test_str_names[type], vector_size_names[vectorSize],
                " *dst)\n"
                "{\n"
                "    int  tid = get_global_id(0);\n"
                "\n"
                "    ", test_str_names[type], vector_size_names[vectorSize], " tmp = sub_sat(srcA[tid], srcB[tid]);\n"
                "    dst[tid] = tmp;\n"
                "}\n"
            };

            const char *sourceV3[] = {
                "__kernel void test_sub_sat_", test_str_names[type], vector_size_names[vectorSize],
                "(__global ", test_str_names[type],
                " *srcA, __global ", test_str_names[type],
                " *srcB, __global ", test_str_names[type],
                " *dst)\n"
                "{\n"
                "    int  tid = get_global_id(0);\n"
                "\n"
                "    ", test_str_names[type], vector_size_names[vectorSize], " tmp = sub_sat(vload3(tid, srcA), vload3(tid, srcB));\n"
                "    vstore3(tmp, tid, dst);\n"
                "}\n"
            };

            char kernelName[128];
            snprintf( kernelName, sizeof( kernelName ), "test_sub_sat_%s%s", test_str_names[type], vector_size_names[vectorSize] );
            if(vector_sizes[vectorSize] != 3)
            {
                err = create_single_kernel_helper(context, &program, &kernel, sizeof( source ) / sizeof( source[0] ), source, kernelName );
            } else {
                err = create_single_kernel_helper(context, &program, &kernel, sizeof( sourceV3 ) / sizeof( sourceV3[0] ), sourceV3, kernelName );
            }
            if (err)
                return -1;

            err  = clSetKernelArg(kernel, 0, sizeof streams[0], &streams[0]);
            err |= clSetKernelArg(kernel, 1, sizeof streams[1], &streams[1]);
            err |= clSetKernelArg(kernel, 2, sizeof streams[2], &streams[2]);
            if (err != CL_SUCCESS)
            {
                log_error("clSetKernelArgs failed\n");
                return -1;
            }

            //Wipe the output buffer clean
            uint32_t pattern = 0xdeadbeef;
            memset_pattern4( output_ptr, &pattern, length );
            err = clEnqueueWriteBuffer(queue, streams[2], CL_TRUE, 0, length, output_ptr, 0, NULL, NULL);
            if (err != CL_SUCCESS)
            {
                log_error("clEnqueueWriteBuffer failed\n");
                return -1;
            }

            size_t size = elementCount / vector_sizes[vectorSize];
            err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &size, NULL, 0, NULL, NULL);
            if (err != CL_SUCCESS)
            {
                log_error("clEnqueueNDRangeKernel failed\n");
                return -1;
            }

            err = clEnqueueReadBuffer(queue, streams[2], CL_TRUE, 0, length, output_ptr, 0, NULL, NULL);
            if (err != CL_SUCCESS)
            {
                log_error("clEnqueueReadBuffer failed\n");
                return -1;
            }

            char *inP = (char *)input_ptr[0];
            char *inP2 = (char *)input_ptr[1];
            char *outP = (char *)output_ptr;

            for( size_t e = 0; e < size; e++ )
            {
                if( f( inP, inP2, outP, vector_sizes[vectorSize], vector_size_names[vectorSize], vector_sizes[vectorSize] ) ) {
                    ++fail_count; break; // return -1;
                }
                inP += kSizes[type] * vector_sizes[vectorSize];
                inP2 += kSizes[type] * vector_sizes[vectorSize];
                outP += kSizes[type] * vector_sizes[vectorSize];
            }

            clReleaseKernel( kernel );
            clReleaseProgram( program );
            log_info( "." );
            fflush( stdout );
        }

        clReleaseMemObject( streams[0] );
        clReleaseMemObject( streams[1] );
        clReleaseMemObject( streams[2] );
        log_info( "done\n" );
    }
    if(fail_count) {
        log_info("Failed on %d types\n", fail_count);
        return -1;
    }

    free(input_ptr[0]);
    free(input_ptr[1]);
    free(output_ptr);

    return err;
}


