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

#include "procs.h"


static int verify_abs_char( const void *p, const void *q, size_t n, const char *sizeName, size_t vecSize )
{
    const cl_char *inA = (const cl_char*) p;
    const cl_uchar *outptr = (const cl_uchar*) q;
    size_t i;
    for( i = 0; i < n; i++ )
    {
        cl_uchar r = inA[i];
        if( inA[i] < 0 )
            r = -inA[i];
        if( r != outptr[i] )
        { log_info( "%ld) Failure for abs( (char%s) 0x%2.2x) = *0x%2.2x vs 0x%2.2x\n", i, sizeName, inA[i],r, outptr[i] ); return -1; }
    }
    return 0;
}


static int verify_abs_short( const void *p, const void *q, size_t n, const char *sizeName, size_t vecSize )
{
    const cl_short *inA = (const cl_short*) p;
    const cl_ushort *outptr = (const cl_ushort*) q;
    size_t i;
    for( i = 0; i < n; i++ )
    {
        cl_ushort r = inA[i];
        if( inA[i] < 0 )
            r = -inA[i];
        if( r != outptr[i] )
        { log_info( "%ld) Failure for abs( (short%s) 0x%4.4x) = *0x%4.4x vs 0x%4.4x\n", i, sizeName, inA[i],r, outptr[i] ); return -1; }
    }
    return 0;
}

static int verify_abs_int( const void *p, const void *q, size_t n, const char *sizeName , size_t vecSize)
{
    const cl_int *inA = (const cl_int*) p;
    const cl_uint *outptr = (const cl_uint*) q;
    size_t i;
    for( i = 0; i < n; i++ )
    {
        cl_uint r = inA[i];
        if( inA[i] < 0 )
            r = -inA[i];
        if( r != outptr[i] )
        { log_info( "%ld) Failure for abs( (int%s) 0x%2.2x) = *0x%8.8x vs 0x%8.8x\n", i, sizeName, inA[i],r, outptr[i] ); return -1; }
    }
    return 0;
}

static int verify_abs_long( const void *p, const void *q, size_t n, const char *sizeName, size_t vecSize )
{
    const cl_long *inA = (const cl_long*) p;
    const cl_ulong *outptr = (const cl_ulong*) q;
    size_t i;
    for( i = 0; i < n; i++ )
    {
        cl_ulong r = inA[i];
        if( inA[i] < 0 )
            r = -inA[i];
        if( r != outptr[i] )
        { log_info( "%ld) Failure for abs( (long%s) 0x%16.16llx) = *0x%16.16llx vs 0x%16.16llx\n", i, sizeName, inA[i],r, outptr[i] ); return -1; }
    }
    return 0;
}



static int verify_abs_uchar( const void *p, const void *q, size_t n, const char *sizeName, size_t vecSize )
{
    const cl_uchar *inA = (const cl_uchar*) p;
    const cl_uchar *outptr = (const cl_uchar*) q;
    size_t i;
    for( i = 0; i < n; i++ )
    {
        cl_uchar r = inA[i];
        if( r != outptr[i] )
        { log_info( "%ld) Failure for abs( (uchar%s) 0x%2.2x) = *0x%2.2x vs 0x%2.2x\n", i, sizeName, inA[i],r, outptr[i] ); return -1; }
    }
    return 0;
}


static int verify_abs_ushort( const void *p, const void *q, size_t n, const char *sizeName, size_t vecSize )
{
    const cl_ushort *inA = (const cl_ushort*) p;
    const cl_ushort *outptr = (const cl_ushort*) q;
    size_t i;
    for( i = 0; i < n; i++ )
    {
        cl_ushort r = inA[i];
        if( r != outptr[i] )
        { log_info( "%ld) Failure for abs( (short%s) 0x%4.4x) = *0x%4.4x vs 0x%4.4x\n", i, sizeName, inA[i],r, outptr[i] ); return -1; }
    }
    return 0;
}

static int verify_abs_uint( const void *p, const void *q, size_t n, const char *sizeName , size_t vecSize)
{
    const cl_uint *inA = (const cl_uint*) p;
    const cl_uint *outptr = (const cl_uint*) q;
    size_t i;
    for( i = 0; i < n; i++ )
    {
        cl_uint r = inA[i];
        if( r != outptr[i] )
        { log_info( "%ld) Failure for abs( (int%s) 0x%2.2x) = *0x%8.8x vs 0x%8.8x\n", i, sizeName, inA[i],r, outptr[i] ); return -1; }
    }
    return 0;
}

static int verify_abs_ulong( const void *p, const void *q, size_t n, const char *sizeName, size_t vecSize )
{
    const cl_ulong *inA = (const cl_ulong*) p;
    const cl_ulong *outptr = (const cl_ulong*) q;
    size_t i;
    for( i = 0; i < n; i++ )
    {
        cl_ulong r = inA[i];
        if( r != outptr[i] )
        { log_info( "%ld) Failure for abs( (long%s) 0x%16.16llx) = *0x%16.16llx vs 0x%16.16llx\n", i, sizeName, inA[i],r, outptr[i] ); return -1; }
    }
    return 0;
}


typedef int (*verifyFunc)( const void *, const void *, size_t n, const char *sizeName, size_t vecSize );
static const verifyFunc verify[] = {
    verify_abs_char, verify_abs_short, verify_abs_int, verify_abs_long,
    verify_abs_uchar, verify_abs_ushort, verify_abs_uint, verify_abs_ulong
};

static const char *test_str_names[] = { "char", "short", "int", "long" ,
    "uchar", "ushort", "uint", "ulong"};
static const char *test_ustr_names[] = { "uchar", "ushort", "uint", "ulong" ,
    "uchar", "ushort", "uint", "ulong"};
static const int vector_sizes[] = {1, 2, 3, 4, 8, 16};
static const char *vector_size_names[] = { "", "2", "3", "4", "8", "16" };
static const char *vector_size_names_io_types[] = { "", "2", "", "4", "8", "16" };
static const size_t  kSizes[9] = { 1, 2, 4, 8, 1, 2, 4, 8 };

static const char * source_loads[] = {
    "srcA[tid]",
    "vload3(tid, srcA)"
};

static const char * dest_stores[] = {
    "    dst[tid] = tmp;\n",
    "    vstore3(tmp, tid, dst);\n"
};

int test_integer_abs(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
    cl_int *input_ptr, *output_ptr, *p;
    int err;
    int i;
    cl_uint vectorSizeIdx;
    cl_uint type;
    MTdata d;
    int fail_count = 0;

    size_t length = sizeof(cl_int) * 4 * n_elems;

    input_ptr   = (cl_int*)malloc(length);
    output_ptr  = (cl_int*)malloc(length);

    p = input_ptr;
    d = init_genrand( gRandomSeed );
    for (i=0; i<n_elems * 4; i++)
        p[i] = genrand_int32(d);
    free_mtdata(d); d = NULL;

    for( type = 0; type < sizeof( test_str_names ) / sizeof( test_str_names[0] ); type++ )
    {
        //embedded devices don't support long/ulong so skip over
        if (! gHasLong && strstr(test_str_names[type],"long"))
        {
           log_info( "WARNING: 64 bit integers are not supported on this device. Skipping %s\n", test_str_names[type] );
           continue;
        }

        verifyFunc f = verify[ type ];

        size_t elementCount = length / kSizes[type];
        cl_mem streams[2];

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

        err = clEnqueueWriteBuffer(queue, streams[0], CL_TRUE, 0, length, input_ptr, 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            log_error("clEnqueueWriteBuffer failed\n");
            return -1;
        }



        for( vectorSizeIdx = 0; vectorSizeIdx < sizeof( vector_size_names ) / sizeof( vector_size_names[0] ); vectorSizeIdx++ )
        {
            cl_program program = NULL;
            cl_kernel kernel = NULL;

            const char *source[] = {
                "__kernel void test_abs_",
                test_str_names[type],
                vector_size_names[vectorSizeIdx],
                "(__global ", test_str_names[type],
                vector_size_names_io_types[vectorSizeIdx],
                " *srcA, __global ", test_ustr_names[type],
                vector_size_names_io_types[vectorSizeIdx],
                " *dst)\n"
                "{\n"
                "    int  tid = get_global_id(0);\n"
                "\n"
                "    ", test_ustr_names[type], vector_size_names[vectorSizeIdx],
                " tmp = abs(", source_loads[!!(vector_sizes[vectorSizeIdx]==3)], ");\n",
                dest_stores[!!(vector_sizes[vectorSizeIdx]==3)],
                "}\n"
            };

            char kernelName[128];
            snprintf( kernelName, sizeof( kernelName ), "test_abs_%s%s", test_str_names[type], vector_size_names[vectorSizeIdx] );
            err = create_single_kernel_helper(context, &program, &kernel, sizeof( source ) / sizeof( source[0] ), source, kernelName );
            if (err)
                return -1;

            err  = clSetKernelArg(kernel, 0, sizeof streams[0], &streams[0]);
            err |= clSetKernelArg(kernel, 1, sizeof streams[1], &streams[1]);
            if (err != CL_SUCCESS)
            {
                log_error("clSetKernelArgs failed\n");
                return -1;
            }

            //Wipe the output buffer clean
            uint32_t pattern = 0xdeadbeef;
            memset_pattern4( output_ptr, &pattern, length );
            err = clEnqueueWriteBuffer(queue, streams[1], CL_TRUE, 0, length, output_ptr, 0, NULL, NULL);
            if (err != CL_SUCCESS)
            {
                log_error("clEnqueueWriteBuffer failed\n");
                return -1;
            }

            size_t size = elementCount / ((vector_sizes[vectorSizeIdx]));
            err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &size, NULL, 0, NULL, NULL);
            if (err != CL_SUCCESS)
            {
                log_error("clEnqueueNDRangeKernel failed\n");
                return -1;
            }

            err = clEnqueueReadBuffer(queue, streams[1], CL_TRUE, 0, length, output_ptr, 0, NULL, NULL);
            if (err != CL_SUCCESS)
            {
                log_error("clEnqueueReadBuffer failed\n");
                return -1;
            }

            char *inP = (char *)input_ptr;
            char *outP = (char *)output_ptr;

            for( size_t e = 0; e < size; e++ )
            {
                if( f( inP, outP, (vector_sizes[vectorSizeIdx]), vector_size_names[vectorSizeIdx], vector_sizes[vectorSizeIdx] ) ) {
                    ++fail_count; break; // return -1;
                }
                inP += kSizes[type] * (vector_sizes[vectorSizeIdx] );
                outP += kSizes[type] * (vector_sizes[vectorSizeIdx]);
            }

            clReleaseKernel( kernel );
            clReleaseProgram( program );
            log_info( "." );
            fflush( stdout );
        }

        clReleaseMemObject( streams[0] );
        clReleaseMemObject( streams[1] );
        log_info( "done\n" );
    }

    if(fail_count) {
        log_info("Failed on %d types\n", fail_count);
        return -1;
    }

    free(input_ptr);
    free(output_ptr);

    return err;
}


