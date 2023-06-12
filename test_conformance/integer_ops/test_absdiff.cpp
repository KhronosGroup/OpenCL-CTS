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

template <class Integer>
static typename std::make_unsigned<Integer>::type abs_diff(Integer a, Integer b)
{
    using Unsigned = typename std::make_unsigned<Integer>::type;
    Unsigned ua = a;
    Unsigned ub = b;
    Unsigned diff = ua - ub;
    if (a < b) diff = -diff;
    return diff;
}

static int verify_absdiff_char( const void *p, const void *q, const void *r, size_t n, const char *sizeName, size_t vecSize )
{
    const cl_char *inA = (const cl_char *)p;
    const cl_char *inB = (const cl_char *)q;
    const cl_uchar *outptr = (const cl_uchar *)r;
    size_t i;
    for( i = 0; i < n; i++ )
    {
        cl_uchar r = abs_diff(inA[i], inB[i]);
        if( r != outptr[i] )
        { log_info( "%ld) Failure for absdiff( (char%s) 0x%2.2x, (char%s) 0x%2.2x) = *0x%2.2x vs 0x%2.2x\n", i, sizeName, inA[i], sizeName, inB[i], r, outptr[i] ); return -1; }
    }
    return 0;
}

static int verify_absdiff_uchar( const void *p, const void *q, const void *r, size_t n, const char *sizeName, size_t vecSize )
{
    const cl_uchar *inA = (const cl_uchar *)p;
    const cl_uchar *inB = (const cl_uchar *)q;
    const cl_uchar *outptr = (const cl_uchar *)r;
    size_t i;
    for( i = 0; i < n; i++ )
    {
        cl_uchar r = abs_diff(inA[i], inB[i]);
        if( r != outptr[i] )
        { log_info( "%ld) Failure for absdiff( (uchar%s) 0x%2.2x, (uchar%s) 0x%2.2x) = *0x%2.2x vs 0x%2.2x\n", i, sizeName, inA[i], sizeName, inB[i], r, outptr[i] ); return -1; }
    }
    return 0;
}

static int verify_absdiff_short( const void *p, const void *q, const void *r, size_t n, const char *sizeName, size_t vecSize )
{
    const cl_short *inA = (const cl_short *)p;
    const cl_short *inB = (const cl_short *)q;
    const cl_ushort *outptr = (const cl_ushort *)r;
    size_t i;
    for( i = 0; i < n; i++ )
    {
        cl_ushort r = abs_diff(inA[i], inB[i]);
        if( r != outptr[i] )
        { log_info( "%ld) Failure for absdiff( (short%s) 0x%4.4x, (short%s) 0x%4.4x) = *0x%4.4x vs 0x%4.4x\n", i, sizeName, inA[i], sizeName, inB[i], r, outptr[i] ); return -1; }
    }
    return 0;
}

static int verify_absdiff_ushort( const void *p, const void *q, const void *r, size_t n, const char *sizeName, size_t vecSize )
{
    const cl_ushort *inA = (const cl_ushort *)p;
    const cl_ushort *inB = (const cl_ushort *)q;
    const cl_ushort *outptr = (const cl_ushort *)r;
    size_t i;
    for( i = 0; i < n; i++ )
    {
        cl_ushort r = abs_diff(inA[i], inB[i]);
        if( r != outptr[i] )
        { log_info( "%ld) Failure for absdiff( (ushort%s) 0x%4.4x, (ushort%s) 0x%4.4x) = *0x%4.4x vs 0x%4.4x\n", i, sizeName, inA[i], sizeName, inB[i], r, outptr[i] ); return -1; }
    }
    return 0;
}

static int verify_absdiff_int( const void *p, const void *q, const void *r, size_t n, const char *sizeName, size_t vecSize )
{
    const cl_int *inA = (const cl_int *)p;
    const cl_int *inB = (const cl_int *)q;
    const cl_uint *outptr = (const cl_uint *)r;
    size_t i;
    for( i = 0; i < n; i++ )
    {
        cl_uint r = abs_diff(inA[i], inB[i]);
        if( r != outptr[i] )
        {
            log_info( "%ld) Failure for absdiff( (int%s) 0x%8.8x, (int%s) 0x%8.8x) = *0x%8.8x vs 0x%8.8x\n", i, sizeName, inA[i], sizeName, inB[i], r, outptr[i] );
            return -1;
        }
    }
    return 0;
}

static int verify_absdiff_uint( const void *p, const void *q, const void *r, size_t n, const char *sizeName, size_t vecSize )
{
    const cl_uint *inA = (const cl_uint *)p;
    const cl_uint *inB = (const cl_uint *)q;
    const cl_uint *outptr = (const cl_uint *)r;
    size_t i;
    for( i = 0; i < n; i++ )
    {
        cl_uint r = abs_diff(inA[i], inB[i]);
        if( r != outptr[i] )
        { log_info( "%ld) Failure for absdiff( (uint%s) 0x%8.8x, (uint%s) 0x%8.8x) = *0x%8.8x vs 0x%8.8x\n", i, sizeName, inA[i], sizeName, inB[i], r, outptr[i] ); return -1; }
    }
    return 0;
}

static int verify_absdiff_long( const void *p, const void *q, const void *r, size_t n, const char *sizeName, size_t vecSize )
{
    const cl_long *inA = (const cl_long *)p;
    const cl_long *inB = (const cl_long *)q;
    const cl_ulong *outptr = (const cl_ulong *)r;
    size_t i;
    for( i = 0; i < n; i++ )
    {
        cl_ulong r = abs_diff(inA[i], inB[i]);
        if( r != outptr[i] )
        { log_info( "%ld) Failure for absdiff( (long%s) 0x%16.16llx, (long%s) 0x%16.16llx) = *0x%16.16llx vs 0x%16.16llx\n", i, sizeName, inA[i], sizeName, inB[i], r, outptr[i] ); return -1; }
    }
    return 0;
}

static int verify_absdiff_ulong( const void *p, const void *q, const void *r, size_t n, const char *sizeName, size_t vecSize )
{
    const cl_ulong *inA = (const cl_ulong *)p;
    const cl_ulong *inB = (const cl_ulong *)q;
    const cl_ulong *outptr = (const cl_ulong *)r;
    size_t i;
    for( i = 0; i < n; i++ )
    {
        cl_ulong r = abs_diff(inA[i], inB[i]);
        if( r != outptr[i] )
        { log_info( "%ld) Failure for absdiff( (ulong%s) 0x%16.16llx, (ulong%s) 0x%16.16llx) = *0x%16.16llx vs 0x%16.16llx\n", i, sizeName, inA[i], sizeName, inB[i], r, outptr[i] ); return -1; }
    }
    return 0;
}

typedef int (*verifyFunc)( const void *, const void *, const void *, size_t n, const char *sizeName, size_t vecSize);
static const verifyFunc verify[] = {   verify_absdiff_char, verify_absdiff_uchar,
    verify_absdiff_short, verify_absdiff_ushort,
    verify_absdiff_int, verify_absdiff_uint,
    verify_absdiff_long, verify_absdiff_ulong };

//FIXME:  enable long and ulong when GPU path is working
static const char *test_str_names[] = { "char", "uchar", "short", "ushort", "int", "uint", "long", "ulong" };

//FIXME:  enable "16" when support for > 64 byte vectors go into LLVM
static const int vector_sizes[] = {1, 2, 3, 4, 8, 16};
static const char *vector_size_names[] = { "", "2", "3", "4", "8", "16" };
static const char *vector_param_size_names[] = { "", "2", "", "4", "8", "16" };
static const size_t  kSizes[8] = { 1, 1, 2, 2, 4, 4, 8, 8 };

static void printSrc(const char *src[], int nSrcStrings) {
    int i;
    for(i = 0; i < nSrcStrings; ++i) {
        log_info("%s", src[i]);
    }
}

int test_integer_abs_diff(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
    cl_int *input_ptr[2], *output_ptr, *p;
    int err;
    int i;
    cl_uint vectorSize;
    cl_uint type;
    MTdata d;
    int fail_count = 0;

    size_t length = sizeof(cl_int) * 4 * n_elems;

    input_ptr[0] = (cl_int*)malloc(length);
    input_ptr[1] = (cl_int*)malloc(length);
    output_ptr   = (cl_int*)malloc(length);

    d = init_genrand( gRandomSeed );
    p = input_ptr[0];
    for (i=0; i<4 * n_elems; i++)
        p[i] = genrand_int32(d);
    p = input_ptr[1];
    for (i=0; i<4 * n_elems; i++)
        p[i] = genrand_int32(d);
    free_mtdata(d);  d = NULL;

    for( type = 0; type < sizeof( test_str_names ) / sizeof( test_str_names[0] ); type++ )
    {
        //embedded devices don't support long/ulong so skip over
        if (! gHasLong && strstr(test_str_names[type],"long"))
        {
           log_info( "WARNING: 64 bit integers are not supported on this device. Skipping %s\n", test_str_names[type] );
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
                "__kernel void test_absdiff_", test_str_names[type], vector_size_names[vectorSize],
                "(__global ", test_str_names[type], vector_param_size_names[vectorSize],
                " *srcA, __global ", test_str_names[type], vector_param_size_names[vectorSize],
                " *srcB, __global u", test_str_names[type & -2], vector_param_size_names[vectorSize],
                " *dst)\n"
                "{\n"
                "    int  tid = get_global_id(0);\n"
                "\n"
                "    ", test_str_names[type], vector_size_names[vectorSize], " sA, sB;\n",
                "    sA = ", ( vector_sizes[ vectorSize ] == 3 ) ? "vload3( tid, srcA )" : "srcA[tid]", ";\n",
                "    sB = ", ( vector_sizes[ vectorSize ] == 3 ) ? "vload3( tid, srcB )" : "srcB[tid]", ";\n",
                "    u", test_str_names[type & -2], vector_size_names[vectorSize], " dstVal = abs_diff(sA, sB);\n"
                "     ", ( vector_sizes[ vectorSize ] == 3 ) ? "vstore3( dstVal, tid, dst )" : "dst[ tid ] = dstVal", ";\n",
                "}\n" };


            char kernelName[128];
            snprintf( kernelName, sizeof( kernelName ), "test_absdiff_%s%s", test_str_names[type], vector_size_names[vectorSize] );

            err = create_single_kernel_helper(context, &program, &kernel, sizeof( source ) / sizeof( source[0] ), source, kernelName );

            if (err) {
                return -1;
            }

#if 0
            log_info("About to run\n");
            log_info("=====\n");
            printSrc(source, sizeof(source)/sizeof(source[0]));
            log_info("=====\n");
#endif

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

            size_t size = elementCount / (vector_sizes[vectorSize]);
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
                if( f( inP, inP2, outP, (vector_sizes[vectorSize]), vector_size_names[vectorSize], vector_sizes[vectorSize] ) ) {
                    printSrc(source, sizeof(source)/sizeof(source[0]));
                    ++fail_count; break; // return -1;
                }
                inP += kSizes[type] * ( (vector_sizes[vectorSize]) );
                inP2 += kSizes[type] * ( (vector_sizes[vectorSize]) );
                outP += kSizes[type] * ( (vector_sizes[vectorSize]) );
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


