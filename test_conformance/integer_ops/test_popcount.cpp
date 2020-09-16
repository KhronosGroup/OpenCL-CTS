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

#define str(s) #s

#define __popcnt(x, __T, __n, __r) \
    { \
        __T y = x; \
        __r = 0; \
        int k; \
        for(k = 0; k < __n; k++) \
        { \
            if(y & (__T)0x1) __r++; \
            y >>= (__T)1; \
        } \
    }

#define __verify_popcount_func(__T) \
    static int verify_popcount_##__T( const void *p, const void *r, size_t n, const char *sizeName, size_t vecSize ) \
    { \
        const __T *inA = (const __T *) p; \
        const __T *outptr = (const __T *) r; \
        size_t i; \
        int _n = sizeof(__T)*8; \
        __T ref; \
        for(i = 0; i < n; i++) \
        { \
            __T x = inA[i]; \
            __T res = outptr[i]; \
            __popcnt(x, __T, _n, ref); \
            if(res != ref) \
            { \
                log_info( "%ld) Failure for popcount( (%s%s) 0x%x ) = *%d vs %d\n", i, str(__T), sizeName, x, (int)ref, (int)res ); \
                return -1; \
            }\
        } \
        return 0; \
    }

__verify_popcount_func(cl_char);
__verify_popcount_func(cl_uchar);
__verify_popcount_func(cl_short);
__verify_popcount_func(cl_ushort);
__verify_popcount_func(cl_int);
__verify_popcount_func(cl_uint);
__verify_popcount_func(cl_long);
__verify_popcount_func(cl_ulong);

typedef int (*verifyFunc)( const void *, const void *, size_t n, const char *sizeName, size_t vecSize);
static const verifyFunc verify[] = {   verify_popcount_cl_char, verify_popcount_cl_uchar,
    verify_popcount_cl_short, verify_popcount_cl_ushort,
    verify_popcount_cl_int, verify_popcount_cl_uint,
    verify_popcount_cl_long, verify_popcount_cl_ulong };

static const char *test_str_names[] = { "char", "uchar", "short", "ushort", "int", "uint", "long", "ulong" };

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

int test_popcount(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
    cl_int *input_ptr[1], *output_ptr, *p;
    int err;
    int i;
    cl_uint vectorSize;
    cl_uint type;
    MTdata d;
    int fail_count = 0;

    size_t length = sizeof(cl_int) * 8 * n_elems;

    input_ptr[0] = (cl_int*)malloc(length);
    output_ptr   = (cl_int*)malloc(length);

    d = init_genrand( gRandomSeed );
    p = input_ptr[0];
    for (i=0; i<8 * n_elems; i++)
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

        err = clEnqueueWriteBuffer(queue, streams[0], CL_TRUE, 0, length, input_ptr[0], 0, NULL, NULL);
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
                "__kernel void test_popcount_", test_str_names[type], vector_size_names[vectorSize],
                "(__global ", test_str_names[type], vector_param_size_names[vectorSize],
                " *srcA, __global ", test_str_names[type], vector_param_size_names[vectorSize],
                " *dst)\n"
                "{\n"
                "    int  tid = get_global_id(0);\n"
                "\n"
                "    ", test_str_names[type], vector_size_names[vectorSize], " sA;\n",
                "    sA = ", ( vector_sizes[ vectorSize ] == 3 ) ? "vload3( tid, srcA )" : "srcA[tid]", ";\n",
                "    ", test_str_names[type], vector_size_names[vectorSize], " dstVal = popcount(sA);\n"
                "     ", ( vector_sizes[ vectorSize ] == 3 ) ? "vstore3( dstVal, tid, dst )" : "dst[ tid ] = dstVal", ";\n",
                "}\n" };


            char kernelName[128];
            snprintf( kernelName, sizeof( kernelName ), "test_popcount_%s%s", test_str_names[type], vector_size_names[vectorSize] );

            err = create_single_kernel_helper(context, &program, &kernel, sizeof( source ) / sizeof( source[0] ), source, kernelName );

            if (err) {
                return -1;
            }

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

            size_t size = elementCount / (vector_sizes[vectorSize]);
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

            char *inP = (char *)input_ptr[0];
            char *outP = (char *)output_ptr;

            for( size_t e = 0; e < size; e++ )
            {
                if( f( inP, outP, (vector_sizes[vectorSize]), vector_size_names[vectorSize], vector_sizes[vectorSize] ) ) {
                    printSrc(source, sizeof(source)/sizeof(source[0]));
                    ++fail_count; break; // return -1;
                }
                inP += kSizes[type] * ( (vector_sizes[vectorSize]) );
                outP += kSizes[type] * ( (vector_sizes[vectorSize]) );
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

    free(input_ptr[0]);
    free(output_ptr);

    return err;
}


