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

int hi_offset( int index, int vectorSize) { return index + vectorSize / 2; }
int lo_offset( int index, int vectorSize) { return index; }
int even_offset( int index, int vectorSize ) { return index * 2; }
int odd_offset( int index, int vectorSize ) { return index * 2 + 1; }

typedef int (*OffsetFunc)( int index, int vectorSize );
static const OffsetFunc offsetFuncs[4] = { hi_offset, lo_offset, even_offset, odd_offset };
typedef int (*verifyFunc)( const void *, const void *, const void *, int n, const char *sizeName );
static const char *operatorToUse_names[] = { "hi", "lo", "even", "odd" };
static const char *test_str_names[] = { "char", "uchar", "short", "ushort", "int", "uint", "long", "ulong", "float", "double" };

static const unsigned int vector_sizes[] =     { 1, 2, 3, 4, 8, 16};
static const unsigned int vector_aligns[] =    { 1, 2, 4, 4, 8, 16};
static const unsigned int out_vector_idx[] =   { 0, 0, 1, 1, 3, 4};
// if input is size vector_sizes[i], output is size
// vector_sizes[out_vector_idx[i]]
// input type name is strcat(gentype, vector_size_names[i]);
// and output type name is
// strcat(gentype, vector_size_names[out_vector_idx[i]]);
static const char *vector_size_names[] = { "", "2", "3", "4", "8", "16"};

static const size_t  kSizes[] = { 1, 1, 2, 2, 4, 4, 8, 8, 4, 8 };
static int CheckResults( void *in, void *out, size_t elementCount, int type, int vectorSize, int operatorToUse );

int test_hiloeo(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
    cl_int *input_ptr, *output_ptr, *p;
    int err;
    cl_uint i;
    int hasDouble = is_extension_available( device, "cl_khr_fp64" );
    cl_uint vectorSize, operatorToUse;
    cl_uint type;
    MTdata d;

    int expressionMode;
    int numExpressionModes = 2;

    size_t length = sizeof(cl_int) * 4 * n_elems;

    input_ptr   = (cl_int*)malloc(length);
    output_ptr  = (cl_int*)malloc(length);

    p = input_ptr;
    d = init_genrand( gRandomSeed );
    for (i=0; i<4 * (cl_uint) n_elems; i++)
        p[i] = genrand_int32(d);
    free_mtdata(d); d = NULL;

    for( type = 0; type < sizeof( test_str_names ) / sizeof( test_str_names[0] ); type++ )
    {
        // Note: restrict the element count here so we don't end up overrunning the output buffer if we're compensating for 32-bit writes
        size_t elementCount = length / kSizes[type];
        cl_mem streams[2];

        // skip double if unavailable
        if( !hasDouble && ( 0 == strcmp( test_str_names[type], "double" )))
            continue;

        if( !gHasLong &&
            (( 0 == strcmp( test_str_names[type], "long" )) ||
            ( 0 == strcmp( test_str_names[type], "ulong" ))))
            continue;

        log_info( "%s", test_str_names[type] );
        fflush( stdout );

        // Set up data streams for the type
        streams[0] = clCreateBuffer(context, CL_MEM_READ_WRITE, length, NULL, NULL);
        if (!streams[0])
        {
            log_error("clCreateBuffer failed\n");
            return -1;
        }
        streams[1] = clCreateBuffer(context, CL_MEM_READ_WRITE, length, NULL, NULL);
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

        for( operatorToUse = 0; operatorToUse < sizeof( operatorToUse_names ) / sizeof( operatorToUse_names[0] ); operatorToUse++ )
        {
            log_info( " %s", operatorToUse_names[ operatorToUse ] );
            fflush( stdout );
            for( vectorSize = 1; vectorSize < sizeof( vector_size_names ) / sizeof( vector_size_names[0] ); vectorSize++ ) {
                for(expressionMode = 0; expressionMode < numExpressionModes; ++expressionMode) {

                    cl_program program = NULL;
                    cl_kernel kernel = NULL;
                    cl_uint outVectorSize = out_vector_idx[vectorSize];
                    char expression[1024];

                    const char *source[] = {
                        "", // optional pragma string
                        "__kernel void test_", operatorToUse_names[ operatorToUse ], "_", test_str_names[type], vector_size_names[vectorSize],
                        "(__global ", test_str_names[type], vector_size_names[vectorSize],
                        " *srcA, __global ", test_str_names[type], vector_size_names[outVectorSize],
                        " *dst)\n"
                        "{\n"
                        "    int  tid = get_global_id(0);\n"
                        "\n"
                        "    ", test_str_names[type],
                        vector_size_names[out_vector_idx[vectorSize]],
                        " tmp = ", expression, ".", operatorToUse_names[ operatorToUse ], ";\n"
                        "    dst[tid] = tmp;\n"
                        "}\n"
                    };

                    if(expressionMode == 0) {
                        sprintf(expression, "srcA[tid]");
                    } else if(expressionMode == 1) {
                        switch(vector_sizes[vectorSize]) {
                            case 16:
                                sprintf(expression,
                                        "((%s16)(srcA[tid].s0, srcA[tid].s1, srcA[tid].s2, srcA[tid].s3, srcA[tid].s4, srcA[tid].s5, srcA[tid].s6, srcA[tid].s7, srcA[tid].s8, srcA[tid].s9, srcA[tid].sA, srcA[tid].sB, srcA[tid].sC, srcA[tid].sD, srcA[tid].sE, srcA[tid].sf))",
                                        test_str_names[type]
                                        );
                                break;
                            case 8:
                                sprintf(expression,
                                        "((%s8)(srcA[tid].s0, srcA[tid].s1, srcA[tid].s2, srcA[tid].s3, srcA[tid].s4, srcA[tid].s5, srcA[tid].s6, srcA[tid].s7))",
                                        test_str_names[type]
                                        );
                                break;
                            case 4:
                                sprintf(expression,
                                        "((%s4)(srcA[tid].s0, srcA[tid].s1, srcA[tid].s2, srcA[tid].s3))",
                                        test_str_names[type]
                                        );
                                break;
                            case 3:
                                sprintf(expression,
                                        "((%s3)(srcA[tid].s0, srcA[tid].s1, srcA[tid].s2))",
                                        test_str_names[type]
                                        );
                                break;
                            case 2:
                                sprintf(expression,
                                        "((%s2)(srcA[tid].s0, srcA[tid].s1))",
                                        test_str_names[type]
                                        );
                                break;
                            default :
                                sprintf(expression, "srcA[tid]");
                                log_info("Default\n");
                        }
                    } else {
                        sprintf(expression, "srcA[tid]");
                    }

                    if (0 == strcmp( test_str_names[type], "double" ))
                        source[0] = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

                    char kernelName[128];
                    snprintf( kernelName, sizeof( kernelName ), "test_%s_%s%s", operatorToUse_names[ operatorToUse ], test_str_names[type], vector_size_names[vectorSize] );
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

                    size_t size = elementCount / (vector_aligns[vectorSize]);
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
                    outP += kSizes[type] * ( ( vector_sizes[outVectorSize] ) -
                                            ( vector_sizes[ out_vector_idx[vectorSize] ] ) );
                    // was                outP += kSizes[type] * ( ( 1 << outVectorSize ) - ( 1 << ( vectorSize - 1 ) ) );
                    for( size_t e = 0; e < size; e++ )
                    {
                        if( CheckResults( inP, outP, 1, type, vectorSize, operatorToUse ) ) {

                            log_info("e is %d\n", (int)e);
                            fflush(stdout);
                            // break;
                            return -1;
                        }
                        inP += kSizes[type] * ( vector_aligns[vectorSize] );
                        outP += kSizes[type] * ( vector_aligns[outVectorSize] );
                    }

                    clReleaseKernel( kernel );
                    clReleaseProgram( program );
                    log_info( "." );
                    fflush( stdout );
                }
            }
        }

        clReleaseMemObject( streams[0] );
        clReleaseMemObject( streams[1] );
        log_info( "done\n" );
    }

    log_info("HiLoEO test passed\n");

    free(input_ptr);
    free(output_ptr);

    return err;
}

static int CheckResults( void *in, void *out, size_t elementCount, int type, int vectorSize, int operatorToUse )
{
    cl_ulong  array[8];
    void *p = array;
    size_t halfVectorSize  = vector_sizes[out_vector_idx[vectorSize]];
    size_t cmpVectorSize =  vector_sizes[out_vector_idx[vectorSize]];
    // was 1 << (vectorSize-1);
    OffsetFunc f = offsetFuncs[ operatorToUse ];
    size_t elementSize =  kSizes[type];

    if(vector_size_names[vectorSize][0] == '3') {
        if(operatorToUse_names[operatorToUse][0] == 'h' ||
           operatorToUse_names[operatorToUse][0] == 'o') // hi or odd
        {
            cmpVectorSize = 1; // special case for vec3 ignored values
        }
    }

    switch( elementSize )
    {
        case 1:
        {
            char *i = (char*)in;
            char *o = (char*)out;
            size_t j;
            cl_uint k;
            OffsetFunc f = offsetFuncs[ operatorToUse ];

            for( k = 0; k  < elementCount; k++ )
            {
                char *o2 = (char*)p;
                for( j = 0; j < halfVectorSize; j++ )
                    o2[j] = i[ f((int)j, (int)halfVectorSize*2) ];

                if( memcmp( o, o2, elementSize * cmpVectorSize ) )
                {
                    log_info( "\n%d) Failure for %s%s.%s { %d", k, test_str_names[type], vector_size_names[ vectorSize ], operatorToUse_names[ operatorToUse ], i[0] );
                    for( j = 1; j < halfVectorSize * 2; j++ )
                        log_info( ", %d", i[j] );
                    log_info( " } --> { %d", o[0] );
                    for( j = 1; j < halfVectorSize; j++ )
                        log_info( ", %d", o[j] );
                    log_info( " }\n" );
                    return -1;
                }
                i += 2 * halfVectorSize;
                o += halfVectorSize;
            }
        }
            break;

        case 2:
        {
            short *i = (short*)in;
            short *o = (short*)out;
            size_t j;
            cl_uint k;

            for( k = 0; k  < elementCount; k++ )
            {
                short *o2 = (short*)p;
                for( j = 0; j < halfVectorSize; j++ )
                    o2[j] = i[ f((int)j, (int)halfVectorSize*2) ];

                if( memcmp( o, o2, elementSize * cmpVectorSize ) )
                {
                    log_info( "\n%d) Failure for %s%s.%s { %d", k, test_str_names[type], vector_size_names[ vectorSize ], operatorToUse_names[ operatorToUse ], i[0] );
                    for( j = 1; j < halfVectorSize * 2; j++ )
                        log_info( ", %d", i[j] );
                    log_info( " } --> { %d", o[0] );
                    for( j = 1; j < halfVectorSize; j++ )
                        log_info( ", %d", o[j] );
                    log_info( " }\n" );
                    return -1;
                }
                i += 2 * halfVectorSize;
                o += halfVectorSize;
            }
        }
            break;

        case 4:
        {
            int *i = (int*)in;
            int *o = (int*)out;
            size_t j;
            cl_uint k;

            for( k = 0; k  < elementCount; k++ )
            {
                int *o2 = (int *)p;
                for( j = 0; j < halfVectorSize; j++ )
                    o2[j] = i[ f((int)j, (int)halfVectorSize*2) ];

                for( j = 0; j < cmpVectorSize; j++ )
        {
            /* Allow float nans to be binary different */
            if( memcmp( &o[j], &o2[j], elementSize ) && !((strcmp(test_str_names[type], "float") == 0) && isnan(((float *)o)[j]) && isnan(((float *)o2)[j])))
            {
                log_info( "\n%d) Failure for %s%s.%s { 0x%8.8x", k, test_str_names[type], vector_size_names[ vectorSize ], operatorToUse_names[ operatorToUse ], i[0] );
            for( j = 1; j < halfVectorSize * 2; j++ )
                log_info( ", 0x%8.8x", i[j] );
            log_info( " } --> { 0x%8.8x", o[0] );
            for( j = 1; j < halfVectorSize; j++ )
                log_info( ", 0x%8.8x", o[j] );
            log_info( " }\n" );
            return -1;
            }
        }
        i += 2 * halfVectorSize;
        o += halfVectorSize;
            }
        }
            break;

        case 8:
        {
            cl_ulong *i = (cl_ulong*)in;
            cl_ulong *o = (cl_ulong*)out;
            size_t j;
            cl_uint k;

            for( k = 0; k  < elementCount; k++ )
            {
                cl_ulong *o2 = (cl_ulong*)p;
                for( j = 0; j < halfVectorSize; j++ )
                    o2[j] = i[ f((int)j, (int)halfVectorSize*2) ];

                if( memcmp( o, o2, elementSize * cmpVectorSize ) )
                {
                    log_info( "\n%d) Failure for %s%s.%s { 0x%16.16llx", k, test_str_names[type], vector_size_names[ vectorSize ], operatorToUse_names[ operatorToUse ], i[0] );
                    for( j = 1; j < halfVectorSize * 2; j++ )
                        log_info( ", 0x%16.16llx", i[j] );
                    log_info( " } --> { 0x%16.16llx", o[0] );
                    for( j = 1; j < halfVectorSize; j++ )
                        log_info( ", 0x%16.16llx", o[j] );
                    log_info( " }\n" );
                    return -1;
                }
                i += 2 * halfVectorSize;
                o += halfVectorSize;
            }
        }
            break;

        default:
            log_info( "Internal error. Unknown data type\n" );
            return -2;
    }

    return 0;
}



