//
// Copyright (c) 2023 The Khronos Group Inc.
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
#include <iomanip>
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <vector>

#include "procs.h"

int hi_offset( int index, int vectorSize) { return index + vectorSize / 2; }
int lo_offset( int index, int vectorSize) { return index; }
int even_offset( int index, int vectorSize ) { return index * 2; }
int odd_offset( int index, int vectorSize ) { return index * 2 + 1; }

typedef int (*OffsetFunc)( int index, int vectorSize );
static const OffsetFunc offsetFuncs[4] = { hi_offset, lo_offset, even_offset, odd_offset };
static const char *operatorToUse_names[] = { "hi", "lo", "even", "odd" };
static const char *test_str_names[] = { "char", "uchar", "short", "ushort",
                                        "int",  "uint",  "long",  "ulong",
                                        "half", "float", "double" };

static const unsigned int vector_sizes[] =     { 1, 2, 3, 4, 8, 16};
static const unsigned int vector_aligns[] =    { 1, 2, 4, 4, 8, 16};
static const unsigned int out_vector_idx[] =   { 0, 0, 1, 1, 3, 4};
// if input is size vector_sizes[i], output is size
// vector_sizes[out_vector_idx[i]]
// input type name is strcat(gentype, vector_size_names[i]);
// and output type name is
// strcat(gentype, vector_size_names[out_vector_idx[i]]);
static const char *vector_size_names[] = { "", "2", "3", "4", "8", "16"};

static const size_t kSizes[] = { 1, 1, 2, 2, 4, 4, 8, 8, 2, 4, 8 };
static int CheckResults( void *in, void *out, size_t elementCount, int type, int vectorSize, int operatorToUse );

int test_hiloeo(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
    int err;
    int hasDouble = is_extension_available( device, "cl_khr_fp64" );
    int hasHalf = is_extension_available(device, "cl_khr_fp16");
    cl_uint vectorSize, operatorToUse;
    cl_uint type;
    MTdataHolder d(gRandomSeed);

    int expressionMode;
    int numExpressionModes = 2;

    size_t length = sizeof(cl_int) * 4 * n_elems;

    std::vector<cl_int> input_ptr(4 * n_elems);
    std::vector<cl_int> output_ptr(4 * n_elems);

    for (cl_uint i = 0; i < 4 * (cl_uint)n_elems; i++)
        input_ptr[i] = genrand_int32(d);

    for( type = 0; type < sizeof( test_str_names ) / sizeof( test_str_names[0] ); type++ )
    {
        // Note: restrict the element count here so we don't end up overrunning the output buffer if we're compensating for 32-bit writes
        size_t elementCount = length / kSizes[type];
        clMemWrapper streams[2];

        // skip double if unavailable
        if( !hasDouble && ( 0 == strcmp( test_str_names[type], "double" )))
            continue;

        if (!hasHalf && (0 == strcmp(test_str_names[type], "half"))) continue;

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

        err = clEnqueueWriteBuffer(queue, streams[0], CL_TRUE, 0, length,
                                   input_ptr.data(), 0, NULL, NULL);
        test_error(err, "clEnqueueWriteBuffer failed\n");

        for( operatorToUse = 0; operatorToUse < sizeof( operatorToUse_names ) / sizeof( operatorToUse_names[0] ); operatorToUse++ )
        {
            log_info( " %s", operatorToUse_names[ operatorToUse ] );
            fflush( stdout );
            for( vectorSize = 1; vectorSize < sizeof( vector_size_names ) / sizeof( vector_size_names[0] ); vectorSize++ ) {
                for(expressionMode = 0; expressionMode < numExpressionModes; ++expressionMode) {

                    clProgramWrapper program;
                    clKernelWrapper kernel;
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

                    if (expressionMode == 1 && vector_sizes[vectorSize] != 1)
                    {
                        std::ostringstream sstr;
                        const char *index_chars[] = { "0", "1", "2", "3",
                                                      "4", "5", "6", "7",
                                                      "8", "9", "A", "B",
                                                      "C", "D", "E", "f" };
                        sstr << "((" << test_str_names[type]
                             << std::to_string(vector_sizes[vectorSize])
                             << ")(";
                        for (unsigned i = 0; i < vector_sizes[vectorSize]; i++)
                            sstr << " srcA[tid].s" << index_chars[i] << ",";
                        sstr.seekp(-1, sstr.cur);
                        sstr << "))";
                        std::snprintf(expression, sizeof(expression), "%s",
                                      sstr.str().c_str());
                    }
                    else
                    {
                        std::snprintf(expression, sizeof(expression),
                                      "srcA[tid]");
                    }

                    if (0 == strcmp( test_str_names[type], "double" ))
                        source[0] = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

                    if (0 == strcmp(test_str_names[type], "half"))
                        source[0] =
                            "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";

                    char kernelName[128];
                    snprintf( kernelName, sizeof( kernelName ), "test_%s_%s%s", operatorToUse_names[ operatorToUse ], test_str_names[type], vector_size_names[vectorSize] );
                    err = create_single_kernel_helper(context, &program, &kernel, sizeof( source ) / sizeof( source[0] ), source, kernelName );
                    test_error(err, "create_single_kernel_helper failed\n");

                    err  = clSetKernelArg(kernel, 0, sizeof streams[0], &streams[0]);
                    err |= clSetKernelArg(kernel, 1, sizeof streams[1], &streams[1]);
                    test_error(err, "clSetKernelArg failed\n");

                    //Wipe the output buffer clean
                    uint32_t pattern = 0xdeadbeef;
                    memset_pattern4(output_ptr.data(), &pattern, length);
                    err = clEnqueueWriteBuffer(queue, streams[1], CL_TRUE, 0,
                                               length, output_ptr.data(), 0,
                                               NULL, NULL);
                    test_error(err, "clEnqueueWriteBuffer failed\n");

                    size_t size = elementCount / (vector_aligns[vectorSize]);
                    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &size, NULL, 0, NULL, NULL);
                    test_error(err, "clEnqueueNDRangeKernel failed\n");

                    err = clEnqueueReadBuffer(queue, streams[1], CL_TRUE, 0,
                                              length, output_ptr.data(), 0,
                                              NULL, NULL);
                    test_error(err, "clEnqueueReadBuffer failed\n");

                    char *inP = (char *)input_ptr.data();
                    char *outP = (char *)output_ptr.data();
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
                    log_info( "." );
                    fflush( stdout );
                }
            }
        }
        log_info( "done\n" );
    }

    log_info("HiLoEO test passed\n");
    return err;
}

template <typename T>
cl_int verify(void *in, void *out, size_t elementCount, int type,
              int vectorSize, int operatorToUse, size_t cmpVectorSize)
{
    size_t halfVectorSize = vector_sizes[out_vector_idx[vectorSize]];
    size_t elementSize = kSizes[type];
    OffsetFunc f = offsetFuncs[operatorToUse];
    cl_ulong array[8];
    void *p = array;

    std::ostringstream ss;

    T *i = (T *)in, *o = (T *)out;

    for (cl_uint k = 0; k < elementCount; k++)
    {
        T *o2 = (T *)p;
        for (size_t j = 0; j < halfVectorSize; j++)
            o2[j] = i[f((int)j, (int)halfVectorSize * 2)];

        if (memcmp(o, o2, elementSize * cmpVectorSize))
        {
            ss << "\n"
               << k << ") Failure for" << test_str_names[type]
               << vector_size_names[vectorSize] << '.'
               << operatorToUse_names[operatorToUse] << " { "
               << "0x" << std::setfill('0') << std::setw(elementSize * 2)
               << std::hex << i[0];

            for (size_t j = 1; j < halfVectorSize * 2; j++) ss << ", " << i[j];
            ss << " } --> { " << o[0];
            for (size_t j = 1; j < halfVectorSize; j++) ss << ", " << o[j];
            ss << " }\n";
            return -1;
        }
        i += 2 * halfVectorSize;
        o += halfVectorSize;
    }
    return 0;
}

static int CheckResults(void *in, void *out, size_t elementCount, int type,
                        int vectorSize, int operatorToUse)
{
    size_t cmpVectorSize = vector_sizes[out_vector_idx[vectorSize]];
    size_t elementSize = kSizes[type];

    if (vector_size_names[vectorSize][0] == '3')
    {
        if (operatorToUse_names[operatorToUse][0] == 'h'
            || operatorToUse_names[operatorToUse][0] == 'o') // hi or odd
        {
            cmpVectorSize = 1; // special case for vec3 ignored values
        }
    }

    switch (elementSize)
    {
        case 1:
            return verify<char>(in, out, elementCount, type, vectorSize,
                                operatorToUse, cmpVectorSize);
        case 2:
            return verify<short>(in, out, elementCount, type, vectorSize,
                                 operatorToUse, cmpVectorSize);
        case 4:
            return verify<int>(in, out, elementCount, type, vectorSize,
                               operatorToUse, cmpVectorSize);
        case 8:
            return verify<cl_ulong>(in, out, elementCount, type, vectorSize,
                                    operatorToUse, cmpVectorSize);
        default: log_info("Internal error. Unknown data type\n"); return -2;
    }
}
