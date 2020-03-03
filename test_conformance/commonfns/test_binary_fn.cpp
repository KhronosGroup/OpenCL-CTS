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

const char *binary_fn_code_pattern =
"%s\n" /* optional pragma */
"__kernel void test_fn(__global %s%s *x, __global %s%s *y, __global %s%s *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = %s(x[tid], y[tid]);\n"
"}\n";

const char *binary_fn_code_pattern_v3 =
"%s\n" /* optional pragma */
"__kernel void test_fn(__global %s *x, __global %s *y, __global %s *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    vstore3(%s(vload3(tid,x), vload3(tid,y) ), tid, dst);\n"
"}\n";

const char *binary_fn_code_pattern_v3_scalar =
"%s\n" /* optional pragma */
"__kernel void test_fn(__global %s *x, __global %s *y, __global %s *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    vstore3(%s(vload3(tid,x), y[tid] ), tid, dst);\n"
"}\n";

int test_binary_fn( cl_device_id device, cl_context context, cl_command_queue queue, int n_elems,
                    const char *fnName, bool vectorSecondParam,
                    binary_verify_float_fn floatVerifyFn, binary_verify_double_fn doubleVerifyFn )
{
    cl_mem      streams[6];
    cl_float      *input_ptr[2], *output_ptr;
    cl_double     *input_ptr_double[2], *output_ptr_double=NULL;
    cl_program  *program;
    cl_kernel   *kernel;
    size_t threads[1];
    int num_elements;
    int err;
    int i, j;
    MTdata d;

      program = (cl_program*)malloc(sizeof(cl_program)*kTotalVecCount*2);
      kernel = (cl_kernel*)malloc(sizeof(cl_kernel)*kTotalVecCount*2);

    num_elements = n_elems * (1 << (kTotalVecCount-1));

    int test_double = 0;
    if(is_extension_available( device, "cl_khr_fp64" ))
    {
        log_info("Testing doubles.\n");
        test_double = 1;
    }

    for( i = 0; i < 2; i++ )
    {
        input_ptr[i] = (cl_float*)malloc(sizeof(cl_float) * num_elements);
        if (test_double) input_ptr_double[i] = (cl_double*)malloc(sizeof(cl_double) * num_elements);
    }
    output_ptr = (cl_float*)malloc(sizeof(cl_float) * num_elements);
    if (test_double) output_ptr_double = (cl_double*)malloc(sizeof(cl_double) * num_elements);

    for( i = 0; i < 3; i++ )
    {
        streams[ i ] = clCreateBuffer( context, (cl_mem_flags)(CL_MEM_READ_WRITE),  sizeof(cl_float) * num_elements, NULL, &err );
        test_error( err, "clCreateBuffer failed");
    }

    if (test_double)
        for( i = 3; i < 6; i++ )
        {
          streams[ i ] = clCreateBuffer( context, (cl_mem_flags)(CL_MEM_READ_WRITE),  sizeof(cl_double) * num_elements, NULL, &err );
          test_error( err, "clCreateBuffer failed");
        }

    d = init_genrand( gRandomSeed );
    for( j = 0; j < num_elements; j++ )
    {
        input_ptr[0][j] = get_random_float(-0x20000000, 0x20000000, d);
        input_ptr[1][j] = get_random_float(-0x20000000, 0x20000000, d);
        if (test_double)
        {
            input_ptr_double[0][j] = get_random_double(-0x20000000, 0x20000000, d);
            input_ptr_double[1][j] = get_random_double(-0x20000000, 0x20000000, d);
        }
    }
    free_mtdata(d);     d = NULL;

    for( i = 0; i < 2; i++ )
    {
        err = clEnqueueWriteBuffer( queue, streams[ i ], CL_TRUE, 0, sizeof( cl_float ) * num_elements, input_ptr[ i ], 0, NULL, NULL );
        test_error( err, "Unable to write input buffer" );

        if (test_double)
        {
          err = clEnqueueWriteBuffer( queue, streams[ 3 + i ], CL_TRUE, 0, sizeof( cl_double ) * num_elements, input_ptr_double[ i ], 0, NULL, NULL );
          test_error( err, "Unable to write input buffer" );
        }
    }

    for( i = 0; i < kTotalVecCount; i++ )
    {
        char programSrc[ 10240 ];
        char vecSizeNames[][ 3 ] = { "", "2", "4", "8", "16", "3" };

        if(i >= kVectorSizeCount) {
            // do vec3 print

            if(vectorSecondParam) {
            sprintf( programSrc,binary_fn_code_pattern_v3, "", "float", "float", "float", fnName );
        } else  {
            sprintf( programSrc,binary_fn_code_pattern_v3_scalar, "", "float", "float", "float", fnName );
            }
        } else  {
            // do regular
            sprintf( programSrc, binary_fn_code_pattern, "", "float", vecSizeNames[ i ], "float", vectorSecondParam ? vecSizeNames[ i ] : "", "float", vecSizeNames[ i ], fnName );
        }
        const char *ptr = programSrc;
        err = create_single_kernel_helper( context, &program[ i ], &kernel[ i ], 1, &ptr, "test_fn" );
        test_error( err, "Unable to create kernel" );

        if (test_double)
        {
        if(i >= kVectorSizeCount) {
        if(vectorSecondParam) {
            sprintf( programSrc, binary_fn_code_pattern_v3, "#pragma OPENCL EXTENSION cl_khr_fp64 : enable",
            "double",  "double",  "double",  fnName );
        } else {

        sprintf( programSrc, binary_fn_code_pattern_v3_scalar, "#pragma OPENCL EXTENSION cl_khr_fp64 : enable",
                 "double",  "double",  "double",  fnName );
        }
        } else {
        sprintf( programSrc, binary_fn_code_pattern, "#pragma OPENCL EXTENSION cl_khr_fp64 : enable",
            "double", vecSizeNames[ i ], "double", vectorSecondParam ? vecSizeNames[ i ] : "", "double", vecSizeNames[ i ], fnName );
        }
            ptr = programSrc;
            err = create_single_kernel_helper( context, &program[ kTotalVecCount + i ], &kernel[ kTotalVecCount + i ], 1, &ptr, "test_fn" );
            test_error( err, "Unable to create kernel" );
        }
    }

    for( i = 0; i < kTotalVecCount; i++ )
    {
        for( j = 0; j < 3; j++ )
        {
            err = clSetKernelArg( kernel[ i ], j, sizeof( streams[ j ] ), &streams[ j ] );
            test_error( err, "Unable to set kernel argument" );
        }

        threads[0] = (size_t)n_elems;

        err = clEnqueueNDRangeKernel( queue, kernel[i], 1, NULL, threads, NULL, 0, NULL, NULL );
        test_error( err, "Unable to execute kernel" );

        err = clEnqueueReadBuffer( queue, streams[2], true, 0, sizeof(cl_float)*num_elements, (void *)output_ptr, 0, NULL, NULL );
        test_error( err, "Unable to read results" );



        if( floatVerifyFn( input_ptr[0], input_ptr[1], output_ptr, n_elems, ((g_arrVecSizes[i])) ) )
        {
            log_error(" float%d%s test failed\n", ((g_arrVecSizes[i])), vectorSecondParam ? "" : ", float");
            err = -1;
        }
        else
        {
            log_info(" float%d%s test passed\n", ((g_arrVecSizes[i])), vectorSecondParam ? "" : ", float");
            err = 0;
        }

        if (err)
            break;
    }

    if (test_double)
    {
        for( i = 0; i < kTotalVecCount; i++ )
        {
            for( j = 0; j < 3; j++ )
            {
                err = clSetKernelArg( kernel[ kTotalVecCount + i ], j, sizeof( streams[ 3 + j ] ), &streams[ 3 + j ] );
                test_error( err, "Unable to set kernel argument" );
            }

            threads[0] = (size_t)n_elems;

            err = clEnqueueNDRangeKernel( queue, kernel[kTotalVecCount + i], 1, NULL, threads, NULL, 0, NULL, NULL );
            test_error( err, "Unable to execute kernel" );

            err = clEnqueueReadBuffer( queue, streams[5], CL_TRUE, 0, sizeof(cl_double)*num_elements, (void *)output_ptr_double, 0, NULL, NULL );
            test_error( err, "Unable to read results" );

            if( doubleVerifyFn( input_ptr_double[0], input_ptr_double[1], output_ptr_double, n_elems, ((g_arrVecSizes[i]))))
            {
                log_error(" double%d%s test failed\n", ((g_arrVecSizes[i])), vectorSecondParam ? "" : ", double");
                err = -1;
            }
            else
            {
                log_info(" double%d%s test passed\n", ((g_arrVecSizes[i])), vectorSecondParam ? "" : ", double");
                err = 0;
            }

            if (err)
            break;
        }
    }


    for( i = 0; i < ((test_double) ? 6 : 3); i++ )
    {
        clReleaseMemObject(streams[i]);
    }
    for (i=0; i < ((test_double) ? kTotalVecCount * 2 : kTotalVecCount) ; i++)
    {
        clReleaseKernel(kernel[i]);
        clReleaseProgram(program[i]);
    }
    free(input_ptr[0]);
    free(input_ptr[1]);
    free(output_ptr);
      free(program);
      free(kernel);

    if (test_double)
    {
        free(input_ptr_double[0]);
        free(input_ptr_double[1]);
        free(output_ptr_double);
    }

    return err;
}


