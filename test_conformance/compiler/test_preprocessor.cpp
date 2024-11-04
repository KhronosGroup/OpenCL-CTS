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
#include "testBase.h"
#include "harness/os_helpers.h"

const char *define_kernel_code[] = {
" #define VALUE\n"
"__kernel void define_test(__global int *src, __global int *dstA, __global int *dstB)\n"
"{\n"
" int tid = get_global_id(0);\n"
"#ifdef VALUE\n"
" dstA[tid] = src[tid] * 2;\n"
"#else\n"
" dstA[tid] = src[tid] * 4;\n"
"#endif\n"
"\n"
"#undef VALUE\n"
"#ifdef VALUE\n"
" dstB[tid] = src[tid] * 2;\n"
"#else\n"
" dstB[tid] = src[tid] * 4;\n"
"#endif\n"
"\n"
"}\n"};




int test_preprocessor_define_udef(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {

  cl_int error;
  clKernelWrapper kernel;
  clProgramWrapper program;
  clMemWrapper buffer[3];
  cl_int *srcData, *resultData;
  int i;
  MTdata d;

  error = create_single_kernel_helper(context, &program, &kernel, 1, define_kernel_code, "define_test");
  if (error)
    return -1;

  buffer[0] = clCreateBuffer(context, CL_MEM_READ_WRITE, num_elements*sizeof(cl_int), NULL, &error);
  test_error( error, "clCreateBuffer failed");
  buffer[1] = clCreateBuffer(context, CL_MEM_READ_WRITE, num_elements*sizeof(cl_int), NULL, &error);
  test_error( error, "clCreateBuffer failed");
  buffer[2] = clCreateBuffer(context, CL_MEM_READ_WRITE, num_elements*sizeof(cl_int), NULL, &error);
  test_error( error, "clCreateBuffer failed");

  srcData = (cl_int*)malloc(sizeof(cl_int)*num_elements);
  if (srcData == NULL) {
    log_error("Failed to allocate storage for source data (%d cl_ints).\n", num_elements);
    return -1;
  }

  d = init_genrand( gRandomSeed );
  for (i=0; i<num_elements; i++)
    srcData[i] = (int)get_random_float(-1024, 1024,d);
  free_mtdata(d);   d = NULL;

  resultData = (cl_int*)malloc(sizeof(cl_int)*num_elements);
  if (resultData == NULL) {
    free(srcData);
    log_error("Failed to allocate storage for result data (%d cl_ints).\n", num_elements);
    return -1;
  }

  error = clSetKernelArg(kernel, 0, sizeof(buffer[0]), &buffer[0]);
  test_error(error, "clSetKernelArg failed");
  error = clSetKernelArg(kernel, 1, sizeof(buffer[1]), &buffer[1]);
  test_error(error, "clSetKernelArg failed");
  error = clSetKernelArg(kernel, 2, sizeof(buffer[2]), &buffer[2]);
  test_error(error, "clSetKernelArg failed");


  error = clEnqueueWriteBuffer(queue, buffer[0], CL_TRUE, 0, num_elements*sizeof(cl_int), srcData, 0, NULL, NULL);
  test_error(error, "clEnqueueWriteBuffer failed");

  size_t threads[3] = { (size_t)num_elements, 0, 0 };
  error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, threads, NULL, 0, NULL, NULL);
  test_error(error, "clEnqueueNDRangeKernel failed");

  error = clEnqueueReadBuffer(queue, buffer[1], CL_TRUE, 0, num_elements*sizeof(cl_int), resultData, 0, NULL, NULL);
  test_error(error, "clEnqueueReadBuffer failed");

  for (i=0; i<num_elements; i++)
    if (resultData[i] != srcData[i]*2) {
      free(srcData);
      free(resultData);
      return -1;
    }

  error = clEnqueueReadBuffer(queue, buffer[2], CL_TRUE, 0, num_elements*sizeof(cl_int), resultData, 0, NULL, NULL);
  test_error(error, "clEnqueueReadBuffer failed");

  for (i=0; i<num_elements; i++)
    if (resultData[i] != srcData[i]*4) {
      free(srcData);
      free(resultData);
      return -1;
    }

  free(srcData);
  free(resultData);
  return 0;
}


const char *include_kernel_code =
"#include \"%s\"\n"
"__kernel void include_test(__global int *src, __global int *dstA)\n"
"{\n"
" int tid = get_global_id(0);\n"
"#ifdef HEADER_FOUND\n"
" dstA[tid] = HEADER_FOUND;\n"
"#else\n"
" dstA[tid] = 0;\n"
"#endif\n"
"\n"
"}\n";


int test_preprocessor_include(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {

  cl_int error;
  clKernelWrapper kernel;
  clProgramWrapper program;
  clMemWrapper buffer[2];
  cl_int *resultData;
  int i;

  char include_dir[4096] = {0};
  char include_kernel[4096] = {0};

  char const * sep  = get_dir_sep();
  char const * path = get_exe_dir();

  /* Build with the include directory defined */
  sprintf(include_dir,"%s%sincludeTestDirectory%stestIncludeFile.h", path, sep, sep);
  sprintf(include_kernel, include_kernel_code, include_dir);
  free( (void *) sep );
  free( (void *) path );

  const char* test_kernel[] = { include_kernel, 0 };
  error = create_single_kernel_helper(context, &program, &kernel, 1, test_kernel, "include_test");
  if (error)
    return -1;

  buffer[0] = clCreateBuffer(context, CL_MEM_READ_WRITE, num_elements*sizeof(cl_int), NULL, &error);
  test_error( error, "clCreateBuffer failed");
  buffer[1] = clCreateBuffer(context, CL_MEM_READ_WRITE, num_elements*sizeof(cl_int), NULL, &error);
  test_error( error, "clCreateBuffer failed");

  resultData = (cl_int*)malloc(sizeof(cl_int)*num_elements);
  if (resultData == NULL) {
    log_error("Failed to allocate storage for result data (%d cl_ints).\n", num_elements);
    return -1;
  }

  error = clSetKernelArg(kernel, 0, sizeof(buffer[0]), &buffer[0]);
  test_error(error, "clSetKernelArg failed");
  error = clSetKernelArg(kernel, 1, sizeof(buffer[1]), &buffer[1]);
  test_error(error, "clSetKernelArg failed");

  size_t threads[3] = { (size_t)num_elements, 0, 0 };
  error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, threads, NULL, 0, NULL, NULL);
  test_error(error, "clEnqueueNDRangeKernel failed");

  error = clEnqueueReadBuffer(queue, buffer[1], CL_TRUE, 0, num_elements*sizeof(cl_int), resultData, 0, NULL, NULL);
  test_error(error, "clEnqueueReadBuffer failed");

  for (i=0; i<num_elements; i++)
    if (resultData[i] != 12) {
      free(resultData);
      return -1;
    }

  free(resultData);
  return 0;
}




const char *line_error_kernel_code[] = {
"__kernel void line_error_test(__global int *dstA)\n"
"{\n"
" int tid = get_global_id(0);\n"
"#line 124 \"fictitious/file/name.c\" \n"
"#error  some error\n"
" dstA[tid] = tid;\n"
"\n"
"}\n"};


int test_preprocessor_line_error(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {

  cl_int error, error2;
  clKernelWrapper kernel;
  clProgramWrapper program;
  clMemWrapper buffer[2];

  char buildLog[ 1024 * 128 ];

  log_info("test_preprocessor_line_error may report spurious ERRORS in the conformance log.\n");

  /* Create the program object from source */
  program = clCreateProgramWithSource( context, 1, line_error_kernel_code, NULL, &error );
  test_error(error, "clCreateProgramWithSource failed");

  /* Compile the program */
  error2 = clBuildProgram( program, 0, NULL, NULL, NULL, NULL );
  if (error2) {
    log_info("Build error detected at clBuildProgram.");
  } else {
    log_info("Error not reported by clBuildProgram.\n");
  }

  cl_build_status status;
  error = clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_STATUS, sizeof(status), &status, NULL);
  test_error(error, "clGetProgramBuildInfo failed for CL_PROGRAM_BUILD_STATUS");
  if (status != CL_BUILD_ERROR) {
    log_error("Build status did not return CL_BUILD_ERROR for a program with #error defined.\n");
    return -1;
  } else if (status == CL_BUILD_ERROR || error2) {
        error2 = clGetProgramBuildInfo( program, deviceID, CL_PROGRAM_BUILD_LOG, sizeof( buildLog ), buildLog, NULL );
        test_error( error2, "Unable to get program build log" );

    log_info("Build failed as expected with #error in source:\n");
        log_info( "Build log is: ------------\n" );
        log_info( "%s\n", buildLog );
        log_info( "Original source is: ------------\n" );
    log_info( "%s", line_error_kernel_code[0] );
        log_info( "\n----------\n" );

    if (strstr(buildLog, "fictitious/file/name.c")) {
      log_info("Found file name from #line param in log output.\n");
    } else {
      log_info("WARNING: Did not find file name from #line param in log output.\n");
    }

    if (strstr(buildLog, "124")) {
      log_info("Found line number from #line param in log output.\n");
    } else {
      log_info("WARNING: Did not find line number from #line param in log output.\n");
    }

    log_info("test_preprocessor_line_error PASSED.\n");
        return 0;
    }

    /* And create a kernel from it */
    kernel = clCreateKernel( program, "line_error_test", &error );
  test_error(error, "clCreateKernel failed");

  buffer[0] = clCreateBuffer(context, CL_MEM_READ_WRITE, num_elements*sizeof(cl_int), NULL, &error);
  test_error( error, "clCreateBuffer failed");

  error = clSetKernelArg(kernel, 0, sizeof(buffer[0]), &buffer[0]);
  test_error(error, "clSetKernelArg failed");

  size_t threads[3] = { (size_t)num_elements, 0, 0 };
  error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, threads, NULL, 0, NULL, NULL);
  test_error(error, "clEnqueueNDRangeKernel failed");

  log_error("Program built and ran with #error defined.");
  return -1;
}



const char *pragma_kernel_code[] = {
"__kernel void pragma_test(__global int *dstA)\n"
"{\n"
"#pragma A fool thinks himself to be wise, but a wise man knows himself to be a fool.\n"
" int tid = get_global_id(0);\n"
"#pragma\n"
" dstA[tid] = tid;\n"
"#pragma  mark Though I am not naturally honest, I am so sometimes by chance.\n"
"\n"
"}\n"};


int test_preprocessor_pragma(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {

  cl_int error;
  clKernelWrapper kernel;
  clProgramWrapper program;
  clMemWrapper buffer[2];
  cl_int *resultData;
  int i;

  error = create_single_kernel_helper(context, &program, &kernel, 1, pragma_kernel_code, "pragma_test");
  if (error)
    return -1;

  buffer[0] = clCreateBuffer(context, CL_MEM_READ_WRITE, num_elements*sizeof(cl_int), NULL, &error);
  test_error( error, "clCreateBuffer failed");

  error = clSetKernelArg(kernel, 0, sizeof(buffer[0]), &buffer[0]);
  test_error(error, "clSetKernelArg failed");

  size_t threads[3] = { (size_t)num_elements, 0, 0 };
  error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, threads, NULL, 0, NULL, NULL);
  test_error(error, "clEnqueueNDRangeKernel failed");

  resultData = (cl_int*)malloc(sizeof(cl_int)*num_elements);
  if (resultData == NULL) {
    log_error("Failed to allocate storage for result data (%d cl_ints).\n", num_elements);
    return -1;
  }

  error = clEnqueueReadBuffer(queue, buffer[0], CL_TRUE, 0, num_elements*sizeof(cl_int), resultData, 0, NULL, NULL);
  test_error(error, "clEnqueueReadBuffer failed");

  for (i=0; i<num_elements; i++)
    if (resultData[i] != i) {
      free(resultData);
      return -1;
    }

  free(resultData);
  return 0;
}



