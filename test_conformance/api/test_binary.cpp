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

static const char *sample_binary_kernel_source[] = {
"__kernel void sample_test(__global float *src, __global int *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = (int)src[tid] + 1;\n"
"\n"
"}\n" };


int test_binary_get(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
	int error;
	clProgramWrapper program;
	size_t			binarySize;
	

	program = clCreateProgramWithSource( context, 1, sample_binary_kernel_source, NULL, &error );
	test_error( error, "Unable to create program from source" );
	
	// Build so we have a binary to get
	error = clBuildProgram( program, 1, &deviceID, NULL, NULL, NULL );
	test_error( error, "Unable to build test program" );
	
	// Get the size of the resulting binary (only one device)
	error = clGetProgramInfo( program, CL_PROGRAM_BINARY_SIZES, sizeof( binarySize ), &binarySize, NULL );
	test_error( error, "Unable to get binary size" );

	// Sanity check
	if( binarySize == 0 )
	{
		log_error( "ERROR: Binary size of program is zero\n" );
		return -1;
	}
	
	// Create a buffer and get the actual binary
	unsigned char *binary;
  binary = (unsigned char*)malloc(sizeof(unsigned char)*binarySize);
	unsigned char *buffers[ 1 ] = { binary };

	// Do another sanity check here first
	size_t size;
	error = clGetProgramInfo( program, CL_PROGRAM_BINARIES, 0, NULL, &size );
	test_error( error, "Unable to get expected size of binaries array" );
	if( size != sizeof( buffers ) )
	{
		log_error( "ERROR: Expected size of binaries array in clGetProgramInfo is incorrect (should be %d, got %d)\n", (int)sizeof( buffers ), (int)size );
		free(binary);
    return -1;
	}
	
	error = clGetProgramInfo( program, CL_PROGRAM_BINARIES, sizeof( buffers ), &buffers, NULL );
	test_error( error, "Unable to get program binary" );
	
	// No way to verify the binary is correct, so just be good with that
  free(binary);
	return 0;
}
	

int test_program_binary_create(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
	/* To test this in a self-contained fashion, we have to create a program with 
   source, then get the binary, then use that binary to reload the program, and then verify */
	
	int error;
	clProgramWrapper program, program_from_binary;
	size_t			binarySize;
	
	
	program = clCreateProgramWithSource( context, 1, sample_binary_kernel_source, NULL, &error );
	test_error( error, "Unable to create program from source" );
	
	// Build so we have a binary to get
	error = clBuildProgram( program, 1, &deviceID, NULL, NULL, NULL );
	test_error( error, "Unable to build test program" );
	
	// Get the size of the resulting binary (only one device)
	error = clGetProgramInfo( program, CL_PROGRAM_BINARY_SIZES, sizeof( binarySize ), &binarySize, NULL );
	test_error( error, "Unable to get binary size" );
	
	// Sanity check
	if( binarySize == 0 )
	{
		log_error( "ERROR: Binary size of program is zero\n" );
		return -1;
	}
	
	// Create a buffer and get the actual binary
	unsigned char *binary;
    binary = (unsigned char*)malloc(sizeof(unsigned char)*binarySize);
	const unsigned char *buffers[ 1 ] = { binary };
	
	error = clGetProgramInfo( program, CL_PROGRAM_BINARIES, sizeof( buffers ), &buffers, NULL );
	test_error( error, "Unable to get program binary" );
	
	cl_int loadErrors[ 1 ];
	program_from_binary = clCreateProgramWithBinary( context, 1, &deviceID, &binarySize, buffers, loadErrors, &error );
	test_error( error, "Unable to load valid program binary" );	
	test_error( loadErrors[ 0 ], "Unable to load valid device binary into program" );
	
  error = clBuildProgram( program_from_binary, 1, &deviceID, NULL, NULL, NULL );
  test_error( error, "Unable to build binary program" );
    
	// Now get the binary one more time and verify it loaded the right binary
	unsigned char *binary2;
    binary2 = (unsigned char*)malloc(sizeof(unsigned char)*binarySize);
	buffers[ 0 ] = binary2;
	error = clGetProgramInfo( program_from_binary, CL_PROGRAM_BINARIES, sizeof( buffers ), &buffers, NULL );
	test_error( error, "Unable to get program binary second time" );
	
	if( memcmp( binary, binary2, binarySize ) != 0 )
	{
		log_error( "ERROR: Program binary is different when loaded from binary!\n" );
    free(binary2);
    free(binary);
		return -1;
	}
  
	// Try again, this time without passing the status ptr in, to make sure we still 
	// get a valid binary
	clProgramWrapper programWithoutStatus = clCreateProgramWithBinary( context, 1, &deviceID, &binarySize, buffers, NULL, &error );
	test_error( error, "Unable to load valid program binary when binary_status pointer is NULL" );	
	
	error = clBuildProgram( programWithoutStatus, 1, &deviceID, NULL, NULL, NULL );
	test_error( error, "Unable to build binary program" );
    
	// Now get the binary one more time and verify it loaded the right binary
	unsigned char *binary3;
    binary3 = (unsigned char*)malloc(sizeof(unsigned char)*binarySize);
	buffers[ 0 ] = binary3;
	error = clGetProgramInfo( program_from_binary, CL_PROGRAM_BINARIES, sizeof( buffers ), &buffers, NULL );
	test_error( error, "Unable to get program binary second time" );
	
	if( memcmp( binary, binary3, binarySize ) != 0 )
	{
		log_error( "ERROR: Program binary is different when status pointer is NULL!\n" );
		free(binary3);
		free(binary2);
		free(binary);
		return -1;
	}
	free(binary3);
	
  // Now execute them both to see that they both do the same thing.
  clMemWrapper in, out, out_binary;
  clKernelWrapper kernel, kernel_binary;
  cl_int *out_data, *out_data_binary;
  cl_float *in_data;
  size_t size_to_run = 1000;
  
  // Allocate some data
  in_data = (cl_float*)malloc(sizeof(cl_float)*size_to_run);
  out_data = (cl_int*)malloc(sizeof(cl_int)*size_to_run);
  out_data_binary = (cl_int*)malloc(sizeof(cl_int)*size_to_run);
  memset(out_data, 0, sizeof(cl_int)*size_to_run);
  memset(out_data_binary, 0, sizeof(cl_int)*size_to_run);
  for (size_t i=0; i<size_to_run; i++)
    in_data[i] = (cl_float)i;
  
  // Create the buffers
  in = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, sizeof(cl_float)*size_to_run, in_data, &error);
  test_error( error, "clCreateBuffer failed");
  out = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, sizeof(cl_int)*size_to_run, out_data, &error);
  test_error( error, "clCreateBuffer failed");
  out_binary = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, sizeof(cl_int)*size_to_run, out_data_binary, &error);
  test_error( error, "clCreateBuffer failed");

  // Create the kernels
  kernel = clCreateKernel(program, "sample_test", &error);
  test_error( error, "clCreateKernel failed");
  kernel_binary = clCreateKernel(program_from_binary, "sample_test", &error);
  test_error( error, "clCreateKernel from binary failed");
  
  // Set the arguments
  error = clSetKernelArg(kernel, 0, sizeof(in), &in);
  test_error( error, "clSetKernelArg failed");
  error = clSetKernelArg(kernel, 1, sizeof(out), &out);
  test_error( error, "clSetKernelArg failed");
  error = clSetKernelArg(kernel_binary, 0, sizeof(in), &in);
  test_error( error, "clSetKernelArg failed");
  error = clSetKernelArg(kernel_binary, 1, sizeof(out_binary), &out_binary);
  test_error( error, "clSetKernelArg failed");
  
  // Execute the kernels
  error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &size_to_run, NULL, 0, NULL, NULL);
  test_error( error, "clEnqueueNDRangeKernel failed");
  error = clEnqueueNDRangeKernel(queue, kernel_binary, 1, NULL, &size_to_run, NULL, 0, NULL, NULL);
  test_error( error, "clEnqueueNDRangeKernel for binary kernel failed");

  // Finish up
  error = clFinish(queue);
  test_error( error, "clFinish failed");
  
  // Get the results back
  error = clEnqueueReadBuffer(queue, out, CL_TRUE, 0, sizeof(cl_int)*size_to_run, out_data, 0, NULL, NULL);
  test_error( error, "clEnqueueReadBuffer failed");
  error = clEnqueueReadBuffer(queue, out_binary, CL_TRUE, 0, sizeof(cl_int)*size_to_run, out_data_binary, 0, NULL, NULL);
  test_error( error, "clEnqueueReadBuffer failed");

  // Compare the results
	if( memcmp( out_data, out_data_binary, sizeof(cl_int)*size_to_run ) != 0 )
	{
		log_error( "ERROR: Results from executing binary and regular kernel differ.\n" );
    free(binary2);
    free(binary);
		return -1;
	}
	
	// All done!
  free(in_data);
  free(out_data);
  free(out_data_binary);
  free(binary2);
  free(binary);
	return 0;
}


