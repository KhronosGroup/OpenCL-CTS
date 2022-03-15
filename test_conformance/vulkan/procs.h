/******************************************************************
//
//  OpenCL Conformance Tests
// 
//  Copyright:	(c) 2008-2011 by Apple Inc. All Rights Reserved.
//
******************************************************************/

#include "../../test_common/harness/mt19937.h"

#pragma mark -
#pragma Misc tests

extern int test_vulkan_interop_buffer( cl_device_id device, cl_context context, 
  cl_command_queue queue, int num_elements );
extern int test_vulkan_interop_image( cl_device_id device, cl_context context, 
  cl_command_queue queue, int num_elements );
extern int test_consistency_external_buffer( cl_device_id device, cl_context context, 
  cl_command_queue queue, int num_elements );
extern int test_consistency_external_image( cl_device_id device, cl_context context, 
  cl_command_queue queue, int num_elements );
extern int test_consistency_external_semaphore( cl_device_id device, cl_context context, 
  cl_command_queue queue, int num_elements );
extern int test_platform_info( cl_device_id device, cl_context context, 
  cl_command_queue queue, int num_elements );
extern int test_device_info( cl_device_id device, cl_context context, 
  cl_command_queue queue, int num_elements );
