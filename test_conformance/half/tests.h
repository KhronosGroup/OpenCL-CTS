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
#ifndef TESTS_H
#define TESTS_H

#include <CL/cl.h>

typedef enum
{
	AS_Global,
	AS_Private,
	AS_Local,
	AS_Constant,
	AS_NumAddressSpaces
} AddressSpaceEnum;

extern const char *addressSpaceNames[AS_NumAddressSpaces];


int test_vload_half( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
int test_vloada_half( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
int test_vstore_half( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
int test_vstorea_half( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
int test_vstore_half_rte( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
int test_vstorea_half_rte( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
int test_vstore_half_rtz( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
int test_vstorea_half_rtz( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
int test_vstore_half_rtp( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
int test_vstorea_half_rtp( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
int test_vstore_half_rtn( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
int test_vstorea_half_rtn( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
int test_roundTrip( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );

typedef cl_ushort (*f2h)( float );
typedef cl_ushort (*d2h)( double );
int Test_vStoreHalf_private( cl_device_id device, f2h referenceFunc, d2h referenceDoubleFunc, const char *roundName );
int Test_vStoreaHalf_private( cl_device_id device, f2h referenceFunc, d2h referenceDoubleFunc, const char *roundName );

#endif /* TESTS_H */


