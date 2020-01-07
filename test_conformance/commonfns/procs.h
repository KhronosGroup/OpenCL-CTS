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
#include "harness/testHarness.h"
#include "harness/kernelHelpers.h"
#include "harness/errorHelpers.h"
#include "harness/conversions.h"
#include "harness/mt19937.h"

#define kVectorSizeCount 5
#define kStrangeVectorSizeCount 1
#define kTotalVecCount (kVectorSizeCount + kStrangeVectorSizeCount)

extern int g_arrVecSizes[kVectorSizeCount + kStrangeVectorSizeCount];
// int g_arrStrangeVectorSizes[kStrangeVectorSizeCount] = {3};

extern int        test_clamp(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_degrees(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_fmax(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_fmaxf(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_fmin(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_fminf(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_max(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_maxf(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_min(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_minf(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_mix(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_radians(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_step(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_stepf(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_smoothstep(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_smoothstepf(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_sign(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements);

typedef int     (*binary_verify_float_fn)( float *x, float *y, float *out, int numElements, int vecSize );
typedef int     (*binary_verify_double_fn)( double *x, double *y, double *out, int numElements, int vecSize );

extern int      test_binary_fn( cl_device_id device, cl_context context, cl_command_queue queue, int n_elems,
                           const char *fnName, bool vectorSecondParam,
                           binary_verify_float_fn floatVerifyFn, binary_verify_double_fn doubleVerifyFn );


