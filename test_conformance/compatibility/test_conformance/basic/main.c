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

#if !defined(_WIN32)
#include <unistd.h>
#endif

#include <stdio.h>
#include <string.h>
#include "harness/testHarness.h"
#include "procs.h"

// FIXME: To use certain functions in harness/imageHelpers.h
// (for example, generate_random_image_data()), the tests are required to declare
// the following variables:
cl_device_type gDeviceType = CL_DEVICE_TYPE_DEFAULT;
bool gTestRounding = false;

test_definition test_list[] = {
    ADD_TEST( sizeof ),
    ADD_TEST( readimage ),
    ADD_TEST( readimage_int16 ),
    ADD_TEST( readimage_fp32 ),
    ADD_TEST( writeimage ),
    ADD_TEST( mri_one ),

    ADD_TEST( mri_multiple ),
    ADD_TEST( image_r8 ),
    ADD_TEST( readimage3d ),
    ADD_TEST( readimage3d_int16 ),
    ADD_TEST( readimage3d_fp32 ),
    ADD_TEST( bufferreadwriterect ),
    ADD_TEST( imagearraycopy3d ),
    ADD_TEST( imagenpot ),

    ADD_TEST( imagedim_pow2 ),
    ADD_TEST( imagedim_non_pow2 ),
    ADD_TEST( image_param ),
    ADD_TEST( image_multipass_integer_coord ),
    ADD_TEST( image_multipass_float_coord ),

    ADD_TEST( async_copy_global_to_local ),
    ADD_TEST( async_copy_local_to_global ),
    ADD_TEST( async_strided_copy_global_to_local ),
    ADD_TEST( async_strided_copy_local_to_global ),
    ADD_TEST( prefetch ),
};

const int test_num = ARRAY_SIZE( test_list );

int main(int argc, const char *argv[])
{
    return runTestHarness( argc, argv, test_num, test_list, false, false, 0 );
}

