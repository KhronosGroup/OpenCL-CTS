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
#include <stdlib.h>
#include <string.h>
#include "harness/testHarness.h"

#if !defined(_WIN32)
#include <unistd.h>
#endif

#include "procs.h"

test_definition test_list[] = {
    ADD_TEST( timer_resolution_queries ),
    ADD_TEST( device_and_host_timers ),
};

test_status InitCL(cl_device_id device) {
	auto version = get_device_cl_version(device);
	auto expected_min_version = Version(2, 1);
	cl_platform_id platform;
	cl_ulong timer_res;
	cl_int error;

	if (version < expected_min_version)
	{
		version_expected_info("Test", expected_min_version.to_string().c_str(), version.to_string().c_str());
		return TEST_SKIP;
	}

	error = clGetDeviceInfo(device, CL_DEVICE_PLATFORM,
	                        sizeof(platform), &platform, NULL);
	if (error != CL_SUCCESS)
	{
		print_error(error, "Unable to get device platform");
		return TEST_FAIL;
	}

	error = clGetPlatformInfo(platform, CL_PLATFORM_HOST_TIMER_RESOLUTION,
	                          sizeof(timer_res), &timer_res, NULL);
	if (error != CL_SUCCESS)
	{
		print_error(error, "Unable to get host timer capabilities");
		return TEST_FAIL;
	}

    if ((timer_res == 0) && (version >= Version(3, 0)))
    {
        return TEST_SKIP;
    }

    return TEST_PASS;
}


const int test_num = ARRAY_SIZE( test_list );

int main(int argc, const char *argv[])
{
    return runTestHarnessWithCheck( argc, argv, test_num, test_list, false, 0, InitCL );
}

