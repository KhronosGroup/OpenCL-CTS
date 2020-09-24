//
// Copyright (c) 2020 The Khronos Group Inc.
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
#include "featureHelpers.h"
#include "errorHelpers.h"

#include <assert.h>
#include <string.h>

#include <vector>

int get_device_cl_c_features(cl_device_id device, OpenCLCFeatures& features)
{
    // Initially, all features are unsupported.
    features = { 0 };

    // The CL_DEVICE_OPENCL_C_FEATURES query does not exist pre-3.0.
    const Version version = get_device_cl_version(device);
    if (version < Version(3, 0))
    {
        return TEST_PASS;
    }

    cl_int error = CL_SUCCESS;

    size_t sz = 0;
    error = clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_FEATURES, 0, NULL, &sz);
    test_error(error, "Unable to query CL_DEVICE_OPENCL_C_FEATURES size");

    std::vector<cl_name_version> clc_features(sz / sizeof(cl_name_version));
    error = clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_FEATURES, sz,
                            clc_features.data(), NULL);
    test_error(error, "Unable to query CL_DEVICE_OPENCL_C_FEATURES");

#define CHECK_OPENCL_C_FEATURE(_feature)                                       \
    if (strcmp(clc_feature.name, #_feature) == 0)                              \
    {                                                                          \
        features.supports##_feature = true;                                    \
    }

    for (const auto& clc_feature : clc_features)
    {
        CHECK_OPENCL_C_FEATURE(__opencl_c_3d_image_writes);
        CHECK_OPENCL_C_FEATURE(__opencl_c_atomic_order_acq_rel);
        CHECK_OPENCL_C_FEATURE(__opencl_c_atomic_order_seq_cst);
        CHECK_OPENCL_C_FEATURE(__opencl_c_atomic_scope_device);
        CHECK_OPENCL_C_FEATURE(__opencl_c_atomic_scope_all_devices);
        CHECK_OPENCL_C_FEATURE(__opencl_c_device_enqueue);
        CHECK_OPENCL_C_FEATURE(__opencl_c_generic_address_space);
        CHECK_OPENCL_C_FEATURE(__opencl_c_fp64);
        CHECK_OPENCL_C_FEATURE(__opencl_c_images);
        CHECK_OPENCL_C_FEATURE(__opencl_c_int64);
        CHECK_OPENCL_C_FEATURE(__opencl_c_pipes);
        CHECK_OPENCL_C_FEATURE(__opencl_c_program_scope_global_variables);
        CHECK_OPENCL_C_FEATURE(__opencl_c_read_write_images);
        CHECK_OPENCL_C_FEATURE(__opencl_c_subgroups);
        CHECK_OPENCL_C_FEATURE(__opencl_c_work_group_collective_functions);
    }

#undef CHECK_OPENCL_C_FEATURE

    return TEST_PASS;
}
