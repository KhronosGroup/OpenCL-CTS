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

#include "errorHelpers.h"
#include "feature.h"

// Base framework
const feature_or feature::operator||(const feature& rhs) const
{
    return feature_or(*this, rhs);
}

const feature_and feature::operator&&(const feature& rhs) const
{
    return feature_and(*this, rhs);
}

// Base feature classes
template <const cl_device_info query, typename T>
struct F_device_info_non_zero : public feature
{
    F_device_info_non_zero(const std::string& name)
    {
        m_name = name;
        m_predicate = [](cl_device_id device) {
            T result;
            auto err = clGetDeviceInfo(device, query, sizeof(result), &result,
                                       nullptr);
            ASSERT_SUCCESS(err, "clGetDeviceInfo");
            return result != 0;
        };
    }
};

// Feature definitions
const feature& F_subgroups_core =
    F_device_info_non_zero<CL_DEVICE_MAX_NUM_SUB_GROUPS, cl_uint>(
        "Subgroups core");

const feature& F_subgroups_extension = F_extension("cl_khr_subgroups");

const feature& F_subgroups = F_subgroups_extension
    || (F_version_ge(2, 1) && F_version_lt(3, 0))
    || (F_version_ge(3, 0) && F_subgroups_core);

const feature& F_program_scope_variables_core =
    F_device_info_non_zero<CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE, size_t>(
        "Program scope variables core");

const feature& F_program_scope_variables =
    (F_version_ge(2, 0) && F_version_lt(3, 0))
    || (F_version_ge(3, 0) && F_program_scope_variables_core);

const feature& F_non_uniform_work_groups_core =
    F_device_info_non_zero<CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT, cl_bool>(
        "Non-uniform work-groups core");

const feature& F_non_uniform_work_groups =
    (F_version_ge(2, 0) && F_version_lt(3, 0))
    || (F_version_ge(3, 0) && F_non_uniform_work_groups_core);

const feature& F_read_write_images_core =
    F_device_info_non_zero<CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS, cl_uint>(
        "Read-write images core");

const feature& F_read_write_images = (F_version_ge(2, 0) && F_version_lt(3, 0))
    || (F_version_ge(3, 0) && F_read_write_images_core);
