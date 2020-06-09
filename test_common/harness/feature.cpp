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

// Feature definitions
struct feature_subgroups_core_optional : public feature
{
    feature_subgroups_core_optional()
    {
        m_name = "Subgroups core optional";
        m_predicate = [](cl_device_id device) {
            cl_uint max_sub_groups;
            auto err =
                clGetDeviceInfo(device, CL_DEVICE_MAX_NUM_SUB_GROUPS,
                                sizeof(max_sub_groups), &max_sub_groups, NULL);
            ASSERT_SUCCESS(err, "clGetDeviceInfo");
            return max_sub_groups != 0;
        };
    }
} F_subgroups_core_optional;

const feature& F_subgroups_extension = F_extension("cl_khr_subgroups");

const feature& F_subgroups = F_subgroups_extension
    || (F_version_ge(2, 1) && F_version_lt(3, 0))
    || (F_version_ge(3, 0) && F_subgroups_core_optional);

struct feature_program_scope_variables_core_optional : public feature
{
    feature_program_scope_variables_core_optional()
    {
        m_name = "Program scope variables core optional";
        m_predicate = [](cl_device_id device) {
            size_t max_size;
            auto err =
                clGetDeviceInfo(device, CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE,
                                sizeof(max_size), &max_size, nullptr);
            ASSERT_SUCCESS(err, "clGetDeviceInfo");
            return max_size > 0;
        };
    }
} F_program_scope_variables_core_optional;

const feature& F_program_scope_variables =
    (F_version_ge(2, 0) && F_version_lt(3, 0))
    || (F_version_ge(3, 0) && F_program_scope_variables_core_optional);

struct feature_non_uniform_work_groups_core_optional : public feature
{
    feature_non_uniform_work_groups_core_optional()
    {
        m_name = "Non-uniform work-groups core optional";
        m_predicate = [](cl_device_id device) {
            cl_bool supported;
            auto err = clGetDeviceInfo(device,
                                       CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT,
                                       sizeof(supported), &supported, nullptr);
            ASSERT_SUCCESS(err, "clGetDeviceInfo");
            return supported;
        };
    }
} F_non_uniform_work_groups_core_optional;

const feature& F_non_uniform_work_groups =
    (F_version_ge(2, 0) && F_version_lt(3, 0))
    || (F_version_ge(3, 0) && F_non_uniform_work_groups_core_optional);

struct feature_read_write_images_core_optional : public feature
{
    feature_read_write_images_core_optional()
    {
        m_name = "Read-write images core optional";
        m_predicate = [](cl_device_id device) {
            cl_uint num_args;
            auto err =
                clGetDeviceInfo(device, CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS,
                                sizeof(num_args), &num_args, nullptr);
            ASSERT_SUCCESS(err, "clGetDeviceInfo");
            return num_args > 0;
        };
    }
} F_read_write_images_core_optional;

const feature& F_read_write_images = (F_version_ge(2, 0) && F_version_lt(3, 0))
    || (F_version_ge(3, 0) && F_read_write_images_core_optional);
