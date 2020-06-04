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

#ifndef HARNESS_FEATURE_H_
#define HARNESS_FEATURE_H_

#include "deviceInfo.h"
#include "version.h"

#include <functional>
#include <string>

using feature_predicate = std::function<bool(cl_device_id)>;

struct feature_or;
struct feature_and;

struct feature
{
    const std::string& name() const { return m_name; };
    bool is_supported(cl_device_id device) const { return m_predicate(device); }
    const feature_or operator||(const feature& rhs) const;
    const feature_and operator&&(const feature& rhs) const;
    feature_predicate predicate() const { return m_predicate; }

protected:
    feature_predicate m_predicate;
    std::string m_name;
};

struct feature_or : public feature
{
    feature_or(const feature& lhs, const feature& rhs)
    {
        auto const lhs_pred = lhs.predicate();
        auto const rhs_pred = rhs.predicate();
        m_predicate = [lhs_pred, rhs_pred](cl_device_id device) -> bool {
            auto lhs = lhs_pred(device);
            auto rhs = rhs_pred(device);
            return lhs || rhs;
        };
        m_name = "(" + lhs.name() + " || " + rhs.name() + ")";
    }
};

struct feature_and : public feature
{
    feature_and(const feature& lhs, const feature& rhs)
    {
        auto const lhs_pred = lhs.predicate();
        auto const rhs_pred = rhs.predicate();
        m_predicate = [lhs_pred, rhs_pred](cl_device_id device) -> bool {
            auto lhs = lhs_pred(device);
            if (!lhs)
            {
                return false;
            }
            auto rhs = rhs_pred(device);
            if (!rhs)
            {
                return false;
            }
            return true;
        };
        m_name = "(" + lhs.name() + " && " + rhs.name() + ")";
    }
};

struct F_version_eq : public feature
{
    F_version_eq(int major, int minor)
    {
        auto version = Version(major, minor);
        m_predicate = [version](cl_device_id device) {
            auto devver = get_device_cl_version(device);
            return devver == version;
        };
        m_name = "Version == " + version.to_string();
    }
};

struct F_version_ge : public feature
{
    F_version_ge(int major, int minor)
    {
        auto version = Version(major, minor);
        m_predicate = [version](cl_device_id device) {
            auto devver = get_device_cl_version(device);
            return devver >= version;
        };
        m_name = "Version >= " + version.to_string();
    }
};

struct F_version_gt : public feature
{
    F_version_gt(int major, int minor)
    {
        auto version = Version(major, minor);
        m_predicate = [version](cl_device_id device) {
            auto devver = get_device_cl_version(device);
            return devver > version;
        };
        m_name = "Version > " + version.to_string();
    }
};

struct F_version_le : public feature
{
    F_version_le(int major, int minor)
    {
        auto version = Version(major, minor);
        m_predicate = [version](cl_device_id device) {
            auto devver = get_device_cl_version(device);
            return devver <= version;
        };
        m_name = "Version <= " + version.to_string();
    }
};

struct F_version_lt : public feature
{
    F_version_lt(int major, int minor)
    {
        auto version = Version(major, minor);
        m_predicate = [version](cl_device_id device) {
            auto devver = get_device_cl_version(device);
            return devver < version;
        };
        m_name = "version < " + version.to_string();
    }
};

struct F_extension : public feature
{
    F_extension(const std::string& name): feature()
    {
        m_name = "extension(" + name + ")";
        m_predicate = [name](cl_device_id device) {
            return is_extension_available(device, name.c_str());
        };
    }
};

// Feature declarations
#define FEATURE(X) extern const feature& X
FEATURE(F_true);
FEATURE(F_subgroups_extension);
FEATURE(F_subgroups);
FEATURE(F_program_scope_variables);
FEATURE(F_non_uniform_work_groups);
FEATURE(F_read_write_images);
#undef FEATURE

#endif // #ifndef HARNESS_FEATURE_H_
