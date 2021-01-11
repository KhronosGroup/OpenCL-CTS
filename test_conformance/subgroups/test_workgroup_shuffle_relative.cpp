//
// Copyright (c) 2021 The Khronos Group Inc.
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
#include "procs.h"
#include "subhelpers.h"
#include "workgroup_common_templates.h"
#include "harness/conversions.h"
#include "harness/typeWrappers.h"

// Global/local work group sizes
// Adjust these individually below if desired/needed
#define GWS 2000
#define LWS 200

namespace {

static const char* shuffle_down_source =
    "__kernel void test_sub_group_shuffle_down(const __global Type *in, "
    "__global int4 *xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    "    Type x = in[gid];\n"
    "    out[gid] = sub_group_shuffle_down(x, xy[gid].z);"
    "}\n";
static const char* shuffle_up_source =
    "__kernel void test_sub_group_shuffle_up(const __global Type *in, __global "
    "int4 *xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    "    Type x = in[gid];\n"
    "    out[gid] = sub_group_shuffle_up(x, xy[gid].z);"
    "}\n";

struct run_for_type
{
    run_for_type(cl_device_id device, cl_context context,
                 cl_command_queue queue, int num_elements,
                 bool useCoreSubgroups,
                 std::vector<std::string> required_extensions = {})
        : device_(device), context_(context), queue_(queue),
          num_elements_(num_elements), useCoreSubgroups_(useCoreSubgroups),
          required_extensions_(required_extensions)
    {}

    template <typename T> int run_shuffle_relative()
    {
        int error = test<T, SHF<T, ShuffleOp::shuffle_up>, GWS, LWS>::run(
            device_, context_, queue_, num_elements_,
            "test_sub_group_shuffle_up", shuffle_up_source, 0,
            useCoreSubgroups_, required_extensions_);
        error |= test<T, SHF<T, ShuffleOp::shuffle_down>, GWS, LWS>::run(
            device_, context_, queue_, num_elements_,
            "test_sub_group_shuffle_down", shuffle_down_source, 0,
            useCoreSubgroups_, required_extensions_);
        return error;
    }

private:
    cl_device_id device_;
    cl_context context_;
    cl_command_queue queue_;
    int num_elements_;
    bool useCoreSubgroups_;
    std::vector<std::string> required_extensions_;
};

}

int test_work_group_functions_shuffle_relative(cl_device_id device,
                                               cl_context context,
                                               cl_command_queue queue,
                                               int num_elements)
{
    std::vector<std::string> required_extensions = {
        "cl_khr_subgroup_shuffle_relative"
    };
    run_for_type rft(device, context, queue, num_elements, true,
                     required_extensions);

    int error = rft.run_shuffle_relative<cl_int>();
    error |= rft.run_shuffle_relative<cl_uint>();
    error |= rft.run_shuffle_relative<cl_long>();
    error |= rft.run_shuffle_relative<cl_ulong>();
    error |= rft.run_shuffle_relative<cl_short>();
    error |= rft.run_shuffle_relative<cl_ushort>();
    error |= rft.run_shuffle_relative<cl_char>();
    error |= rft.run_shuffle_relative<cl_uchar>();
    error |= rft.run_shuffle_relative<cl_float>();
    error |= rft.run_shuffle_relative<cl_double>();
    error |= rft.run_shuffle_relative<subgroups::cl_half>();

    return error;
}
