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
#include "harness/typeWrappers.h"
#include <set>

namespace {

template <typename T, NonUniformVoteOp operation> struct VOTE
{
    static void log_test(const WorkGroupParams &test_params,
                         const char *extra_text)
    {
        log_info("  sub_group_%s%s(%s)...%s\n",
                 (operation == NonUniformVoteOp::elect) ? "" : "non_uniform_",
                 operation_names(operation), TypeManager<T>::name(),
                 extra_text);
    }

    static void gen(T *x, T *t, cl_int *m, const WorkGroupParams &test_params)
    {
        int i, ii, j, k, n;
        int nw = test_params.local_workgroup_size;
        int ns = test_params.subgroup_size;
        int ng = test_params.global_workgroup_size;
        int nj = (nw + ns - 1) / ns;
        int non_uniform_size = ng % nw;
        ng = ng / nw;
        int last_subgroup_size = 0;
        ii = 0;

        if (operation == NonUniformVoteOp::elect) return;

        for (k = 0; k < ng; ++k)
        { // for each work_group
            if (non_uniform_size && k == ng - 1)
            {
                set_last_workgroup_params(non_uniform_size, nj, ns, nw,
                                          last_subgroup_size);
            }
            for (j = 0; j < nj; ++j)
            { // for each subgroup
                ii = j * ns;
                if (last_subgroup_size && j == nj - 1)
                {
                    n = last_subgroup_size;
                }
                else
                {
                    n = ii + ns > nw ? nw - ii : ns;
                }
                int e = genrand_int32(gMTdata) % 3;

                for (i = 0; i < n; i++)
                {
                    if (e == 2)
                    { // set once 0 and once 1 alternately
                        int value = i % 2;
                        set_value(t[ii + i], value);
                    }
                    else
                    { // set 0/1 for all work items in subgroup
                        set_value(t[ii + i], e);
                    }
                }
            }
            // Now map into work group using map from device
            for (j = 0; j < nw; ++j)
            {
                x[j] = t[j];
            }
            x += nw;
            m += 4 * nw;
        }
    }

    static test_status chk(T *x, T *y, T *mx, T *my, cl_int *m,
                           const WorkGroupParams &test_params)
    {
        int ii, i, j, k, n;
        int nw = test_params.local_workgroup_size;
        int ns = test_params.subgroup_size;
        int ng = test_params.global_workgroup_size;
        int nj = (nw + ns - 1) / ns;
        cl_int tr, rr;
        int non_uniform_size = ng % nw;
        ng = ng / nw;
        if (non_uniform_size) ng++;
        int last_subgroup_size = 0;

        for (k = 0; k < ng; ++k)
        { // for each work_group
            if (non_uniform_size && k == ng - 1)
            {
                set_last_workgroup_params(non_uniform_size, nj, ns, nw,
                                          last_subgroup_size);
            }
            for (j = 0; j < nw; ++j)
            { // inside the work_group
                mx[j] = x[j]; // read host inputs for work_group
                my[j] = y[j]; // read device outputs for work_group
            }

            for (j = 0; j < nj; ++j)
            { // for each subgroup
                ii = j * ns;
                if (last_subgroup_size && j == nj - 1)
                {
                    n = last_subgroup_size;
                }
                else
                {
                    n = ii + ns > nw ? nw - ii : ns;
                }

                rr = 0;
                if (operation == NonUniformVoteOp::all
                    || operation == NonUniformVoteOp::all_equal)
                    tr = 1;
                if (operation == NonUniformVoteOp::any) tr = 0;

                std::set<int> active_work_items;
                for (i = 0; i < n; ++i)
                {
                    if (test_params.work_items_mask.test(i))
                    {
                        active_work_items.insert(i);
                        switch (operation)
                        {
                            case NonUniformVoteOp::elect: break;

                            case NonUniformVoteOp::all:
                                tr &=
                                    !compare_ordered<T>(mx[ii + i], 0) ? 1 : 0;
                                break;
                            case NonUniformVoteOp::any:
                                tr |=
                                    !compare_ordered<T>(mx[ii + i], 0) ? 1 : 0;
                                break;
                            case NonUniformVoteOp::all_equal:
                                tr &= compare_ordered<T>(
                                          mx[ii + i],
                                          mx[ii + *active_work_items.begin()])
                                    ? 1
                                    : 0;
                                break;
                            default:
                                log_error("Unknown operation\n");
                                return TEST_FAIL;
                        }
                    }
                }
                if (active_work_items.empty())
                {
                    continue;
                }
                auto lowest_active = active_work_items.begin();
                for (const int &active_work_item : active_work_items)
                {
                    i = active_work_item;
                    if (operation == NonUniformVoteOp::elect)
                    {
                        i == *lowest_active ? tr = 1 : tr = 0;
                    }

                    // normalize device values on host, non zero set 1.
                    rr = compare_ordered<T>(my[ii + i], 0) ? 0 : 1;

                    if (rr != tr)
                    {
                        log_error("ERROR: sub_group_%s() \n",
                                  operation_names(operation));
                        log_error("mismatch for work item %d sub group %d in "
                                  "work group %d. Expected: %d Obtained: %d\n",
                                  i, j, k, tr, rr);
                        return TEST_FAIL;
                    }
                }
            }

            x += nw;
            y += nw;
            m += 4 * nw;
        }

        return TEST_PASS;
    }
};

std::string sub_group_elect_source = R"(
    __kernel void test_sub_group_elect(const __global Type *in, __global int4 *xy, __global Type *out) {
        int gid = get_global_id(0);
        XY(xy,gid);
        uint subgroup_local_id = get_sub_group_local_id();
        uint elect_work_item = 1 << (subgroup_local_id % 32);
        uint work_item_mask;
        if(subgroup_local_id < 32) {
            work_item_mask = work_item_mask_vector.x;
        } else if(subgroup_local_id < 64) {
            work_item_mask = work_item_mask_vector.y;
        } else if(subgroup_local_id < 96) {
            work_item_mask = work_item_mask_vector.w;
        } else if(subgroup_local_id < 128) {
            work_item_mask = work_item_mask_vector.z;
        }
        if (elect_work_item & work_item_mask){
            out[gid] = sub_group_elect();
        }
    }
)";

std::string sub_group_non_uniform_any_all_all_equal_source = R"(
    __kernel void test_%s(const __global Type *in, __global int4 *xy, __global Type *out) {
        int gid = get_global_id(0);
        XY(xy,gid);
        uint subgroup_local_id = get_sub_group_local_id();
        uint elect_work_item = 1 << (subgroup_local_id % 32);
        uint work_item_mask;
        if(subgroup_local_id < 32) {
            work_item_mask = work_item_mask_vector.x;
        } else if(subgroup_local_id < 64) {
            work_item_mask = work_item_mask_vector.y;
        } else if(subgroup_local_id < 96) {
            work_item_mask = work_item_mask_vector.w;
        } else if(subgroup_local_id < 128) {
            work_item_mask = work_item_mask_vector.z;
        }
        if (elect_work_item & work_item_mask){
                out[gid] = %s(in[gid]);
            }
    }
)";

template <typename T> int run_vote_all_equal_for_type(RunTestForType rft)
{
    int error = rft.run_impl<T, VOTE<T, NonUniformVoteOp::all_equal>>(
        "sub_group_non_uniform_all_equal");
    return error;
}
}

int test_subgroup_functions_non_uniform_vote(cl_device_id device,
                                             cl_context context,
                                             cl_command_queue queue,
                                             int num_elements)
{
    if (!is_extension_available(device, "cl_khr_subgroup_non_uniform_vote"))
    {
        log_info("cl_khr_subgroup_non_uniform_vote is not supported on this "
                 "device, skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    constexpr size_t global_work_size = 170;
    constexpr size_t local_work_size = 64;
    WorkGroupParams test_params(global_work_size, local_work_size, true);
    test_params.save_kernel_source(
        sub_group_non_uniform_any_all_all_equal_source);
    test_params.save_kernel_source(sub_group_elect_source, "sub_group_elect");
    RunTestForType rft(device, context, queue, num_elements, test_params);

    int error = run_vote_all_equal_for_type<cl_int>(rft);
    error |= run_vote_all_equal_for_type<cl_uint>(rft);
    error |= run_vote_all_equal_for_type<cl_long>(rft);
    error |= run_vote_all_equal_for_type<cl_ulong>(rft);
    error |= run_vote_all_equal_for_type<cl_float>(rft);
    error |= run_vote_all_equal_for_type<cl_double>(rft);
    error |= run_vote_all_equal_for_type<subgroups::cl_half>(rft);

    error |= rft.run_impl<cl_int, VOTE<cl_int, NonUniformVoteOp::all>>(
        "sub_group_non_uniform_all");
    error |= rft.run_impl<cl_int, VOTE<cl_int, NonUniformVoteOp::elect>>(
        "sub_group_elect");
    error |= rft.run_impl<cl_int, VOTE<cl_int, NonUniformVoteOp::any>>(
        "sub_group_non_uniform_any");
    return error;
}
