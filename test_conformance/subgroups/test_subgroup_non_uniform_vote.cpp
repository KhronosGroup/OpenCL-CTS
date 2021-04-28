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
    static void gen(T *x, T *t, cl_int *m, const WorkGroupParams &test_params)
    {
        int i, ii, j, k, n;
        int nw = test_params.local_workgroup_size;
        int ns = test_params.subgroup_size;
        int ng = test_params.global_workgroup_size;
        uint32_t work_items_mask = test_params.work_items_mask;
        int nj = (nw + ns - 1) / ns;
        int non_uniform_size = ng % nw;
        ng = ng / nw;
        int last_subgroup_size = 0;
        ii = 0;

        log_info("  sub_group_%s%s... \n",
                 (operation == NonUniformVoteOp::elect) ? "" : "non_uniform_",
                 operation_names(operation));

        log_info("  test params: global size = %d local size = %d subgroups "
                 "size = %d work item mask = 0x%x data type (%s)\n",
                 test_params.global_workgroup_size, nw, ns, work_items_mask,
                 TypeManager<T>::name());
        if (non_uniform_size)
        {
            log_info("  non uniform work group size mode ON\n");
        }
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

    static int chk(T *x, T *y, T *mx, T *my, cl_int *m,
                   const WorkGroupParams &test_params)
    {
        int ii, i, j, k, n;
        int nw = test_params.local_workgroup_size;
        int ns = test_params.subgroup_size;
        int ng = test_params.global_workgroup_size;
        uint32_t work_items_mask = test_params.work_items_mask;
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
                    uint32_t check_work_item = 1 << (i % 32);
                    if (work_items_mask & check_work_item)
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
                    log_info("  no one workitem acitve... in workgroup id = %d "
                             "subgroup id = %d\n",
                             k, j);
                }
                else
                {
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
                            log_error(
                                "mismatch for work item %d sub group %d in "
                                "work group %d. Expected: %d Obtained: %d\n",
                                i, j, k, tr, rr);
                            return TEST_FAIL;
                        }
                    }
                }
            }

            x += nw;
            y += nw;
            m += 4 * nw;
        }

        log_info("  sub_group_%s%s... passed\n",
                 (operation == NonUniformVoteOp::elect) ? "" : "non_uniform_",
                 operation_names(operation));
        return TEST_PASS;
    }
};
static const char *elect_source = R"(
    __kernel void test_elect(const __global Type *in, __global int4 *xy, __global Type *out) {
        int gid = get_global_id(0);
        XY(xy,gid);
        uint elect_work_item = 1 << (get_sub_group_local_id() % 32);
            if (elect_work_item & WORK_ITEMS_MASK){
                out[gid] = sub_group_elect();
            }
    }
)";

static const char *non_uniform_any_source = R"(
    __kernel void test_non_uniform_any(const __global Type *in, __global int4 *xy, __global Type *out) {
        int gid = get_global_id(0);
        XY(xy,gid);
        uint elect_work_item = 1 << (get_sub_group_local_id() % 32);
            if (elect_work_item & WORK_ITEMS_MASK){
                out[gid] = sub_group_non_uniform_any(in[gid]);
            }
    }
)";

static const char *non_uniform_all_source = R"(
    __kernel void test_non_uniform_all(const __global Type *in, __global int4 *xy, __global Type *out) {
        int gid = get_global_id(0);
        XY(xy,gid);
        uint elect_work_item = 1 << (get_sub_group_local_id() % 32);
            if (elect_work_item & WORK_ITEMS_MASK){
                out[gid] = sub_group_non_uniform_all(in[gid]);
            }
    }
)";

static const char *non_uniform_all_equal_source = R"(
    __kernel void test_non_uniform_all_equal(const __global Type *in, __global int4 *xy, __global Type *out) {
        int gid = get_global_id(0);
        XY(xy,gid);
        uint elect_work_item = 1 << (get_sub_group_local_id() % 32);
            if (elect_work_item & WORK_ITEMS_MASK){
                out[gid] = sub_group_non_uniform_all_equal(in[gid]);
            }
    }
)";

template <typename T> int run_vote_all_equal_for_type(RunTestForType rft)
{
    int error = rft.run_impl<T, VOTE<T, NonUniformVoteOp::all_equal>>(
        "test_non_uniform_all_equal", non_uniform_all_equal_source);
    return error;
}
}

int test_subgroup_functions_non_uniform_vote(cl_device_id device,
                                             cl_context context,
                                             cl_command_queue queue,
                                             int num_elements)
{
    std::vector<std::string> required_extensions = {
        "cl_khr_subgroup_non_uniform_vote"
    };

    std::vector<uint32_t> masks{ 0xffffffff, 0x55aaaa55, 0x5555aaaa, 0xaaaa5555,
                                 0x0f0ff0f0, 0x0f0f0f0f, 0xff0000ff, 0xff00ff00,
                                 0x00ffff00, 0x80000000 };
    constexpr size_t global_work_size = 170;
    constexpr size_t local_work_size = 64;
    WorkGroupParams test_params(global_work_size, local_work_size,
                                required_extensions, masks);
    RunTestForType rft(device, context, queue, num_elements, test_params);

    int error = run_vote_all_equal_for_type<cl_int>(rft);
    error |= run_vote_all_equal_for_type<cl_uint>(rft);
    error |= run_vote_all_equal_for_type<cl_long>(rft);
    error |= run_vote_all_equal_for_type<cl_ulong>(rft);
    error |= run_vote_all_equal_for_type<cl_float>(rft);
    error |= run_vote_all_equal_for_type<cl_double>(rft);
    error |= run_vote_all_equal_for_type<subgroups::cl_half>(rft);

    error |= rft.run_impl<cl_int, VOTE<cl_int, NonUniformVoteOp::all>>(
        "test_non_uniform_all", non_uniform_all_source);
    error |= rft.run_impl<cl_int, VOTE<cl_int, NonUniformVoteOp::elect>>(
        "test_elect", elect_source);
    error |= rft.run_impl<cl_int, VOTE<cl_int, NonUniformVoteOp::any>>(
        "test_non_uniform_any", non_uniform_any_source);
    return error;
}
