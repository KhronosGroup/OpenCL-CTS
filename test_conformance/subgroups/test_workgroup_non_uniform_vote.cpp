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

// Global/local work group sizes
// Adjust these individually below if desired/needed
#define GWS_NON_UNIFORM 170
#define LWS_NON_UNIFORM 64

namespace {

template <typename T, NonUniformVoteOp operation, unsigned int work_items_mask>
struct VOTE
{
    static void gen(T *x, T *t, cl_int *m, int ns, int nw, int ng)
    {
        int i, ii, j, k, n;
        int nj = (nw + ns - 1) / ns;
        int non_uniform_size = ng % nw;
        ng = ng / nw;
        int last_subgroup_size = 0;
        ii = 0;

        log_info("  sub_group_%s... \n", operation_names(operation));
        log_info("  test params:\n  subgroups size = %d work item mask = 0x%x "
                 "data type (%s)\n",
                 ns, work_items_mask, TypeManager<T>::name());
        if (non_uniform_size)
        {
            log_info("  non uniform work group size mode ON\n");
        }

        for (k = 0; k < ng; ++k)
        { // for each work_group
            if (non_uniform_size && k == ng - 1)
            {
                set_last_worgroup_params(non_uniform_size, nj, ns, nw,
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

                // Initialize data matrix indexed by local id and sub group id
                switch (e)
                {
                    case 0: memset(&t[ii], 0, n * sizeof(T)); break;
                    case 1:
                        memset(&t[ii], 0, n * sizeof(T));
                        i = (int)(genrand_int32(gMTdata) % (cl_uint)n);
                        set_value(t[ii + i], 41);
                        break;
                    case 2: memset(&t[ii], 0xff, n * sizeof(T)); break;
                }
            }
            // Now map into work group using map from device
            for (j = 0; j < nw; ++j)
            {
                i = m[4 * j + 1] * ns + m[4 * j];
                x[j] = t[i];
            }
            x += nw;
            m += 4 * nw;
        }
    }

    static int chk(T *x, T *y, T *mx, T *my, cl_int *m, int ns, int nw, int ng)
    {
        int ii, i, j, k, n;
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
                set_last_worgroup_params(non_uniform_size, nj, ns, nw,
                                         last_subgroup_size);
            }
            for (j = 0; j < nw; ++j)
            { // inside the work_group
                i = m[4 * j + 1] * ns + m[4 * j];
                mx[i] = x[j]; // read host inputs for work_group
                my[i] = y[j]; // read device outputs for work_group
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
                    int check_work_item = 1 << i % 32;
                    if (work_items_mask & check_work_item)
                    {
                        active_work_items.insert(i);

                        if (operation == NonUniformVoteOp::all)
                        {
                            tr &= !compare_ordered<T>(mx[ii + i], 0) ? 1 : 0;
                        }

                        if (operation == NonUniformVoteOp::any)
                        {
                            tr |= !compare_ordered<T>(mx[ii + i], 0) ? 1 : 0;
                        }

                        if (operation == NonUniformVoteOp::all_equal)
                        {
                            tr &= compare_ordered<T>(
                                      mx[ii + i],
                                      mx[ii + *active_work_items.begin()])
                                ? 1
                                : 0;
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
        log_info("  sub_group_%s... passed\n", operation_names(operation));
        return TEST_PASS;
    }
};
static const char *elect_source = R"(
    __kernel void test_elect(const __global Type *in, __global int4 *xy, __global Type *out) {
        int gid = get_global_id(0);
        XY(xy,gid);
        int elect_work_item = 1 << (get_sub_group_local_id() % 32);
            if (elect_work_item & WORK_ITEMS_MASK){
                out[gid] = sub_group_elect();
            }
    }
)";

static const char *non_uniform_any_source = R"(
    __kernel void test_non_uniform_any(const __global Type *in, __global int4 *xy, __global Type *out) {
        int gid = get_global_id(0);
        XY(xy,gid);
        int elect_work_item = 1 << (get_sub_group_local_id() % 32);
            if (elect_work_item & WORK_ITEMS_MASK){
                out[gid] = sub_group_non_uniform_any(in[gid]);
            }
    }
)";

static const char *non_uniform_all_source = R"(
    __kernel void test_non_uniform_all(const __global Type *in, __global int4 *xy, __global Type *out) {
        int gid = get_global_id(0);
        XY(xy,gid);
        int elect_work_item = 1 << (get_sub_group_local_id() % 32);
            if (elect_work_item & WORK_ITEMS_MASK){
                out[gid] = sub_group_non_uniform_all(in[gid]);
            }
    }
)";

static const char *non_uniform_all_equal_source = R"(
    __kernel void test_non_uniform_all_equal(const __global Type *in, __global int4 *xy, __global Type *out) {
        int gid = get_global_id(0);
        XY(xy,gid);
        int elect_work_item = 1 << (get_sub_group_local_id() % 32);
            if (elect_work_item & WORK_ITEMS_MASK){
                out[gid] = sub_group_non_uniform_all_equal(in[gid]);
            }
    }
)";

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

    template <typename T, unsigned int work_items_mask> int run_nu_all_eq()
    {
        int error =
            test<T, VOTE<T, NonUniformVoteOp::all_equal, work_items_mask>,
                 GWS_NON_UNIFORM,
                 LWS_NON_UNIFORM>::run(device_, context_, queue_, num_elements_,
                                       "test_non_uniform_all_equal",
                                       non_uniform_all_equal_source, 0,
                                       useCoreSubgroups_, required_extensions_,
                                       work_items_mask);
        return error;
    }

    template <typename T, unsigned int work_items_mask> int run_elect()
    {
        int error =
            test<T, VOTE<T, NonUniformVoteOp::elect, work_items_mask>,
                 GWS_NON_UNIFORM, LWS_NON_UNIFORM>::run(device_, context_,
                                                        queue_, num_elements_,
                                                        "test_elect",
                                                        elect_source, 0,
                                                        useCoreSubgroups_,
                                                        required_extensions_,
                                                        work_items_mask);

        return error;
    }

    template <typename T, unsigned int work_items_mask> int run_nu_any()
    {
        int error =
            test<T, VOTE<T, NonUniformVoteOp::any, work_items_mask>,
                 GWS_NON_UNIFORM, LWS_NON_UNIFORM>::run(device_, context_,
                                                        queue_, num_elements_,
                                                        "test_non_uniform_any",
                                                        non_uniform_any_source,
                                                        0, useCoreSubgroups_,
                                                        required_extensions_,
                                                        work_items_mask);
        return error;
    }

    template <typename T, unsigned int work_items_mask> int run_nu_all()
    {
        int error =
            test<T, VOTE<T, NonUniformVoteOp::all, work_items_mask>,
                 GWS_NON_UNIFORM, LWS_NON_UNIFORM>::run(device_, context_,
                                                        queue_, num_elements_,
                                                        "test_non_uniform_all",
                                                        non_uniform_all_source,
                                                        0, useCoreSubgroups_,
                                                        required_extensions_,
                                                        static_cast<int>(
                                                            work_items_mask));
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

int test_work_group_functions_non_uniform_vote(cl_device_id device,
                                               cl_context context,
                                               cl_command_queue queue,
                                               int num_elements)
{
    std::vector<std::string> required_extensions = {
        "cl_khr_subgroup_non_uniform_vote"
    };
    run_for_type rft(device, context, queue, num_elements, true,
                     required_extensions);

    int error = rft.run_nu_all_eq<cl_char, 0xffffaaaa>();
    error |= rft.run_nu_all_eq<cl_uchar, 0xffffaaaa>();
    error |= rft.run_nu_all_eq<cl_short, 0xffffaaaa>();
    error |= rft.run_nu_all_eq<cl_ushort, 0xffffaaaa>();
    error |= rft.run_nu_all_eq<cl_int, 0xffffaaaa>();
    error |= rft.run_nu_all_eq<cl_uint, 0xffffaaaa>();
    error |= rft.run_nu_all_eq<cl_long, 0xffffaaaa>();
    error |= rft.run_nu_all_eq<cl_ulong, 0xffffaaaa>();
    error |= rft.run_nu_all_eq<cl_float, 0xffffaaaa>();
    error |= rft.run_nu_all_eq<cl_double, 0xffffaaaa>();
    error |= rft.run_nu_all_eq<subgroups::cl_half, 0xffffaaaa>();

    error |= rft.run_nu_all_eq<cl_char, 0x80000000>();
    error |= rft.run_nu_all_eq<cl_uchar, 0x80000000>();
    error |= rft.run_nu_all_eq<cl_short, 0x80000000>();
    error |= rft.run_nu_all_eq<cl_ushort, 0x80000000>();
    error |= rft.run_nu_all_eq<cl_int, 0x80000000>();
    error |= rft.run_nu_all_eq<cl_uint, 0x80000000>();
    error |= rft.run_nu_all_eq<cl_long, 0x80000000>();
    error |= rft.run_nu_all_eq<cl_ulong, 0x80000000>();
    error |= rft.run_nu_all_eq<cl_float, 0x80000000>();
    error |= rft.run_nu_all_eq<cl_double, 0x80000000>();
    error |= rft.run_nu_all_eq<subgroups::cl_half, 0x80000000>();

    error |= rft.run_nu_all_eq<cl_char, 0x00ffff00>();
    error |= rft.run_nu_all_eq<cl_uchar, 0x00ffff00>();
    error |= rft.run_nu_all_eq<cl_short, 0x00ffff00>();
    error |= rft.run_nu_all_eq<cl_ushort, 0x00ffff00>();
    error |= rft.run_nu_all_eq<cl_int, 0x00ffff00>();
    error |= rft.run_nu_all_eq<cl_uint, 0x00ffff00>();
    error |= rft.run_nu_all_eq<cl_long, 0x00ffff00>();
    error |= rft.run_nu_all_eq<cl_ulong, 0x00ffff00>();
    error |= rft.run_nu_all_eq<cl_float, 0x00ffff00>();
    error |= rft.run_nu_all_eq<cl_double, 0x00ffff00>();
    error |= rft.run_nu_all_eq<subgroups::cl_half, 0x00ffff00>();

    error |= rft.run_nu_all_eq<cl_char, 0xff00ff00>();
    error |= rft.run_nu_all_eq<cl_uchar, 0xff00ff00>();
    error |= rft.run_nu_all_eq<cl_short, 0xff00ff00>();
    error |= rft.run_nu_all_eq<cl_ushort, 0xff00ff00>();
    error |= rft.run_nu_all_eq<cl_int, 0xff00ff00>();
    error |= rft.run_nu_all_eq<cl_uint, 0xff00ff00>();
    error |= rft.run_nu_all_eq<cl_long, 0xff00ff00>();
    error |= rft.run_nu_all_eq<cl_ulong, 0xff00ff00>();
    error |= rft.run_nu_all_eq<cl_float, 0xff00ff00>();
    error |= rft.run_nu_all_eq<cl_double, 0xff00ff00>();
    error |= rft.run_nu_all_eq<subgroups::cl_half, 0xff00ff00>();

    error |= rft.run_nu_all_eq<cl_char, 0xff0000ff>();
    error |= rft.run_nu_all_eq<cl_uchar, 0xff0000ff>();
    error |= rft.run_nu_all_eq<cl_short, 0xff0000ff>();
    error |= rft.run_nu_all_eq<cl_ushort, 0xff0000ff>();
    error |= rft.run_nu_all_eq<cl_int, 0xff0000ff>();
    error |= rft.run_nu_all_eq<cl_uint, 0xff0000ff>();
    error |= rft.run_nu_all_eq<cl_long, 0xff0000ff>();
    error |= rft.run_nu_all_eq<cl_ulong, 0xff0000ff>();
    error |= rft.run_nu_all_eq<cl_float, 0xff0000ff>();
    error |= rft.run_nu_all_eq<cl_double, 0xff0000ff>();
    error |= rft.run_nu_all_eq<subgroups::cl_half, 0xff0000ff>();

    error |= rft.run_nu_all_eq<cl_char, 0x0f0f0f0f>();
    error |= rft.run_nu_all_eq<cl_uchar, 0x0f0f0f0f>();
    error |= rft.run_nu_all_eq<cl_short, 0x0f0f0f0f>();
    error |= rft.run_nu_all_eq<cl_ushort, 0x0f0f0f0f>();
    error |= rft.run_nu_all_eq<cl_int, 0x0f0f0f0f>();
    error |= rft.run_nu_all_eq<cl_uint, 0x0f0f0f0f>();
    error |= rft.run_nu_all_eq<cl_long, 0x0f0f0f0f>();
    error |= rft.run_nu_all_eq<cl_ulong, 0x0f0f0f0f>();
    error |= rft.run_nu_all_eq<cl_float, 0x0f0f0f0f>();
    error |= rft.run_nu_all_eq<cl_double, 0x0f0f0f0f>();
    error |= rft.run_nu_all_eq<subgroups::cl_half, 0x0f0f0f0f>();

    error |= rft.run_nu_all_eq<cl_char, 0x0f0ff0f0>();
    error |= rft.run_nu_all_eq<cl_uchar, 0x0f0ff0f0>();
    error |= rft.run_nu_all_eq<cl_short, 0x0f0ff0f0>();
    error |= rft.run_nu_all_eq<cl_ushort, 0x0f0ff0f0>();
    error |= rft.run_nu_all_eq<cl_int, 0x0f0ff0f0>();
    error |= rft.run_nu_all_eq<cl_uint, 0x0f0ff0f0>();
    error |= rft.run_nu_all_eq<cl_long, 0x0f0ff0f0>();
    error |= rft.run_nu_all_eq<cl_ulong, 0x0f0ff0f0>();
    error |= rft.run_nu_all_eq<cl_float, 0x0f0ff0f0>();
    error |= rft.run_nu_all_eq<cl_double, 0x0f0ff0f0>();
    error |= rft.run_nu_all_eq<subgroups::cl_half, 0x0f0ff0f0>();

    error |= rft.run_nu_all_eq<cl_char, 0xaaaa5555>();
    error |= rft.run_nu_all_eq<cl_uchar, 0xaaaa5555>();
    error |= rft.run_nu_all_eq<cl_short, 0xaaaa5555>();
    error |= rft.run_nu_all_eq<cl_ushort, 0xaaaa5555>();
    error |= rft.run_nu_all_eq<cl_int, 0xaaaa5555>();
    error |= rft.run_nu_all_eq<cl_uint, 0xaaaa5555>();
    error |= rft.run_nu_all_eq<cl_long, 0xaaaa5555>();
    error |= rft.run_nu_all_eq<cl_ulong, 0xaaaa5555>();
    error |= rft.run_nu_all_eq<cl_float, 0xaaaa5555>();
    error |= rft.run_nu_all_eq<cl_double, 0xaaaa5555>();
    error |= rft.run_nu_all_eq<subgroups::cl_half, 0xaaaa5555>();

    error |= rft.run_nu_all_eq<cl_char, 0x5555aaaa>();
    error |= rft.run_nu_all_eq<cl_uchar, 0x5555aaaa>();
    error |= rft.run_nu_all_eq<cl_short, 0x5555aaaa>();
    error |= rft.run_nu_all_eq<cl_ushort, 0x5555aaaa>();
    error |= rft.run_nu_all_eq<cl_int, 0x5555aaaa>();
    error |= rft.run_nu_all_eq<cl_uint, 0x5555aaaa>();
    error |= rft.run_nu_all_eq<cl_long, 0x5555aaaa>();
    error |= rft.run_nu_all_eq<cl_ulong, 0x5555aaaa>();
    error |= rft.run_nu_all_eq<cl_float, 0x5555aaaa>();
    error |= rft.run_nu_all_eq<cl_double, 0x5555aaaa>();
    error |= rft.run_nu_all_eq<subgroups::cl_half, 0x5555aaaa>();

    error |= rft.run_nu_all_eq<cl_char, 0x55aaaa55>();
    error |= rft.run_nu_all_eq<cl_uchar, 0x55aaaa55>();
    error |= rft.run_nu_all_eq<cl_short, 0x55aaaa55>();
    error |= rft.run_nu_all_eq<cl_ushort, 0x55aaaa55>();
    error |= rft.run_nu_all_eq<cl_int, 0x55aaaa55>();
    error |= rft.run_nu_all_eq<cl_uint, 0x55aaaa55>();
    error |= rft.run_nu_all_eq<cl_long, 0x55aaaa55>();
    error |= rft.run_nu_all_eq<cl_ulong, 0x55aaaa55>();
    error |= rft.run_nu_all_eq<cl_float, 0x55aaaa55>();
    error |= rft.run_nu_all_eq<cl_double, 0x55aaaa55>();
    error |= rft.run_nu_all_eq<subgroups::cl_half, 0x55aaaa55>();

    error |= rft.run_nu_all_eq<cl_char, 0xffffffff>();
    error |= rft.run_nu_all_eq<cl_uchar, 0xffffffff>();
    error |= rft.run_nu_all_eq<cl_short, 0xffffffff>();
    error |= rft.run_nu_all_eq<cl_ushort, 0xffffffff>();
    error |= rft.run_nu_all_eq<cl_int, 0xffffffff>();
    error |= rft.run_nu_all_eq<cl_uint, 0xffffffff>();
    error |= rft.run_nu_all_eq<cl_long, 0xffffffff>();
    error |= rft.run_nu_all_eq<cl_ulong, 0xffffffff>();
    error |= rft.run_nu_all_eq<cl_float, 0xffffffff>();
    error |= rft.run_nu_all_eq<cl_double, 0xffffffff>();
    error |= rft.run_nu_all_eq<subgroups::cl_half, 0xffffffff>();

    error |= rft.run_elect<int, 0xffffffff>();
    error |= rft.run_elect<int, 0xffffaaaa>();
    error |= rft.run_elect<int, 0x80000000>();
    error |= rft.run_elect<int, 0x00ffff00>();
    error |= rft.run_elect<int, 0xff00ff00>();
    error |= rft.run_elect<int, 0xff0000ff>();
    error |= rft.run_elect<int, 0x0f0f0f0f>();
    error |= rft.run_elect<int, 0x0f0ff0f0>();
    error |= rft.run_elect<int, 0xaaaa5555>();
    error |= rft.run_elect<int, 0x5555aaaa>();
    error |= rft.run_elect<int, 0x55aaaa55>();

    error |= rft.run_nu_any<int, 0xffffffff>();
    error |= rft.run_nu_any<int, 0xffffaaaa>();
    error |= rft.run_nu_any<int, 0x80000000>();
    error |= rft.run_nu_any<int, 0x00ffff00>();
    error |= rft.run_nu_any<int, 0xff00ff00>();
    error |= rft.run_nu_any<int, 0xff0000ff>();
    error |= rft.run_nu_any<int, 0x0f0f0f0f>();
    error |= rft.run_nu_any<int, 0x0f0ff0f0>();
    error |= rft.run_nu_any<int, 0xaaaa5555>();
    error |= rft.run_nu_any<int, 0x5555aaaa>();
    error |= rft.run_nu_any<int, 0x55aaaa55>();

    error |= rft.run_nu_all<int, 0xffffffff>();
    error |= rft.run_nu_all<int, 0xffffaaaa>();
    error |= rft.run_nu_all<int, 0x80000000>();
    error |= rft.run_nu_all<int, 0x00ffff00>();
    error |= rft.run_nu_all<int, 0xff00ff00>();
    error |= rft.run_nu_all<int, 0xff0000ff>();
    error |= rft.run_nu_all<int, 0x0f0f0f0f>();
    error |= rft.run_nu_all<int, 0x0f0ff0f0>();
    error |= rft.run_nu_all<int, 0xaaaa5555>();
    error |= rft.run_nu_all<int, 0x5555aaaa>();
    error |= rft.run_nu_all<int, 0x55aaaa55>();

    return error;
}