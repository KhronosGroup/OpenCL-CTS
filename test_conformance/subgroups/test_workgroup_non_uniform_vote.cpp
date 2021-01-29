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

// Global/local work group sizes
// Adjust these individually below if desired/needed
#define GWS 2000
#define LWS 200

#define GWS_NON_UNIFORM 170
#define LWS_NON_UNIFORM 64
namespace {
template <typename T>
void calculate(const T *mx, NonUniformVoteOp operation, int subgroup_size,
               bool *compare_result)
{
    compare_result[0] = 1;
    compare_result[1] = 1;
    switch (operation)
    {
        case NonUniformVoteOp::all_equal:
            for (int i = 0; i < subgroup_size; ++i)
            {
                if (i < NR_OF_ACTIVE_WORK_ITEMS)
                { // return non zero if all the same
                    compare_result[0] &= compare_ordered(mx[i], mx[0]);
                }
                else
                {
                    compare_result[1] &=
                        compare_ordered(mx[i], mx[NR_OF_ACTIVE_WORK_ITEMS]);
                }
            }
            break;
        default: log_error("Unknown operation request"); break;
    }
}
template <>
void calculate(const int *mx, NonUniformVoteOp operation, int subgroup_size,
               bool *compare_result)
{
    compare_result[0] = false;
    compare_result[1] = false;
    switch (operation)
    {
        case NonUniformVoteOp::all_equal:
            compare_result[0] = 1;
            compare_result[1] = 1;
            for (int i = 0; i < subgroup_size; ++i)
            {
                if (i < NR_OF_ACTIVE_WORK_ITEMS)
                {
                    // return non zero if all the same
                    compare_result[0] &= compare_ordered(mx[i], mx[0]);
                }
                else
                {
                    compare_result[1] &=
                        compare_ordered(mx[i], mx[NR_OF_ACTIVE_WORK_ITEMS]);
                }
            }
            break;
        case NonUniformVoteOp::any:
            compare_result[0] = false;
            compare_result[1] = false;
            for (int i = 0; i < subgroup_size; ++i)
            {
                if (i < NR_OF_ACTIVE_WORK_ITEMS)
                {
                    // return non zero if value non zero at least for one
                    compare_result[0] |= mx[i] != 0;
                }
                else
                {

                    compare_result[1] |= mx[i] != 0;
                }
            }
            break;
        case NonUniformVoteOp::all:
            compare_result[0] = true;
            compare_result[1] = true;
            for (int i = 0; i < subgroup_size; ++i)
            {
                if (i < NR_OF_ACTIVE_WORK_ITEMS)
                {
                    // return non zero if value non zero for all
                    compare_result[0] &= mx[i] != 0;
                }
                else
                {
                    compare_result[1] &= mx[i] != 0;
                }
            }
            break;
        default: log_error("Unknown operation request"); break;
    }
}

// Test any/all/all_equal non uniform test functions.
template <typename Ty, NonUniformVoteOp operation> struct AAN
{
    static void gen(Ty *x, Ty *t, cl_int *m, int ns, int nw, int ng)
    {
        int i, ii, j, k, n;
        int nj = (nw + ns - 1) / ns;
        int non_uniform_size = ng % nw;
        ng = ng / nw;
        int last_subgroup_size = 0;
        ii = 0;
        log_info("  sub_group_non_uniform_%s(%s)...\n",
                 operation_names(operation), TypeManager<Ty>::name());
        if (non_uniform_size)
        {
            log_info("  non uniform work group size mode ON\n");
            ng++;
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
                    case 0: memset(&t[ii], 0, n * sizeof(Ty)); break;
                    case 1:
                        memset(&t[ii], 0, n * sizeof(Ty));
                        i = (int)(genrand_int32(gMTdata) % (cl_uint)n);
                        set_value(t[ii + i], 41);
                        break;
                    case 2: memset(&t[ii], 0xff, n * sizeof(Ty)); break;
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

    static int chk(Ty *x, Ty *y, Ty *mx, Ty *my, cl_int *m, int ns, int nw,
                   int ng)
    {
        int ii, i, j, k, n;
        int nj = (nw + ns - 1) / ns;
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
            // Map to array indexed to array indexed by local ID and sub group
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

                // Compute target
                bool calculation_result[2];
                calculate<Ty>(mx + ii, operation, n, calculation_result);

                // Check result
                static const Ty false_value{};
                for (i = 0; i < n; ++i)
                {
                    bool device_result = !compare(my[ii + i], false_value);
                    bool expected_result = i < NR_OF_ACTIVE_WORK_ITEMS
                        ? calculation_result[0]
                        : calculation_result[1];
                    if (device_result != expected_result)
                    {
                        log_error("ERROR: sub_group_non_uniform_%s mismatch "
                                  "for local id %d in sub group %d in group "
                                  "%d, obtained %d, expected %d\n",
                                  operation_names(operation), i, j, k,
                                  device_result, expected_result);
                        return TEST_FAIL;
                    }
                }
            }
            x += nw;
            y += nw;
            m += 4 * nw;
        }
        log_info("  sub_group_non_uniform_%s(%s)... passed\n",
                 operation_names(operation), TypeManager<Ty>::name());
        return TEST_PASS;
    }
};

// Test for elect function.
// Discover only one elected work item in subgroup - with the
// lowest subgroup local id
struct ELECT
{
    static void gen(cl_int *x, cl_int *t, cl_int *m, int ns, int nw, int ng)
    {
        int non_uniform_size = ng % nw;
        log_info("  sub_group_elect...\n");
        if (non_uniform_size)
        {
            log_info("  non uniform work group size mode ON\n");
        }
        // no work here needed.
    }

    static int chk(cl_int *x, cl_int *y, cl_int *mx, cl_int *my, cl_int *m,
                   int ns, int nw, int ng)
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
                for (i = 0; i < n; ++i)
                { // for each work_item in subgroup
                  // sum of output values should be 1

                    my[ii + i] > 0 ? rr += 1 : rr += 0;
                }
                // expectation is that only one elected returned true
                tr = n <= NR_OF_ACTIVE_WORK_ITEMS ? 1 : 2;
                if (rr != tr)
                {
                    log_error(
                        "ERROR: sub_group_elect() mismatch for sub group %d in "
                        "work group %d. Expected: %d Obtained: %d  \n",
                        j, k, tr, rr);
                    return TEST_FAIL;
                }
            }

            x += nw;
            y += nw;
            m += 4 * nw;
        }
        log_info("  sub_group_elect... passed\n");
        return TEST_PASS;
    }
};

static const char *elect_source =
    "__kernel void test_elect(const __global Type *in, __global int4 *xy, "
    "__global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    "    Type x = in[gid];\n"
    "    if (xy[gid].x < NR_OF_ACTIVE_WORK_ITEMS) {\n"
    // one in subgroup true others false.
    "        out[gid] = sub_group_elect();\n"
    "    } else {\n"
    "        out[gid] = sub_group_elect();\n"
    "    }\n"
    "}\n";

static const char *non_uniform_any_source =
    "__kernel void test_non_uniform_any(const __global Type *in, __global int4 "
    "*xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    "    if (xy[gid].x < NR_OF_ACTIVE_WORK_ITEMS) {\n"
    "        out[gid] = sub_group_non_uniform_any(in[gid]);\n"
    "    } else {\n"
    "        out[gid] = sub_group_non_uniform_any(in[gid]);\n"
    "    }\n"
    "}\n";
static const char *non_uniform_all_source =
    "__kernel void test_non_uniform_all(const __global Type *in, __global int4 "
    "*xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    "    if (xy[gid].x < NR_OF_ACTIVE_WORK_ITEMS) {\n"
    "        out[gid] = sub_group_non_uniform_all(in[gid]);\n"
    "    } else {\n"
    "        out[gid] = sub_group_non_uniform_all(in[gid]);\n"
    "    }\n"
    "}\n";
static const char *non_uniform_all_equal_source =
    "__kernel void test_non_uniform_all_equal(const __global Type *in, "
    "__global int4 *xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    "    if (xy[gid].x < NR_OF_ACTIVE_WORK_ITEMS) {\n"
    "        out[gid] = sub_group_non_uniform_all_equal(in[gid]);\n"
    "    } else {\n"
    "        out[gid] = sub_group_non_uniform_all_equal(in[gid]);\n"
    "    }\n"
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

    template <typename T> int run_nu_all_eq()
    {
        int error =
            test<T, AAN<T, NonUniformVoteOp::all_equal>, GWS_NON_UNIFORM,
                 LWS_NON_UNIFORM>::run(device_, context_, queue_, num_elements_,
                                       "test_non_uniform_all_equal",
                                       non_uniform_all_equal_source, 0,
                                       useCoreSubgroups_, required_extensions_);
        return error;
    }

    template <typename T> int run_elect()
    {
        int error = test<T, ELECT, GWS_NON_UNIFORM, LWS_NON_UNIFORM>::run(
            device_, context_, queue_, num_elements_, "test_elect",
            elect_source, 0, useCoreSubgroups_, required_extensions_);
        return error;
    }

    template <typename T> int run_nu_any()
    {
        int error =
            test<T, AAN<T, NonUniformVoteOp::any>, GWS_NON_UNIFORM,
                 LWS_NON_UNIFORM>::run(device_, context_, queue_, num_elements_,
                                       "test_non_uniform_any",
                                       non_uniform_any_source, 0,
                                       useCoreSubgroups_, required_extensions_);
        return error;
    }

    template <typename T> int run_nu_all()
    {
        int error =
            test<T, AAN<T, NonUniformVoteOp::all>, GWS_NON_UNIFORM,
                 LWS_NON_UNIFORM>::run(device_, context_, queue_, num_elements_,
                                       "test_non_uniform_all",
                                       non_uniform_all_source, 0,
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

    int error = rft.run_nu_all_eq<cl_char>();
    error |= rft.run_nu_all_eq<cl_uchar>();
    error |= rft.run_nu_all_eq<cl_short>();
    error |= rft.run_nu_all_eq<cl_ushort>();
    error |= rft.run_nu_all_eq<cl_int>();
    error |= rft.run_nu_all_eq<cl_uint>();
    error |= rft.run_nu_all_eq<cl_long>();
    error |= rft.run_nu_all_eq<cl_ulong>();
    error |= rft.run_nu_all_eq<cl_float>();
    error |= rft.run_nu_all_eq<cl_double>();
    error |= rft.run_nu_all_eq<subgroups::cl_half>();
    error |= rft.run_elect<int>();
    error |= rft.run_nu_any<int>();
    error |= rft.run_nu_all<int>();

    return error;
}