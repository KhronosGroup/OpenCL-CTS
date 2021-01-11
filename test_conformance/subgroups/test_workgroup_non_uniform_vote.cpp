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
namespace {
template <typename T> bool calculate(const T *mx, NonUniformVoteOp operation)
{
    bool taa = 1;
    switch (operation)
    {
        case NonUniformVoteOp::all_equal:
            for (int i = 0; i < NON_UNIFORM_WG_SIZE; ++i)
            {
                taa &=
                    compare_ordered(mx[i],
                                    mx[0]); // return non zero if all the same
            }
            break;
        default: log_error("Unknown operation request"); break;
    }
    return taa;
}
template <> bool calculate(const int *mx, NonUniformVoteOp operation)
{
    bool taa = false;
    switch (operation)
    {
        case NonUniformVoteOp::all_equal:
            taa = 1;
            for (int i = 0; i < NON_UNIFORM_WG_SIZE; ++i)
            {
                taa &=
                    compare_ordered(mx[i],
                                    mx[0]); // return non zero if all the same
            }
            break;
        case NonUniformVoteOp::any:
            taa = false;
            for (int i = 0; i < NON_UNIFORM_WG_SIZE; ++i)
            {
                // return non zero if value non zero at least for one
                taa |= mx[i] != 0;
            }
            break;
        case NonUniformVoteOp::all:
            taa = true;
            for (int i = 0; i < NON_UNIFORM_WG_SIZE; ++i)
            {
                // return non zero if value non zero for all
                taa &= mx[i] != 0;
            }
            break;
        default: log_error("Unknown operation request"); break;
    }
    return taa;
}

// Test any/all/all_equal non uniform test functions.
template <typename Ty, NonUniformVoteOp operation> struct AAN
{
    static void gen(Ty *x, Ty *t, cl_int *m, int ns, int nw, int ng)
    {
        int i, ii, j, k, n;
        int nj = (nw + ns - 1) / ns;

        ii = 0;
        for (k = 0; k < ng; ++k)
        { // for each work_group
            for (j = 0; j < nj; ++j)
            { // for each subgroup
                ii = j * ns;
                n = ii + ns > nw ? nw - ii : ns;
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

        log_info("  sub_group_non_uniform_%s(%s)...\n",
                 operation_names(operation), TypeManager<Ty>::name());

        for (k = 0; k < ng; ++k)
        { // for each work_group
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
                n = ii + ns > nw ? nw - ii : ns;

                // Compute target
                bool taa = calculate<Ty>(mx + ii, operation);

                // Check result
                static const Ty false_value{};
                for (i = 0; i < n && i < NON_UNIFORM_WG_SIZE; ++i)
                {
                    bool raa = !compare(my[ii + i], false_value);
                    if (raa != taa)
                    {
                        log_error("ERROR: sub_group_non_uniform_%s mismatch "
                                  "for local id %d in sub group %d in group "
                                  "%d, obtained %d, expected %d\n",
                                  operation_names(operation), i, j, k, raa,
                                  taa);
                        return -1;
                    }
                }
            }
            x += nw;
            y += nw;
            m += 4 * nw;
        }
        return 0;
    }
};

// Test for elect function.
// Discover only one elected work item in subgroup - with the
// lowest subgroup local id
struct ELECT
{
    static void gen(cl_int *x, cl_int *t, cl_int *m, int ns, int nw, int ng)
    {
        // no work here needed.
    }

    static int chk(cl_int *x, cl_int *y, cl_int *mx, cl_int *my, cl_int *m,
                   int ns, int nw, int ng)
    {
        int ii, i, j, k, n;
        int nj = (nw + ns - 1) / ns;
        cl_int tr, rr;
        log_info("  sub_group_elect...\n");

        for (k = 0; k < ng; ++k)
        { // for each work_group
            for (j = 0; j < nw; ++j)
            { // inside the work_group
                i = m[4 * j + 1] * ns + m[4 * j];
                my[i] = y[j]; // read device outputs for work_group
            }

            for (j = 0; j < nj; ++j)
            { // for each subgroup
                ii = j * ns;
                n = ii + ns > nw ? nw - ii : ns;
                rr = 0;
                for (i = 0; i < n; ++i)
                { // for each work_item in subgroup
                    my[ii + i] > 0
                        ? rr += 1
                        : rr += 0; // sum of output values should be 1
                }
                tr = 1; // expectation is that only one elected returned true
                if (rr != tr)
                {
                    log_error(
                        "ERROR: sub_group_elect() mismatch for sub group %d in "
                        "work group %d. Expected: %d Obtained: %d  \n",
                        j, k, tr, rr);
                    return -1;
                }
            }

            x += nw;
            y += nw;
            m += 4 * nw;
        }
        return 0;
    }
};

static const char *elect_source =
    "__kernel void test_elect(const __global Type *in, __global int4 *xy, "
    "__global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    "    Type x = in[gid];\n"
    "    int am_i_elected = sub_group_elect();\n"
    "    out[gid] = am_i_elected;\n" // one in subgroup true others false.
    "}\n";

static const char *non_uniform_any_source =
    "__kernel void test_non_uniform_any(const __global Type *in, __global int4 "
    "*xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    "    if (xy[gid].x < NON_UNIFORM_WG_SIZE) {\n"
    "        out[gid] = sub_group_non_uniform_any(in[gid]);\n"
    "    }\n"
    "}\n";
static const char *non_uniform_all_source =
    "__kernel void test_non_uniform_all(const __global Type *in, __global int4 "
    "*xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    "    if (xy[gid].x < NON_UNIFORM_WG_SIZE) {"
    "        out[gid] = sub_group_non_uniform_all(in[gid]);\n"
    "    }"
    "}\n";
static const char *non_uniform_all_equal_source =
    "__kernel void test_non_uniform_all_equal(const __global Type *in, "
    "__global int4 *xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    " if (xy[gid].x < NON_UNIFORM_WG_SIZE) {"
    "    out[gid] = sub_group_non_uniform_all_equal(in[gid]);\n"
    "}"
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
        int error = test<T, AAN<T, NonUniformVoteOp::all_equal>, GWS, LWS>::run(
            device_, context_, queue_, num_elements_,
            "test_non_uniform_all_equal", non_uniform_all_equal_source, 0,
            useCoreSubgroups_, required_extensions_);
        return error;
    }

    template <typename T> int run_elect()
    {
        int error = test<T, ELECT, GWS, LWS>::run(
            device_, context_, queue_, num_elements_, "test_elect",
            elect_source, 0, useCoreSubgroups_, required_extensions_);
        return error;
    }

    template <typename T> int run_nu_any()
    {
        int error = test<T, AAN<T, NonUniformVoteOp::any>, GWS, LWS>::run(
            device_, context_, queue_, num_elements_, "test_non_uniform_any",
            non_uniform_any_source, 0, useCoreSubgroups_, required_extensions_);
        return error;
    }

    template <typename T> int run_nu_all()
    {
        int error = test<T, AAN<T, NonUniformVoteOp::all>, GWS, LWS>::run(
            device_, context_, queue_, num_elements_, "test_non_uniform_all",
            non_uniform_all_source, 0, useCoreSubgroups_, required_extensions_);
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