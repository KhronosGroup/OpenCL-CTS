//
// Copyright (c) 2017 The Khronos Group Inc.
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
#include "workgroup_common_kernels.h"
#include "workgroup_common_templates.h"
#include "harness/conversions.h"
#include "harness/typeWrappers.h"

// Any/All test functions
template <int Which> struct AA
{
    static void gen(cl_int *x, cl_int *t, cl_int *m, int ns, int nw, int ng)
    {
        int i, ii, j, k, n;
        int nj = (nw + ns - 1) / ns;
        int e;

        ii = 0;
        for (k = 0; k < ng; ++k)
        {
            for (j = 0; j < nj; ++j)
            {
                ii = j * ns;
                n = ii + ns > nw ? nw - ii : ns;
                e = (int)(genrand_int32(gMTdata) % 3);

                // Initialize data matrix indexed by local id and sub group id
                switch (e)
                {
                    case 0: memset(&t[ii], 0, n * sizeof(cl_int)); break;
                    case 1:
                        memset(&t[ii], 0, n * sizeof(cl_int));
                        i = (int)(genrand_int32(gMTdata) % (cl_uint)n);
                        t[ii + i] = 41;
                        break;
                    case 2: memset(&t[ii], 0xff, n * sizeof(cl_int)); break;
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

    static int chk(cl_int *x, cl_int *y, cl_int *mx, cl_int *my, cl_int *m,
                   int ns, int nw, int ng)
    {
        int ii, i, j, k, n;
        int nj = (nw + ns - 1) / ns;
        cl_int taa, raa;

        log_info("  sub_group_%s...\n", Which == 0 ? "any" : "all");

        for (k = 0; k < ng; ++k)
        {
            // Map to array indexed to array indexed by local ID and sub group
            for (j = 0; j < nw; ++j)
            {
                i = m[4 * j + 1] * ns + m[4 * j];
                mx[i] = x[j];
                my[i] = y[j];
            }

            for (j = 0; j < nj; ++j)
            {
                ii = j * ns;
                n = ii + ns > nw ? nw - ii : ns;

                // Compute target
                if (Which == 0)
                {
                    taa = 0;
                    for (i = 0; i < n; ++i) taa |= mx[ii + i] != 0;
                }
                else
                {
                    taa = 1;
                    for (i = 0; i < n; ++i) taa &= mx[ii + i] != 0;
                }

                // Check result
                for (i = 0; i < n; ++i)
                {
                    raa = my[ii + i] != 0;
                    if (raa != taa)
                    {
                        log_error("ERROR: sub_group_%s mismatch for local id "
                                  "%d in sub group %d in group %d\n",
                                  Which == 0 ? "any" : "all", i, j, k);
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

static const char *any_source = "__kernel void test_any(const __global Type "
                                "*in, __global int4 *xy, __global Type *out)\n"
                                "{\n"
                                "    int gid = get_global_id(0);\n"
                                "    XY(xy,gid);\n"
                                "    out[gid] = sub_group_any(in[gid]);\n"
                                "}\n";

static const char *all_source = "__kernel void test_all(const __global Type "
                                "*in, __global int4 *xy, __global Type *out)\n"
                                "{\n"
                                "    int gid = get_global_id(0);\n"
                                "    XY(xy,gid);\n"
                                "    out[gid] = sub_group_all(in[gid]);\n"
                                "}\n";

struct run_for_type
{
    run_for_type(cl_device_id device, cl_context context,
                 cl_command_queue queue, int num_elements,
                 bool useCoreSubgroups)
    {
        device_ = device;
        context_ = context;
        queue_ = queue;
        num_elements_ = num_elements;
        useCoreSubgroups_ = useCoreSubgroups;
    }

    template <typename T> cl_int run()
    {
        cl_int error;
        error = test<T, BC<T>, G, L>::run(device_, context_, queue_,
                                          num_elements_, "test_bcast",
                                          bcast_source, 0, useCoreSubgroups_);
        error |= test<T, RED<T, 0>, G, L>::run(
            device_, context_, queue_, num_elements_, "test_redadd",
            redadd_source, 0, useCoreSubgroups_);
        error |= test<T, RED<T, 1>, G, L>::run(
            device_, context_, queue_, num_elements_, "test_redmax",
            redmax_source, 0, useCoreSubgroups_);
        error |= test<T, RED<T, 2>, G, L>::run(
            device_, context_, queue_, num_elements_, "test_redmin",
            redmin_source, 0, useCoreSubgroups_);
        error |= test<T, SCIN<T, 0>, G, L>::run(
            device_, context_, queue_, num_elements_, "test_scinadd",
            scinadd_source, 0, useCoreSubgroups_);
        error |= test<T, SCIN<T, 1>, G, L>::run(
            device_, context_, queue_, num_elements_, "test_scinmax",
            scinmax_source, 0, useCoreSubgroups_);
        error |= test<T, SCIN<T, 2>, G, L>::run(
            device_, context_, queue_, num_elements_, "test_scinmin",
            scinmin_source, 0, useCoreSubgroups_);
        error |= test<T, SCEX<T, 0>, G, L>::run(
            device_, context_, queue_, num_elements_, "test_scexadd",
            scexadd_source, 0, useCoreSubgroups_);
        error |= test<T, SCEX<T, 1>, G, L>::run(
            device_, context_, queue_, num_elements_, "test_scexmax",
            scexmax_source, 0, useCoreSubgroups_);
        error |= test<T, SCEX<T, 2>, G, L>::run(
            device_, context_, queue_, num_elements_, "test_scexmin",
            scexmin_source, 0, useCoreSubgroups_);

        return error;
    }

private:
    cl_device_id device_;
    cl_context context_;
    cl_command_queue queue_;
    int num_elements_;
    bool useCoreSubgroups_;
};

// Entry point from main
int test_work_group_functions(cl_device_id device, cl_context context,
                              cl_command_queue queue, int num_elements,
                              bool useCoreSubgroups)
{
    int error;
    error = test<int, AA<0>, G, L>::run(device, context, queue, num_elements,
                                        "test_any", any_source, 0,
                                        useCoreSubgroups);
    error |= test<int, AA<1>, G, L>::run(device, context, queue, num_elements,
                                         "test_all", all_source, 0,
                                         useCoreSubgroups);
    run_for_type rft(device, context, queue, num_elements, useCoreSubgroups);
    error |= rft.run<cl_uint>();
    error |= rft.run<cl_int>();
    error |= rft.run<cl_ulong>();
    error |= rft.run<cl_long>();
    error |= rft.run<cl_float>();
    error |= rft.run<cl_double>();
    // error |= rft.run<cl_half>();
    return error;
}

int test_work_group_functions_core(cl_device_id device, cl_context context,
                                   cl_command_queue queue, int num_elements)
{
    return test_work_group_functions(device, context, queue, num_elements,
                                     true);
}

int test_work_group_functions_ext(cl_device_id device, cl_context context,
                                  cl_command_queue queue, int num_elements)
{
    bool hasExtension = is_extension_available(device, "cl_khr_subgroups");

    if (!hasExtension)
    {
        log_info(
            "Device does not support 'cl_khr_subgroups'. Skipping the test.\n");
        return TEST_SKIPPED_ITSELF;
    }
    return test_work_group_functions(device, context, queue, num_elements,
                                     false);
}
