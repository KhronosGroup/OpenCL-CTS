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
#include "subgroup_common_kernels.h"
#include "subgroup_common_templates.h"
#include "harness/conversions.h"
#include "harness/typeWrappers.h"

namespace {
// Any/All test functions
template <NonUniformVoteOp operation> struct AA
{
    static void gen(cl_int *x, cl_int *t, cl_int *m,
                    const WorkGroupParams &test_params)
    {
        int i, ii, j, k, n;
        int ng = test_params.global_workgroup_size;
        int nw = test_params.local_workgroup_size;
        int ns = test_params.subgroup_size;
        int nj = (nw + ns - 1) / ns;
        int e;
        ng = ng / nw;
        ii = 0;
        log_info("  sub_group_%s...\n", operation_names(operation));
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
                x[j] = t[j];
            }

            x += nw;
            m += 4 * nw;
        }
    }

    static int chk(cl_int *x, cl_int *y, cl_int *mx, cl_int *my, cl_int *m,
                   const WorkGroupParams &test_params)
    {
        int ii, i, j, k, n;
        int ng = test_params.global_workgroup_size;
        int nw = test_params.local_workgroup_size;
        int ns = test_params.subgroup_size;
        int nj = (nw + ns - 1) / ns;
        cl_int taa, raa;
        ng = ng / nw;

        for (k = 0; k < ng; ++k)
        {
            // Map to array indexed to array indexed by local ID and sub group
            for (j = 0; j < nw; ++j)
            {
                mx[j] = x[j];
                my[j] = y[j];
            }

            for (j = 0; j < nj; ++j)
            {
                ii = j * ns;
                n = ii + ns > nw ? nw - ii : ns;

                // Compute target
                if (operation == NonUniformVoteOp::any)
                {
                    taa = 0;
                    for (i = 0; i < n; ++i) taa |= mx[ii + i] != 0;
                }

                if (operation == NonUniformVoteOp::all)
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
                                  operation_names(operation), i, j, k);
                        return TEST_FAIL;
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


template <typename T>
int run_broadcast_scan_reduction_for_type(RunTestForType rft)
{
    int error = rft.run_impl<T, BC<T, SubgroupsBroadcastOp::broadcast>>(
        "test_bcast", bcast_source);
    error |= rft.run_impl<T, RED_NU<T, ArithmeticOp::add_>>("test_redadd",
                                                            redadd_source);
    error |= rft.run_impl<T, RED_NU<T, ArithmeticOp::max_>>("test_redmax",
                                                            redmax_source);
    error |= rft.run_impl<T, RED_NU<T, ArithmeticOp::min_>>("test_redmin",
                                                            redmin_source);
    error |= rft.run_impl<T, SCIN_NU<T, ArithmeticOp::add_>>("test_scinadd",
                                                             scinadd_source);
    error |= rft.run_impl<T, SCIN_NU<T, ArithmeticOp::max_>>("test_scinmax",
                                                             scinmax_source);
    error |= rft.run_impl<T, SCIN_NU<T, ArithmeticOp::min_>>("test_scinmin",
                                                             scinmin_source);
    error |= rft.run_impl<T, SCEX_NU<T, ArithmeticOp::add_>>("test_scexadd",
                                                             scexadd_source);
    error |= rft.run_impl<T, SCEX_NU<T, ArithmeticOp::max_>>("test_scexmax",
                                                             scexmax_source);
    error |= rft.run_impl<T, SCEX_NU<T, ArithmeticOp::min_>>("test_scexmin",
                                                             scexmin_source);
    return error;
}

}
// Entry point from main
int test_subgroup_functions(cl_device_id device, cl_context context,
                            cl_command_queue queue, int num_elements,
                            bool useCoreSubgroups)
{
    constexpr size_t global_work_size = 2000;
    constexpr size_t local_work_size = 200;
    WorkGroupParams test_params(global_work_size, local_work_size);
    RunTestForType rft(device, context, queue, num_elements, test_params);
    int error =
        rft.run_impl<cl_int, AA<NonUniformVoteOp::any>>("test_any", any_source);
    error |=
        rft.run_impl<cl_int, AA<NonUniformVoteOp::all>>("test_all", all_source);
    error |= run_broadcast_scan_reduction_for_type<cl_int>(rft);
    error |= run_broadcast_scan_reduction_for_type<cl_uint>(rft);
    error |= run_broadcast_scan_reduction_for_type<cl_long>(rft);
    error |= run_broadcast_scan_reduction_for_type<cl_ulong>(rft);
    error |= run_broadcast_scan_reduction_for_type<cl_float>(rft);
    error |= run_broadcast_scan_reduction_for_type<cl_double>(rft);
    error |= run_broadcast_scan_reduction_for_type<subgroups::cl_half>(rft);
    return error;
}

int test_subgroup_functions_core(cl_device_id device, cl_context context,
                                 cl_command_queue queue, int num_elements)
{
    return test_subgroup_functions(device, context, queue, num_elements, true);
}

int test_subgroup_functions_ext(cl_device_id device, cl_context context,
                                cl_command_queue queue, int num_elements)
{
    bool hasExtension = is_extension_available(device, "cl_khr_subgroups");

    if (!hasExtension)
    {
        log_info(
            "Device does not support 'cl_khr_subgroups'. Skipping the test.\n");
        return TEST_SKIPPED_ITSELF;
    }
    return test_subgroup_functions(device, context, queue, num_elements, false);
}
