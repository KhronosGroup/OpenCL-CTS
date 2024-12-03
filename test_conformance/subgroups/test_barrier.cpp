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
#include "harness/conversions.h"
#include "harness/typeWrappers.h"

static const char *lbar_source =
    "__kernel void test_lbar(const __global Type *in, __global int2 *xy, "
    "__global Type *out)\n"
    "{\n"
    "    __local int tmp[200];\n"
    "    int gid = get_global_id(0);\n"
    "    int nid = get_sub_group_size();\n"
    "    int lid = get_sub_group_local_id();\n"
    "    xy[gid].x = lid;\n"
    "    xy[gid].y = get_sub_group_id();\n"
    "    if (get_sub_group_id() == 0) {\n"
    "        tmp[lid] = in[gid];\n"
    "        sub_group_barrier(CLK_LOCAL_MEM_FENCE);\n"
    "        out[gid] = tmp[nid-1-lid];\n"
    "    } else {\n"
    "        out[gid] = -in[gid];\n"
    "    }\n"
    "}\n";

static const char *gbar_source =
    "__kernel void test_gbar(const __global Type *in, __global int2 *xy, "
    "__global Type *out, __global Type *tmp)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    int nid = get_sub_group_size();\n"
    "    int lid = get_sub_group_local_id();\n"
    "    int tof = get_group_id(0)*get_max_sub_group_size();\n"
    "    xy[gid].x = lid;\n"
    "    xy[gid].y = get_sub_group_id();\n"
    "    if (get_sub_group_id() == 0) {\n"
    "        tmp[tof+lid] = in[gid];\n"
    "        sub_group_barrier(CLK_GLOBAL_MEM_FENCE);\n"
    "        out[gid] = tmp[tof+nid-1-lid];\n"
    "    } else {\n"
    "        out[gid] = -in[gid];\n"
    "    }\n"
    "}\n";

// barrier test functions
template <int Which> struct BAR
{
    static void log_test(const WorkGroupParams &test_params,
                         const char *extra_text)
    {
        if (Which == 0)
            log_info("  sub_group_barrier(CLK_LOCAL_MEM_FENCE)...%s\n",
                     extra_text);
        else
            log_info("  sub_group_barrier(CLK_GLOBAL_MEM_FENCE)...%s\n",
                     extra_text);
    }

    static void gen(cl_int *x, cl_int *t, cl_int *m,
                    const WorkGroupParams &test_params)
    {
        int i, ii, j, k, n;
        int nw = test_params.local_workgroup_size;
        int ns = test_params.subgroup_size;
        int ng = test_params.global_workgroup_size;
        int nj = (nw + ns - 1) / ns;
        ng = ng / nw;

        ii = 0;
        for (k = 0; k < ng; ++k)
        {
            for (j = 0; j < nj; ++j)
            {
                ii = j * ns;
                n = ii + ns > nw ? nw - ii : ns;

                for (i = 0; i < n; ++i) t[ii + i] = genrand_int32(gMTdata);
            }

            // Now map into work group using map from device
            for (j = 0; j < nw; ++j)
            {
                x[j] = t[j];
            }

            x += nw;
            m += 2 * nw;
        }
    }

    static test_status chk(cl_int *x, cl_int *y, cl_int *mx, cl_int *my,
                           cl_int *m, const WorkGroupParams &test_params)
    {
        int ii, i, j, k, n;
        int nw = test_params.local_workgroup_size;
        int ns = test_params.subgroup_size;
        int ng = test_params.global_workgroup_size;
        int nj = (nw + ns - 1) / ns;
        ng = ng / nw;
        cl_int tr, rr;

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

                for (i = 0; i < n; ++i)
                {
                    tr = j == 0 ? mx[ii + n - 1 - i] : -mx[ii + i];
                    rr = my[ii + i];

                    if (tr != rr)
                    {
                        log_error("ERROR: sub_group_barrier mismatch for local "
                                  "id %d in sub group %d in group %d expected "
                                  "%d got %d\n",
                                  i, j, k, tr, rr);
                        return TEST_FAIL;
                    }
                }
            }

            x += nw;
            y += nw;
            m += 2 * nw;
        }

        return TEST_PASS;
    }
};

// Entry point from main
int test_barrier_functions(cl_device_id device, cl_context context,
                           cl_command_queue queue, int num_elements,
                           bool useCoreSubgroups)
{
    int error = TEST_PASS;

    // Adjust these individually below if desired/needed
    constexpr size_t global_work_size = 2000;
    constexpr size_t local_work_size = 200;
    WorkGroupParams test_params(global_work_size, local_work_size);
    test_params.use_core_subgroups = useCoreSubgroups;
    error = subgroup_test<cl_int, BAR<0>>::run(device, context, queue,
                                               num_elements, "test_lbar",
                                               lbar_source, test_params);
    error |= subgroup_test<cl_int, BAR<1>, global_work_size>::run(
        device, context, queue, num_elements, "test_gbar", gbar_source,
        test_params);

    return error;
}

int test_barrier_functions_core(cl_device_id device, cl_context context,
                                cl_command_queue queue, int num_elements)
{
    return test_barrier_functions(device, context, queue, num_elements, true);
}

int test_barrier_functions_ext(cl_device_id device, cl_context context,
                               cl_command_queue queue, int num_elements)
{
    bool hasExtension = is_extension_available(device, "cl_khr_subgroups");

    if (!hasExtension)
    {
        log_info(
            "Device does not support 'cl_khr_subgroups'. Skipping the test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    return test_barrier_functions(device, context, queue, num_elements, false);
}
