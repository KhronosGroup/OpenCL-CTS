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
#include "subgroup_common_templates.h"
#include "harness/typeWrappers.h"

#define CLUSTER_SIZE 4
#define CLUSTER_SIZE_STR "4"

namespace {
std::string sub_group_clustered_reduce_source = R"(
__kernel void test_%s(const __global Type *in, __global int4 *xy, __global Type *out) {
        int gid = get_global_id(0);
        XY(xy,gid);
        xy[gid].w = 0;
        if (sizeof(in[gid]) == sizeof(%s(in[gid], )" CLUSTER_SIZE_STR R"())) {
            xy[gid].w = sizeof(in[gid]);
        }
        out[gid] = %s(in[gid], )" CLUSTER_SIZE_STR R"();
}       
)";

// DESCRIPTION:
// Test for reduce cluster functions
template <typename Ty, ArithmeticOp operation> struct RED_CLU
{
    static void gen(Ty *x, Ty *t, cl_int *m, const WorkGroupParams &test_params)
    {
        int nw = test_params.local_workgroup_size;
        int ns = test_params.subgroup_size;
        int ng = test_params.global_workgroup_size;
        ng = ng / nw;
        log_info("  sub_group_clustered_reduce_%s(%s, %d bytes) ...\n",
                 operation_names(operation), TypeManager<Ty>::name(),
                 sizeof(Ty));
        genrand<Ty, operation>(x, t, m, ns, nw, ng);
    }

    static int chk(Ty *x, Ty *y, Ty *mx, Ty *my, cl_int *m,
                   const WorkGroupParams &test_params)
    {
        int nw = test_params.local_workgroup_size;
        int ns = test_params.subgroup_size;
        int ng = test_params.global_workgroup_size;
        int nj = (nw + ns - 1) / ns;
        ng = ng / nw;

        for (int k = 0; k < ng; ++k)
        {
            std::vector<cl_int> data_type_sizes;
            // Map to array indexed to array indexed by local ID and sub group
            for (int j = 0; j < nw; ++j)
            {
                mx[j] = x[j];
                my[j] = y[j];
                data_type_sizes.push_back(m[4 * j + 3]);
            }

            for (cl_int dts : data_type_sizes)
            {
                if (dts != sizeof(Ty))
                {
                    log_error("ERROR: sub_group_clustered_reduce_%s(%s) "
                              "wrong data type size detected, expected: %d, "
                              "used by device %d, in group %d\n",
                              operation_names(operation),
                              TypeManager<Ty>::name(), sizeof(Ty), dts, k);
                    return TEST_FAIL;
                }
            }

            for (int j = 0; j < nj; ++j)
            {
                int ii = j * ns;
                int n = ii + ns > nw ? nw - ii : ns;
                int midx = 4 * ii + 2;
                std::vector<Ty> clusters_results;
                int clusters_counter = ns / CLUSTER_SIZE;
                clusters_results.resize(clusters_counter);

                // Compute target
                Ty tr = mx[ii];
                for (int i = 0; i < n; ++i)
                {
                    if (i % CLUSTER_SIZE == 0)
                        tr = mx[ii + i];
                    else
                        tr = calculate<Ty>(tr, mx[ii + i], operation);
                    clusters_results[i / CLUSTER_SIZE] = tr;
                }

                // Check result
                for (int i = 0; i < n; ++i)
                {
                    Ty rr = my[ii + i];
                    tr = clusters_results[i / CLUSTER_SIZE];
                    if (!compare(rr, tr))
                    {
                        log_error(
                            "ERROR: sub_group_clustered_reduce_%s(%s) mismatch "
                            "for local id %d in sub group %d in group %d\n",
                            operation_names(operation), TypeManager<Ty>::name(),
                            i, j, k);
                        return TEST_FAIL;
                    }
                }
            }

            x += nw;
            y += nw;
            m += 4 * nw;
        }
        log_info("  sub_group_clustered_reduce_%s(%s, %d bytes) ... passed\n",
                 operation_names(operation), TypeManager<Ty>::name(),
                 sizeof(Ty));
        return TEST_PASS;
    }
};

template <typename T>
int run_cluster_red_add_max_min_mul_for_type(RunTestForType rft)
{
    int error = rft.run_impl<T, RED_CLU<T, ArithmeticOp::add_>>(
        "sub_group_clustered_reduce_add");
    error |= rft.run_impl<T, RED_CLU<T, ArithmeticOp::max_>>(
        "sub_group_clustered_reduce_max");
    error |= rft.run_impl<T, RED_CLU<T, ArithmeticOp::min_>>(
        "sub_group_clustered_reduce_min");
    error |= rft.run_impl<T, RED_CLU<T, ArithmeticOp::mul_>>(
        "sub_group_clustered_reduce_mul");
    return error;
}
template <typename T> int run_cluster_and_or_xor_for_type(RunTestForType rft)
{
    int error = rft.run_impl<T, RED_CLU<T, ArithmeticOp::and_>>(
        "sub_group_clustered_reduce_and");
    error |= rft.run_impl<T, RED_CLU<T, ArithmeticOp::or_>>(
        "sub_group_clustered_reduce_or");
    error |= rft.run_impl<T, RED_CLU<T, ArithmeticOp::xor_>>(
        "sub_group_clustered_reduce_xor");
    return error;
}
template <typename T>
int run_cluster_logical_and_or_xor_for_type(RunTestForType rft)
{
    int error = rft.run_impl<T, RED_CLU<T, ArithmeticOp::logical_and>>(
        "sub_group_clustered_reduce_logical_and");
    error |= rft.run_impl<T, RED_CLU<T, ArithmeticOp::logical_or>>(
        "sub_group_clustered_reduce_logical_or");
    error |= rft.run_impl<T, RED_CLU<T, ArithmeticOp::logical_xor>>(
        "sub_group_clustered_reduce_logical_xor");

    return error;
}
}

int test_subgroup_functions_clustered_reduce(cl_device_id device,
                                             cl_context context,
                                             cl_command_queue queue,
                                             int num_elements)
{
    if (!is_extension_available(device, "cl_khr_subgroup_clustered_reduce"))
    {
        log_info("cl_khr_subgroup_clustered_reduce is not supported on this "
                 "device, skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    constexpr size_t global_work_size = 2000;
    constexpr size_t local_work_size = 200;
    WorkGroupParams test_params(global_work_size, local_work_size);
    test_params.save_kernel_source(sub_group_clustered_reduce_source);
    RunTestForType rft(device, context, queue, num_elements, test_params);

    int error = run_cluster_red_add_max_min_mul_for_type<cl_int>(rft);
    error |= run_cluster_red_add_max_min_mul_for_type<cl_uint>(rft);
    error |= run_cluster_red_add_max_min_mul_for_type<cl_long>(rft);
    error |= run_cluster_red_add_max_min_mul_for_type<cl_ulong>(rft);
    error |= run_cluster_red_add_max_min_mul_for_type<cl_short>(rft);
    error |= run_cluster_red_add_max_min_mul_for_type<cl_ushort>(rft);
    error |= run_cluster_red_add_max_min_mul_for_type<cl_char>(rft);
    error |= run_cluster_red_add_max_min_mul_for_type<cl_uchar>(rft);
    error |= run_cluster_red_add_max_min_mul_for_type<cl_float>(rft);
    error |= run_cluster_red_add_max_min_mul_for_type<cl_double>(rft);
    error |= run_cluster_red_add_max_min_mul_for_type<subgroups::cl_half>(rft);

    error |= run_cluster_and_or_xor_for_type<cl_int>(rft);
    error |= run_cluster_and_or_xor_for_type<cl_uint>(rft);
    error |= run_cluster_and_or_xor_for_type<cl_long>(rft);
    error |= run_cluster_and_or_xor_for_type<cl_ulong>(rft);
    error |= run_cluster_and_or_xor_for_type<cl_short>(rft);
    error |= run_cluster_and_or_xor_for_type<cl_ushort>(rft);
    error |= run_cluster_and_or_xor_for_type<cl_char>(rft);
    error |= run_cluster_and_or_xor_for_type<cl_uchar>(rft);

    error |= run_cluster_logical_and_or_xor_for_type<cl_int>(rft);
    return error;
}
