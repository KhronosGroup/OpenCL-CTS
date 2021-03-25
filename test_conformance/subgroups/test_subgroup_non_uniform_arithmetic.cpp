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
#include "subgroup_common_templates.h"

namespace {

static const char *scinadd_non_uniform_source = R"(
    __kernel void test_scinadd_non_uniform(const __global Type *in, __global int4 *xy, __global Type *out) {
        int gid = get_global_id(0);
        XY(xy,gid);
        int elect_work_item = 1 << (get_sub_group_local_id() % 32);
            if (elect_work_item & WORK_ITEMS_MASK){
                out[gid] = sub_group_non_uniform_scan_inclusive_add(in[gid]);
            }
    }
)";

static const char *scinmax_non_uniform_source = R"(
    __kernel void test_scinmax_non_uniform(const __global Type *in, __global int4 *xy, __global Type *out) {
        int gid = get_global_id(0);
        XY(xy,gid);
        int elect_work_item = 1 << (get_sub_group_local_id() % 32);
            if (elect_work_item & WORK_ITEMS_MASK){
                out[gid] = sub_group_non_uniform_scan_inclusive_max(in[gid]);
            }
    }
)";

static const char *scinmin_non_uniform_source = R"(
    __kernel void test_scinmin_non_uniform(const __global Type *in, __global int4 *xy, __global Type *out) {
        int gid = get_global_id(0);
        XY(xy,gid);
        int elect_work_item = 1 << (get_sub_group_local_id() % 32);
            if (elect_work_item & WORK_ITEMS_MASK){
                out[gid] = sub_group_non_uniform_scan_inclusive_min(in[gid]);
            }
    }
)";

static const char *scinmul_non_uniform_source = R"(
    __kernel void test_scinmul_non_uniform(const __global Type *in, __global int4 *xy, __global Type *out) {
        int gid = get_global_id(0);
        XY(xy,gid);
        int elect_work_item = 1 << (get_sub_group_local_id() % 32);
            if (elect_work_item & WORK_ITEMS_MASK){
                out[gid] = sub_group_non_uniform_scan_inclusive_mul(in[gid]);
            }
    }
)";

static const char *scinand_non_uniform_source = R"(
    __kernel void test_scinand_non_uniform(const __global Type *in, __global int4 *xy, __global Type *out) {
        int gid = get_global_id(0);
        XY(xy,gid);
        int elect_work_item = 1 << (get_sub_group_local_id() % 32);
            if (elect_work_item & WORK_ITEMS_MASK){
                out[gid] = sub_group_non_uniform_scan_inclusive_and(in[gid]);
            }
    }
)";

static const char *scinor_non_uniform_source = R"(
    __kernel void test_scinor_non_uniform(const __global Type *in, __global int4 *xy, __global Type *out) {
        int gid = get_global_id(0);
        XY(xy,gid);
        int elect_work_item = 1 << (get_sub_group_local_id() % 32);
            if (elect_work_item & WORK_ITEMS_MASK){
                out[gid] = sub_group_non_uniform_scan_inclusive_or(in[gid]);
            }
    }
)";

static const char *scinxor_non_uniform_source = R"(
    __kernel void test_scinxor_non_uniform(const __global Type *in, __global int4 *xy, __global Type *out) {
        int gid = get_global_id(0);
        XY(xy,gid);
        int elect_work_item = 1 << (get_sub_group_local_id() % 32);
            if (elect_work_item & WORK_ITEMS_MASK){
                out[gid] = sub_group_non_uniform_scan_inclusive_xor(in[gid]);
            }
    }
)";

static const char *scinand_non_uniform_logical_source = R"(
    __kernel void test_scinand_non_uniform_logical(const __global Type *in, __global int4 *xy, __global Type *out) {
        int gid = get_global_id(0);
        XY(xy,gid);
        int elect_work_item = 1 << (get_sub_group_local_id() % 32);
            if (elect_work_item & WORK_ITEMS_MASK){
                out[gid] = sub_group_non_uniform_scan_inclusive_logical_and(in[gid]);
            }
    }
)";

static const char *scinor_non_uniform_logical_source = R"(
    __kernel void test_scinor_non_uniform_logical(const __global Type *in, __global int4 *xy, __global Type *out) {
        int gid = get_global_id(0);
        XY(xy,gid);
        int elect_work_item = 1 << (get_sub_group_local_id() % 32);
            if (elect_work_item & WORK_ITEMS_MASK){
                out[gid] = sub_group_non_uniform_scan_inclusive_logical_or(in[gid]);
            }
    }
)";

static const char *scinxor_non_uniform_logical_source = R"(
    __kernel void test_scinxor_non_uniform_logical(const __global Type *in, __global int4 *xy, __global Type *out) {
        int gid = get_global_id(0);
        XY(xy,gid);
        int elect_work_item = 1 << (get_sub_group_local_id() % 32);
            if (elect_work_item & WORK_ITEMS_MASK){
                out[gid] = sub_group_non_uniform_scan_inclusive_logical_xor(in[gid]);
            }
    }
)";

static const char *scexadd_non_uniform_source = R"(
    __kernel void test_scexadd_non_uniform(const __global Type *in, __global int4 *xy, __global Type *out) {
        int gid = get_global_id(0);
        XY(xy,gid);
        int elect_work_item = 1 << (get_sub_group_local_id() % 32);
            if (elect_work_item & WORK_ITEMS_MASK){
                out[gid] = sub_group_non_uniform_scan_exclusive_add(in[gid]);
            }
    }
)";

static const char *scexmax_non_uniform_source = R"(
    __kernel void test_scexmax_non_uniform(const __global Type *in, __global int4 *xy, __global Type *out) {
        int gid = get_global_id(0);
        XY(xy,gid);
        int elect_work_item = 1 << (get_sub_group_local_id() % 32);
            if (elect_work_item & WORK_ITEMS_MASK){
                out[gid] = sub_group_non_uniform_scan_exclusive_max(in[gid]);
            }
    }
)";

static const char *scexmin_non_uniform_source = R"(
    __kernel void test_scexmin_non_uniform(const __global Type *in, __global int4 *xy, __global Type *out) {
        int gid = get_global_id(0);
        XY(xy,gid);
        int elect_work_item = 1 << (get_sub_group_local_id() % 32);
            if (elect_work_item & WORK_ITEMS_MASK){
                out[gid] = sub_group_non_uniform_scan_exclusive_min(in[gid]);
            }
    }
)";

static const char *scexmul_non_uniform_source = R"(
    __kernel void test_scexmul_non_uniform(const __global Type *in, __global int4 *xy, __global Type *out) {
        int gid = get_global_id(0);
        XY(xy,gid);
        int elect_work_item = 1 << (get_sub_group_local_id() % 32);
            if (elect_work_item & WORK_ITEMS_MASK){
                out[gid] = sub_group_non_uniform_scan_exclusive_mul(in[gid]);
            }
    }
)";

static const char *scexand_non_uniform_source = R"(
    __kernel void test_scexand_non_uniform(const __global Type *in, __global int4 *xy, __global Type *out) {
        int gid = get_global_id(0);
        XY(xy,gid);
        int elect_work_item = 1 << (get_sub_group_local_id() % 32);
            if (elect_work_item & WORK_ITEMS_MASK){
                out[gid] = sub_group_non_uniform_scan_exclusive_and(in[gid]);
            }
    }
)";

static const char *scexor_non_uniform_source = R"(
    __kernel void test_scexor_non_uniform(const __global Type *in, __global int4 *xy, __global Type *out) {
        int gid = get_global_id(0);
        XY(xy,gid);
        int elect_work_item = 1 << (get_sub_group_local_id() % 32);
            if (elect_work_item & WORK_ITEMS_MASK){
                out[gid] = sub_group_non_uniform_scan_exclusive_or(in[gid]);
            }
    }
)";

static const char *scexxor_non_uniform_source = R"(
    __kernel void test_scexxor_non_uniform(const __global Type *in, __global int4 *xy, __global Type *out) {
        int gid = get_global_id(0);
        XY(xy,gid);
        int elect_work_item = 1 << (get_sub_group_local_id() % 32);
            if (elect_work_item & WORK_ITEMS_MASK){
                out[gid] = sub_group_non_uniform_scan_exclusive_xor(in[gid]);
            }
    }
)";

static const char *scexand_non_uniform_logical_source = R"(
    __kernel void test_scexand_non_uniform_logical(const __global Type *in, __global int4 *xy, __global Type *out) {
        int gid = get_global_id(0);
        XY(xy,gid);
        int elect_work_item = 1 << (get_sub_group_local_id() % 32);
            if (elect_work_item & WORK_ITEMS_MASK){
                out[gid] = sub_group_non_uniform_scan_exclusive_logical_and(in[gid]);
            }
    }
)";

static const char *scexor_non_uniform_logical_source = R"(
    __kernel void test_scexor_non_uniform_logical(const __global Type *in, __global int4 *xy, __global Type *out) {
        int gid = get_global_id(0);
        XY(xy,gid);
        int elect_work_item = 1 << (get_sub_group_local_id() % 32);
            if (elect_work_item & WORK_ITEMS_MASK){
                out[gid] = sub_group_non_uniform_scan_exclusive_logical_or(in[gid]);
            }
    }
)";

static const char *scexxor_non_uniform_logical_source = R"(
    __kernel void test_scexxor_non_uniform_logical(const __global Type *in, __global int4 *xy, __global Type *out) {
        int gid = get_global_id(0);
        XY(xy,gid);
        int elect_work_item = 1 << (get_sub_group_local_id() % 32);
            if (elect_work_item & WORK_ITEMS_MASK){
                out[gid] = sub_group_non_uniform_scan_exclusive_logical_xor(in[gid]);
            }
    }
)";

static const char *redadd_non_uniform_source = R"(
    __kernel void test_redadd_non_uniform(const __global Type *in, __global int4 *xy, __global Type *out) {
        int gid = get_global_id(0);
        XY(xy,gid);
        int elect_work_item = 1 << (get_sub_group_local_id() % 32);
            if (elect_work_item & WORK_ITEMS_MASK){
                out[gid] = sub_group_non_uniform_reduce_add(in[gid]);
            }
    }
)";

static const char *redmax_non_uniform_source = R"(
    __kernel void test_redmax_non_uniform(const __global Type *in, __global int4 *xy, __global Type *out) {
        int gid = get_global_id(0);
        XY(xy,gid);
        int elect_work_item = 1 << (get_sub_group_local_id() % 32);
            if (elect_work_item & WORK_ITEMS_MASK){
                out[gid] = sub_group_non_uniform_reduce_max(in[gid]);
            }
    }
)";

static const char *redmin_non_uniform_source = R"(
    __kernel void test_redmin_non_uniform(const __global Type *in, __global int4 *xy, __global Type *out) {
        int gid = get_global_id(0);
        XY(xy,gid);
        int elect_work_item = 1 << (get_sub_group_local_id() % 32);
            if (elect_work_item & WORK_ITEMS_MASK){
                out[gid] = sub_group_non_uniform_reduce_min(in[gid]);
            }
    }
)";

static const char *redmul_non_uniform_source = R"(
    __kernel void test_redmul_non_uniform(const __global Type *in, __global int4 *xy, __global Type *out) {
        int gid = get_global_id(0);
        XY(xy,gid);
        int elect_work_item = 1 << (get_sub_group_local_id() % 32);
            if (elect_work_item & WORK_ITEMS_MASK){
                out[gid] = sub_group_non_uniform_reduce_mul(in[gid]);
            }
    }
)";

static const char *redand_non_uniform_source = R"(
    __kernel void test_redand_non_uniform(const __global Type *in, __global int4 *xy, __global Type *out) {
        int gid = get_global_id(0);
        XY(xy,gid);
        int elect_work_item = 1 << (get_sub_group_local_id() % 32);
            if (elect_work_item & WORK_ITEMS_MASK){
                out[gid] = sub_group_non_uniform_reduce_and(in[gid]);
            }
    }
)";

static const char *redor_non_uniform_source = R"(
    __kernel void test_redor_non_uniform(const __global Type *in, __global int4 *xy, __global Type *out) {
        int gid = get_global_id(0);
        XY(xy,gid);
        int elect_work_item = 1 << (get_sub_group_local_id() % 32);
            if (elect_work_item & WORK_ITEMS_MASK){
                out[gid] = sub_group_non_uniform_reduce_or(in[gid]);
            }
    }
)";

static const char *redxor_non_uniform_source = R"(
    __kernel void test_redxor_non_uniform(const __global Type *in, __global int4 *xy, __global Type *out) {
        int gid = get_global_id(0);
        XY(xy,gid);
        int elect_work_item = 1 << (get_sub_group_local_id() % 32);
            if (elect_work_item & WORK_ITEMS_MASK){
                out[gid] = sub_group_non_uniform_reduce_xor(in[gid]);
            }
    }
)";

static const char *redand_non_uniform_logical_source = R"(
    __kernel void test_redand_non_uniform_logical(const __global Type *in, __global int4 *xy, __global Type *out) {
        int gid = get_global_id(0);
        XY(xy,gid);
        int elect_work_item = 1 << (get_sub_group_local_id() % 32);
            if (elect_work_item & WORK_ITEMS_MASK){
                out[gid] = sub_group_non_uniform_reduce_logical_and(in[gid]);
            }
    }
)";

static const char *redor_non_uniform_logical_source = R"(
    __kernel void test_redor_non_uniform_logical(const __global Type *in, __global int4 *xy, __global Type *out) {
        int gid = get_global_id(0);
        XY(xy,gid);
        int elect_work_item = 1 << (get_sub_group_local_id() % 32);
            if (elect_work_item & WORK_ITEMS_MASK){
                out[gid] = sub_group_non_uniform_reduce_logical_or(in[gid]);
            }
    }
)";

static const char *redxor_non_uniform_logical_source = R"(
    __kernel void test_redxor_non_uniform_logical(const __global Type *in, __global int4 *xy, __global Type *out) {
        int gid = get_global_id(0);
        XY(xy,gid);
        int elect_work_item = 1 << (get_sub_group_local_id() % 32);
            if (elect_work_item & WORK_ITEMS_MASK){
                out[gid] = sub_group_non_uniform_reduce_logical_xor(in[gid]);
            }
    }
)";

template <typename T>
int run_functions_add_mul_max_min_for_type(RunTestForType rft)
{
    int error = rft.run_impl<T, SCIN_NU<T, ArithmeticOp::add_>>(
        "test_scinadd_non_uniform", scinadd_non_uniform_source);
    error |= rft.run_impl<T, SCIN_NU<T, ArithmeticOp::mul_>>(
        "test_scinmul_non_uniform", scinmul_non_uniform_source);
    error |= rft.run_impl<T, SCIN_NU<T, ArithmeticOp::max_>>(
        "test_scinmax_non_uniform", scinmax_non_uniform_source);
    error |= rft.run_impl<T, SCIN_NU<T, ArithmeticOp::min_>>(
        "test_scinmin_non_uniform", scinmin_non_uniform_source);
    error |= rft.run_impl<T, SCEX_NU<T, ArithmeticOp::add_>>(
        "test_scexadd_non_uniform", scexadd_non_uniform_source);
    error |= rft.run_impl<T, SCEX_NU<T, ArithmeticOp::mul_>>(
        "test_scexmul_non_uniform", scexmul_non_uniform_source);
    error |= rft.run_impl<T, SCEX_NU<T, ArithmeticOp::max_>>(
        "test_scexmax_non_uniform", scexmax_non_uniform_source);
    error |= rft.run_impl<T, SCEX_NU<T, ArithmeticOp::min_>>(
        "test_scexmin_non_uniform", scexmin_non_uniform_source);
    error |= rft.run_impl<T, RED_NU<T, ArithmeticOp::add_>>(
        "test_redadd_non_uniform", redadd_non_uniform_source);
    error |= rft.run_impl<T, RED_NU<T, ArithmeticOp::mul_>>(
        "test_redmul_non_uniform", redmul_non_uniform_source);
    error |= rft.run_impl<T, RED_NU<T, ArithmeticOp::max_>>(
        "test_redmax_non_uniform", redmax_non_uniform_source);
    error |= rft.run_impl<T, RED_NU<T, ArithmeticOp::min_>>(
        "test_redmin_non_uniform", redmin_non_uniform_source);
    return error;
}

template <typename T> int run_functions_and_or_xor_for_type(RunTestForType rft)
{
    int error = rft.run_impl<T, SCIN_NU<T, ArithmeticOp::and_>>(
        "test_scinand_non_uniform", scinand_non_uniform_source);
    error |= rft.run_impl<T, SCIN_NU<T, ArithmeticOp::or_>>(
        "test_scinor_non_uniform", scinor_non_uniform_source);
    error |= rft.run_impl<T, SCIN_NU<T, ArithmeticOp::xor_>>(
        "test_scinxor_non_uniform", scinxor_non_uniform_source);
    error |= rft.run_impl<T, SCEX_NU<T, ArithmeticOp::and_>>(
        "test_scexand_non_uniform", scexand_non_uniform_source);
    error |= rft.run_impl<T, SCEX_NU<T, ArithmeticOp::or_>>(
        "test_scexor_non_uniform", scexor_non_uniform_source);
    error |= rft.run_impl<T, SCEX_NU<T, ArithmeticOp::xor_>>(
        "test_scexxor_non_uniform", scexxor_non_uniform_source);
    error |= rft.run_impl<T, RED_NU<T, ArithmeticOp::and_>>(
        "test_redand_non_uniform", redand_non_uniform_source);
    error |= rft.run_impl<T, RED_NU<T, ArithmeticOp::or_>>(
        "test_redor_non_uniform", redor_non_uniform_source);
    error |= rft.run_impl<T, RED_NU<T, ArithmeticOp::xor_>>(
        "test_redxor_non_uniform", redxor_non_uniform_source);
    return error;
}

template <typename T>
int run_functions_logical_and_or_xor_for_type(RunTestForType rft)
{
    int error = rft.run_impl<T, SCIN_NU<T, ArithmeticOp::logical_and>>(
        "test_scinand_non_uniform_logical", scinand_non_uniform_logical_source);
    error |= rft.run_impl<T, SCIN_NU<T, ArithmeticOp::logical_or>>(
        "test_scinor_non_uniform_logical", scinor_non_uniform_logical_source);
    error |= rft.run_impl<T, SCIN_NU<T, ArithmeticOp::logical_xor>>(
        "test_scinxor_non_uniform_logical", scinxor_non_uniform_logical_source);
    error |= rft.run_impl<T, SCEX_NU<T, ArithmeticOp::logical_and>>(
        "test_scexand_non_uniform_logical", scexand_non_uniform_logical_source);
    error |= rft.run_impl<T, SCEX_NU<T, ArithmeticOp::logical_or>>(
        "test_scexor_non_uniform_logical", scexor_non_uniform_logical_source);
    error |= rft.run_impl<T, SCEX_NU<T, ArithmeticOp::logical_xor>>(
        "test_scexxor_non_uniform_logical", scexxor_non_uniform_logical_source);
    error |= rft.run_impl<T, RED_NU<T, ArithmeticOp::logical_and>>(
        "test_redand_non_uniform_logical", redand_non_uniform_logical_source);
    error |= rft.run_impl<T, RED_NU<T, ArithmeticOp::logical_or>>(
        "test_redor_non_uniform_logical", redor_non_uniform_logical_source);
    error |= rft.run_impl<T, RED_NU<T, ArithmeticOp::logical_xor>>(
        "test_redxor_non_uniform_logical", redxor_non_uniform_logical_source);
    return error;
}

}

int test_subgroup_functions_non_uniform_arithmetic(cl_device_id device,
                                                   cl_context context,
                                                   cl_command_queue queue,
                                                   int num_elements)
{
    std::vector<std::string> required_extensions = {
        "cl_khr_subgroup_non_uniform_arithmetic"
    };
    std::vector<uint32_t> masks{ 0xffffffff, 0x55aaaa55, 0x5555aaaa, 0xaaaa5555,
                                 0x0f0ff0f0, 0x0f0f0f0f, 0xff0000ff, 0xff00ff00,
                                 0x00ffff00, 0x80000000, 0xaaaaaaaa };

    constexpr size_t global_work_size = 2000;
    constexpr size_t local_work_size = 200;
    WorkGroupParams test_params(global_work_size, local_work_size,
                                required_extensions, masks);
    RunTestForType rft(device, context, queue, num_elements, test_params);

    int error = run_functions_add_mul_max_min_for_type<cl_int>(rft);
    error |= run_functions_add_mul_max_min_for_type<cl_uint>(rft);
    error |= run_functions_add_mul_max_min_for_type<cl_long>(rft);
    error |= run_functions_add_mul_max_min_for_type<cl_ulong>(rft);
    error |= run_functions_add_mul_max_min_for_type<cl_short>(rft);
    error |= run_functions_add_mul_max_min_for_type<cl_ushort>(rft);
    error |= run_functions_add_mul_max_min_for_type<cl_char>(rft);
    error |= run_functions_add_mul_max_min_for_type<cl_uchar>(rft);
    error |= run_functions_add_mul_max_min_for_type<cl_float>(rft);
    error |= run_functions_add_mul_max_min_for_type<cl_double>(rft);
    error |= run_functions_add_mul_max_min_for_type<subgroups::cl_half>(rft);

    error |= run_functions_and_or_xor_for_type<cl_int>(rft);
    error |= run_functions_and_or_xor_for_type<cl_uint>(rft);
    error |= run_functions_and_or_xor_for_type<cl_long>(rft);
    error |= run_functions_and_or_xor_for_type<cl_ulong>(rft);
    error |= run_functions_and_or_xor_for_type<cl_short>(rft);
    error |= run_functions_and_or_xor_for_type<cl_ushort>(rft);
    error |= run_functions_and_or_xor_for_type<cl_char>(rft);
    error |= run_functions_and_or_xor_for_type<cl_uchar>(rft);

    error |= run_functions_logical_and_or_xor_for_type<cl_int>(rft);
    return error;
}