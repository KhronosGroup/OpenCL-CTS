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
#include <bitset>

namespace {
// Test for ballot functions
template <typename Ty> struct BALLOT
{
    static void log_test(const WorkGroupParams &test_params,
                         const char *extra_text)
    {
        log_info("  sub_group_ballot...%s\n", extra_text);
    }

    static void gen(Ty *x, Ty *t, cl_int *m, const WorkGroupParams &test_params)
    {
        int gws = test_params.global_workgroup_size;
        int lws = test_params.local_workgroup_size;
        int sbs = test_params.subgroup_size;
        int sb_number = (lws + sbs - 1) / sbs;
        int non_uniform_size = gws % lws;
        int wg_number = gws / lws;
        wg_number = non_uniform_size ? wg_number + 1 : wg_number;
        int last_subgroup_size = 0;

        for (int wg_id = 0; wg_id < wg_number; ++wg_id)
        { // for each work_group
            if (non_uniform_size && wg_id == wg_number - 1)
            {
                set_last_workgroup_params(non_uniform_size, sb_number, sbs, lws,
                                          last_subgroup_size);
            }
            for (int sb_id = 0; sb_id < sb_number; ++sb_id)
            { // for each subgroup
                int wg_offset = sb_id * sbs;
                int current_sbs;
                if (last_subgroup_size && sb_id == sb_number - 1)
                {
                    current_sbs = last_subgroup_size;
                }
                else
                {
                    current_sbs = wg_offset + sbs > lws ? lws - wg_offset : sbs;
                }

                for (int wi_id = 0; wi_id < current_sbs; wi_id++)
                {
                    cl_uint v;
                    if (genrand_bool(gMTdata))
                    {
                        v = genrand_bool(gMTdata);
                    }
                    else if (genrand_bool(gMTdata))
                    {
                        v = 1U << ((genrand_int32(gMTdata) % 31) + 1);
                    }
                    else
                    {
                        v = genrand_int32(gMTdata);
                    }
                    cl_uint4 v4 = { v, 0, 0, 0 };
                    t[wi_id + wg_offset] = v4;
                }
            }
            // Now map into work group using map from device
            for (int wi_id = 0; wi_id < lws; ++wi_id)
            {
                x[wi_id] = t[wi_id];
            }
            x += lws;
            m += 4 * lws;
        }
    }

    static test_status chk(Ty *x, Ty *y, Ty *mx, Ty *my, cl_int *m,
                           const WorkGroupParams &test_params)
    {
        int gws = test_params.global_workgroup_size;
        int lws = test_params.local_workgroup_size;
        int sbs = test_params.subgroup_size;
        int sb_number = (lws + sbs - 1) / sbs;
        int non_uniform_size = gws % lws;
        int wg_number = gws / lws;
        wg_number = non_uniform_size ? wg_number + 1 : wg_number;
        int last_subgroup_size = 0;

        for (int wg_id = 0; wg_id < wg_number; ++wg_id)
        { // for each work_group
            if (non_uniform_size && wg_id == wg_number - 1)
            {
                set_last_workgroup_params(non_uniform_size, sb_number, sbs, lws,
                                          last_subgroup_size);
            }
            for (int wi_id = 0; wi_id < lws; ++wi_id)
            { // inside the work_group
                mx[wi_id] = x[wi_id]; // read host inputs for work_group
                my[wi_id] = y[wi_id]; // read device outputs for work_group
            }

            for (int sb_id = 0; sb_id < sb_number; ++sb_id)
            { // for each subgroup
                int wg_offset = sb_id * sbs;
                int current_sbs;
                if (last_subgroup_size && sb_id == sb_number - 1)
                {
                    current_sbs = last_subgroup_size;
                }
                else
                {
                    current_sbs = wg_offset + sbs > lws ? lws - wg_offset : sbs;
                }

                bs128 expected_result_bs = 0;

                std::set<int> active_work_items;
                for (int wi_id = 0; wi_id < current_sbs; ++wi_id)
                {
                    if (test_params.work_items_mask.test(wi_id))
                    {
                        bool predicate = (mx[wg_offset + wi_id].s0 != 0);
                        expected_result_bs |= (bs128(predicate) << wi_id);
                        active_work_items.insert(wi_id);
                    }
                }
                if (active_work_items.empty())
                {
                    continue;
                }

                cl_uint4 expected_result =
                    bs128_to_cl_uint4(expected_result_bs);
                for (const int &active_work_item : active_work_items)
                {
                    int wi_id = active_work_item;

                    cl_uint4 device_result = my[wg_offset + wi_id];
                    bs128 device_result_bs = cl_uint4_to_bs128(device_result);

                    if (device_result_bs != expected_result_bs)
                    {
                        log_error(
                            "ERROR: sub_group_ballot mismatch for local id "
                            "%d in sub group %d in group %d obtained {%d, %d, "
                            "%d, %d}, expected {%d, %d, %d, %d}\n",
                            wi_id, sb_id, wg_id, device_result.s0,
                            device_result.s1, device_result.s2,
                            device_result.s3, expected_result.s0,
                            expected_result.s1, expected_result.s2,
                            expected_result.s3);
                        return TEST_FAIL;
                    }
                }
            }

            x += lws;
            y += lws;
            m += 4 * lws;
        }

        return TEST_PASS;
    }
};

// Test for bit extract ballot functions
template <typename Ty, BallotOp operation> struct BALLOT_BIT_EXTRACT
{
    static void log_test(const WorkGroupParams &test_params,
                         const char *extra_text)
    {
        log_info("  sub_group_ballot_%s(%s)...%s\n", operation_names(operation),
                 TypeManager<Ty>::name(), extra_text);
    }

    static void gen(Ty *x, Ty *t, cl_int *m, const WorkGroupParams &test_params)
    {
        int wi_id, sb_id, wg_id;
        int gws = test_params.global_workgroup_size;
        int lws = test_params.local_workgroup_size;
        int sbs = test_params.subgroup_size;
        int sb_number = (lws + sbs - 1) / sbs;
        int wg_number = gws / lws;
        int limit_sbs = sbs > 100 ? 100 : sbs;

        for (wg_id = 0; wg_id < wg_number; ++wg_id)
        { // for each work_group
            for (sb_id = 0; sb_id < sb_number; ++sb_id)
            { // for each subgroup
                int wg_offset = sb_id * sbs;
                int current_sbs = wg_offset + sbs > lws ? lws - wg_offset : sbs;
                // rand index to bit extract
                int index_for_odd = (int)(genrand_int32(gMTdata) & 0x7fffffff)
                    % (limit_sbs > current_sbs ? current_sbs : limit_sbs);
                int index_for_even = (int)(genrand_int32(gMTdata) & 0x7fffffff)
                    % (limit_sbs > current_sbs ? current_sbs : limit_sbs);
                for (wi_id = 0; wi_id < current_sbs; ++wi_id)
                {
                    // index of the third element int the vector.
                    int midx = 4 * wg_offset + 4 * wi_id + 2;
                    // storing information about index to bit extract
                    m[midx] = (cl_int)index_for_odd;
                    m[++midx] = (cl_int)index_for_even;
                }
                set_randomdata_for_subgroup<Ty>(t, wg_offset, current_sbs);
            }

            // Now map into work group using map from device
            for (wi_id = 0; wi_id < lws; ++wi_id)
            {
                x[wi_id] = t[wi_id];
            }

            x += lws;
            m += 4 * lws;
        }
    }

    static test_status chk(Ty *x, Ty *y, Ty *mx, Ty *my, cl_int *m,
                           const WorkGroupParams &test_params)
    {
        int wi_id, wg_id, sb_id;
        int gws = test_params.global_workgroup_size;
        int lws = test_params.local_workgroup_size;
        int sbs = test_params.subgroup_size;
        int sb_number = (lws + sbs - 1) / sbs;
        int wg_number = gws / lws;
        cl_uint4 expected_result, device_result;
        int last_subgroup_size = 0;
        int current_sbs = 0;
        int non_uniform_size = gws % lws;

        for (wg_id = 0; wg_id < wg_number; ++wg_id)
        { // for each work_group
            if (non_uniform_size && wg_id == wg_number - 1)
            {
                set_last_workgroup_params(non_uniform_size, sb_number, sbs, lws,
                                          last_subgroup_size);
            }
            // Map to array indexed to array indexed by local ID and sub group
            for (wi_id = 0; wi_id < lws; ++wi_id)
            { // inside the work_group
                // read host inputs for work_group
                mx[wi_id] = x[wi_id];
                // read device outputs for work_group
                my[wi_id] = y[wi_id];
            }

            for (sb_id = 0; sb_id < sb_number; ++sb_id)
            { // for each subgroup
                int wg_offset = sb_id * sbs;
                if (last_subgroup_size && sb_id == sb_number - 1)
                {
                    current_sbs = last_subgroup_size;
                }
                else
                {
                    current_sbs = wg_offset + sbs > lws ? lws - wg_offset : sbs;
                }
                // take index of array where info which work_item will
                // be broadcast its value is stored
                int midx = 4 * wg_offset + 2;
                // take subgroup local id of this work_item
                int index_for_odd = (int)m[midx];
                int index_for_even = (int)m[++midx];

                for (wi_id = 0; wi_id < current_sbs; ++wi_id)
                { // for each subgroup
                    int bit_value = 0;
                    // from which value of bitfield bit
                    // verification will be done
                    int take_shift =
                        (wi_id & 1) ? index_for_odd % 32 : index_for_even % 32;
                    int bit_mask = 1 << take_shift;

                    if (wi_id < 32)
                        (mx[wg_offset + wi_id].s0 & bit_mask) > 0
                            ? bit_value = 1
                            : bit_value = 0;
                    if (wi_id >= 32 && wi_id < 64)
                        (mx[wg_offset + wi_id].s1 & bit_mask) > 0
                            ? bit_value = 1
                            : bit_value = 0;
                    if (wi_id >= 64 && wi_id < 96)
                        (mx[wg_offset + wi_id].s2 & bit_mask) > 0
                            ? bit_value = 1
                            : bit_value = 0;
                    if (wi_id >= 96 && wi_id < 128)
                        (mx[wg_offset + wi_id].s3 & bit_mask) > 0
                            ? bit_value = 1
                            : bit_value = 0;

                    if (wi_id & 1)
                    {
                        bit_value ? expected_result = { 1, 0, 0, 1 }
                                  : expected_result = { 0, 0, 0, 1 };
                    }
                    else
                    {
                        bit_value ? expected_result = { 1, 0, 0, 2 }
                                  : expected_result = { 0, 0, 0, 2 };
                    }

                    device_result = my[wg_offset + wi_id];
                    if (!compare(device_result, expected_result))
                    {
                        log_error(
                            "ERROR: sub_group_%s mismatch for local id %d in "
                            "sub group %d in group %d obtained {%d, %d, %d, "
                            "%d}, expected {%d, %d, %d, %d}\n",
                            operation_names(operation), wi_id, sb_id, wg_id,
                            device_result.s0, device_result.s1,
                            device_result.s2, device_result.s3,
                            expected_result.s0, expected_result.s1,
                            expected_result.s2, expected_result.s3);
                        return TEST_FAIL;
                    }
                }
            }
            x += lws;
            y += lws;
            m += 4 * lws;
        }
        return TEST_PASS;
    }
};

template <typename Ty, BallotOp operation> struct BALLOT_INVERSE
{
    static void log_test(const WorkGroupParams &test_params,
                         const char *extra_text)
    {
        log_info("  sub_group_inverse_ballot...%s\n", extra_text);
    }

    static void gen(Ty *x, Ty *t, cl_int *m, const WorkGroupParams &test_params)
    {
        // no work here
    }

    static test_status chk(Ty *x, Ty *y, Ty *mx, Ty *my, cl_int *m,
                           const WorkGroupParams &test_params)
    {
        int wi_id, wg_id, sb_id;
        int gws = test_params.global_workgroup_size;
        int lws = test_params.local_workgroup_size;
        int sbs = test_params.subgroup_size;
        int sb_number = (lws + sbs - 1) / sbs;
        cl_uint4 expected_result, device_result;
        int non_uniform_size = gws % lws;
        int wg_number = gws / lws;
        int last_subgroup_size = 0;
        int current_sbs = 0;
        if (non_uniform_size) wg_number++;

        for (wg_id = 0; wg_id < wg_number; ++wg_id)
        { // for each work_group
            if (non_uniform_size && wg_id == wg_number - 1)
            {
                set_last_workgroup_params(non_uniform_size, sb_number, sbs, lws,
                                          last_subgroup_size);
            }
            // Map to array indexed to array indexed by local ID and sub group
            for (wi_id = 0; wi_id < lws; ++wi_id)
            { // inside the work_group
                mx[wi_id] = x[wi_id]; // read host inputs for work_group
                my[wi_id] = y[wi_id]; // read device outputs for work_group
            }

            for (sb_id = 0; sb_id < sb_number; ++sb_id)
            { // for each subgroup
                int wg_offset = sb_id * sbs;
                if (last_subgroup_size && sb_id == sb_number - 1)
                {
                    current_sbs = last_subgroup_size;
                }
                else
                {
                    current_sbs = wg_offset + sbs > lws ? lws - wg_offset : sbs;
                }
                // take subgroup local id of this work_item
                // Check result
                for (wi_id = 0; wi_id < current_sbs; ++wi_id)
                { // for each subgroup work item

                    wi_id & 1 ? expected_result = { 1, 0, 0, 1 }
                              : expected_result = { 1, 0, 0, 2 };

                    device_result = my[wg_offset + wi_id];
                    if (!compare(device_result, expected_result))
                    {
                        log_error(
                            "ERROR: sub_group_%s mismatch for local id %d in "
                            "sub group %d in group %d obtained {%d, %d, %d, "
                            "%d}, expected {%d, %d, %d, %d}\n",
                            operation_names(operation), wi_id, sb_id, wg_id,
                            device_result.s0, device_result.s1,
                            device_result.s2, device_result.s3,
                            expected_result.s0, expected_result.s1,
                            expected_result.s2, expected_result.s3);
                        return TEST_FAIL;
                    }
                }
            }
            x += lws;
            y += lws;
            m += 4 * lws;
        }

        return TEST_PASS;
    }
};


// Test for bit count/inclusive and exclusive scan/ find lsb msb ballot function
template <typename Ty, BallotOp operation> struct BALLOT_COUNT_SCAN_FIND
{
    static void log_test(const WorkGroupParams &test_params,
                         const char *extra_text)
    {
        log_info("  sub_group_%s(%s)...%s\n", operation_names(operation),
                 TypeManager<Ty>::name(), extra_text);
    }

    static void gen(Ty *x, Ty *t, cl_int *m, const WorkGroupParams &test_params)
    {
        int wi_id, wg_id, sb_id;
        int gws = test_params.global_workgroup_size;
        int lws = test_params.local_workgroup_size;
        int sbs = test_params.subgroup_size;
        int sb_number = (lws + sbs - 1) / sbs;
        int non_uniform_size = gws % lws;
        int wg_number = gws / lws;
        int last_subgroup_size = 0;
        int current_sbs = 0;

        if (non_uniform_size)
        {
            wg_number++;
        }
        for (wg_id = 0; wg_id < wg_number; ++wg_id)
        { // for each work_group
            if (non_uniform_size && wg_id == wg_number - 1)
            {
                set_last_workgroup_params(non_uniform_size, sb_number, sbs, lws,
                                          last_subgroup_size);
            }
            for (sb_id = 0; sb_id < sb_number; ++sb_id)
            { // for each subgroup
                int wg_offset = sb_id * sbs;
                if (last_subgroup_size && sb_id == sb_number - 1)
                {
                    current_sbs = last_subgroup_size;
                }
                else
                {
                    current_sbs = wg_offset + sbs > lws ? lws - wg_offset : sbs;
                }
                if (operation == BallotOp::ballot_bit_count
                    || operation == BallotOp::ballot_inclusive_scan
                    || operation == BallotOp::ballot_exclusive_scan)
                {
                    set_randomdata_for_subgroup<Ty>(t, wg_offset, current_sbs);
                }
                else if (operation == BallotOp::ballot_find_lsb
                         || operation == BallotOp::ballot_find_msb)
                {
                    // Regarding to the spec, find lsb and find msb result is
                    // undefined behavior if input value is zero, so generate
                    // only non-zero values.
                    for (wi_id = 0; wi_id < current_sbs; ++wi_id)
                    {
                        char x = (genrand_int32(gMTdata)) & 0xff;
                        // undefined behaviour in case of 0;
                        x = x ? x : 1;
                        memset(&t[wg_offset + wi_id], x, sizeof(Ty));
                    }
                }
                else
                {
                    log_error("Unknown operation...\n");
                }
            }

            // Now map into work group using map from device
            for (wi_id = 0; wi_id < lws; ++wi_id)
            {
                x[wi_id] = t[wi_id];
            }

            x += lws;
            m += 4 * lws;
        }
    }

    static bs128 getImportantBits(cl_uint sub_group_local_id,
                                  cl_uint sub_group_size)
    {
        bs128 mask;
        if (operation == BallotOp::ballot_bit_count
            || operation == BallotOp::ballot_find_lsb
            || operation == BallotOp::ballot_find_msb)
        {
            for (cl_uint i = 0; i < sub_group_size; ++i) mask.set(i);
        }
        else if (operation == BallotOp::ballot_inclusive_scan
                 || operation == BallotOp::ballot_exclusive_scan)
        {
            for (cl_uint i = 0; i < sub_group_local_id; ++i) mask.set(i);
            if (operation == BallotOp::ballot_inclusive_scan)
                mask.set(sub_group_local_id);
        }
        return mask;
    }

    static test_status chk(Ty *x, Ty *y, Ty *mx, Ty *my, cl_int *m,
                           const WorkGroupParams &test_params)
    {
        int wi_id, wg_id, sb_id;
        int gws = test_params.global_workgroup_size;
        int lws = test_params.local_workgroup_size;
        int sbs = test_params.subgroup_size;
        int sb_number = (lws + sbs - 1) / sbs;
        int non_uniform_size = gws % lws;
        int wg_number = gws / lws;
        wg_number = non_uniform_size ? wg_number + 1 : wg_number;
        cl_uint expected_result, device_result;
        int last_subgroup_size = 0;
        int current_sbs = 0;

        for (wg_id = 0; wg_id < wg_number; ++wg_id)
        { // for each work_group
            if (non_uniform_size && wg_id == wg_number - 1)
            {
                set_last_workgroup_params(non_uniform_size, sb_number, sbs, lws,
                                          last_subgroup_size);
            }
            // Map to array indexed to array indexed by local ID and sub group
            for (wi_id = 0; wi_id < lws; ++wi_id)
            { // inside the work_group
                // read host inputs for work_group
                mx[wi_id] = x[wi_id];
                // read device outputs for work_group
                my[wi_id] = y[wi_id];
            }

            for (sb_id = 0; sb_id < sb_number; ++sb_id)
            { // for each subgroup
                int wg_offset = sb_id * sbs;
                if (last_subgroup_size && sb_id == sb_number - 1)
                {
                    current_sbs = last_subgroup_size;
                }
                else
                {
                    current_sbs = wg_offset + sbs > lws ? lws - wg_offset : sbs;
                }
                // Check result
                expected_result = 0;
                for (wi_id = 0; wi_id < current_sbs; ++wi_id)
                { // for subgroup element
                    bs128 bs;
                    // convert cl_uint4 input into std::bitset<128>
                    bs |= bs128(mx[wg_offset + wi_id].s0)
                        | (bs128(mx[wg_offset + wi_id].s1) << 32)
                        | (bs128(mx[wg_offset + wi_id].s2) << 64)
                        | (bs128(mx[wg_offset + wi_id].s3) << 96);
                    bs &= getImportantBits(wi_id, sbs);
                    device_result = my[wg_offset + wi_id].s0;
                    if (operation == BallotOp::ballot_inclusive_scan
                        || operation == BallotOp::ballot_exclusive_scan
                        || operation == BallotOp::ballot_bit_count)
                    {
                        expected_result = bs.count();
                        if (!compare(device_result, expected_result))
                        {
                            log_error("ERROR: sub_group_%s "
                                      "mismatch for local id %d in sub group "
                                      "%d in group %d obtained %d, "
                                      "expected %d\n",
                                      operation_names(operation), wi_id, sb_id,
                                      wg_id, device_result, expected_result);
                            return TEST_FAIL;
                        }
                    }
                    else if (operation == BallotOp::ballot_find_lsb)
                    {
                        if (bs.none())
                        {
                            // Return value is undefined when no bits are set,
                            // so skip validation:
                            continue;
                        }
                        for (int id = 0; id < sbs; ++id)
                        {
                            if (bs.test(id))
                            {
                                expected_result = id;
                                break;
                            }
                        }
                        if (!compare(device_result, expected_result))
                        {
                            log_error("ERROR: sub_group_ballot_find_lsb "
                                      "mismatch for local id %d in sub group "
                                      "%d in group %d obtained %d, "
                                      "expected %d\n",
                                      wi_id, sb_id, wg_id, device_result,
                                      expected_result);
                            return TEST_FAIL;
                        }
                    }
                    else if (operation == BallotOp::ballot_find_msb)
                    {
                        if (bs.none())
                        {
                            // Return value is undefined when no bits are set,
                            // so skip validation:
                            continue;
                        }
                        for (int id = sbs - 1; id >= 0; --id)
                        {
                            if (bs.test(id))
                            {
                                expected_result = id;
                                break;
                            }
                        }
                        if (!compare(device_result, expected_result))
                        {
                            log_error("ERROR: sub_group_ballot_find_msb "
                                      "mismatch for local id %d in sub group "
                                      "%d in group %d obtained %d, "
                                      "expected %d\n",
                                      wi_id, sb_id, wg_id, device_result,
                                      expected_result);
                            return TEST_FAIL;
                        }
                    }
                }
            }
            x += lws;
            y += lws;
            m += 4 * lws;
        }
        return TEST_PASS;
    }
};

// test mask functions
template <typename Ty, BallotOp operation> struct SMASK
{
    static void log_test(const WorkGroupParams &test_params,
                         const char *extra_text)
    {
        log_info("  get_sub_group_%s_mask...%s\n", operation_names(operation),
                 extra_text);
    }

    static void gen(Ty *x, Ty *t, cl_int *m, const WorkGroupParams &test_params)
    {
        int wi_id, wg_id, sb_id;
        int gws = test_params.global_workgroup_size;
        int lws = test_params.local_workgroup_size;
        int sbs = test_params.subgroup_size;
        int sb_number = (lws + sbs - 1) / sbs;
        int wg_number = gws / lws;
        for (wg_id = 0; wg_id < wg_number; ++wg_id)
        { // for each work_group
            for (sb_id = 0; sb_id < sb_number; ++sb_id)
            { // for each subgroup
                int wg_offset = sb_id * sbs;
                int current_sbs = wg_offset + sbs > lws ? lws - wg_offset : sbs;
                // Produce expected masks for each work item in the subgroup
                for (wi_id = 0; wi_id < current_sbs; ++wi_id)
                {
                    int midx = 4 * wg_offset + 4 * wi_id;
                    cl_uint max_sub_group_size = m[midx + 2];
                    cl_uint4 expected_mask = { 0 };
                    expected_mask = generate_bit_mask(
                        wi_id, operation_names(operation), max_sub_group_size);
                    set_value(t[wg_offset + wi_id], expected_mask);
                }
            }

            // Now map into work group using map from device
            for (wi_id = 0; wi_id < lws; ++wi_id)
            {
                x[wi_id] = t[wi_id];
            }
            x += lws;
            m += 4 * lws;
        }
    }

    static test_status chk(Ty *x, Ty *y, Ty *mx, Ty *my, cl_int *m,
                           const WorkGroupParams &test_params)
    {
        int wi_id, wg_id, sb_id;
        int gws = test_params.global_workgroup_size;
        int lws = test_params.local_workgroup_size;
        int sbs = test_params.subgroup_size;
        int sb_number = (lws + sbs - 1) / sbs;
        Ty expected_result, device_result;
        int wg_number = gws / lws;

        for (wg_id = 0; wg_id < wg_number; ++wg_id)
        { // for each work_group
            for (wi_id = 0; wi_id < lws; ++wi_id)
            { // inside the work_group
                mx[wi_id] = x[wi_id]; // read host inputs for work_group
                my[wi_id] = y[wi_id]; // read device outputs for work_group
            }

            for (sb_id = 0; sb_id < sb_number; ++sb_id)
            {
                int wg_offset = sb_id * sbs;
                int current_sbs = wg_offset + sbs > lws ? lws - wg_offset : sbs;

                // Check result
                for (wi_id = 0; wi_id < current_sbs; ++wi_id)
                { // inside the subgroup
                    expected_result =
                        mx[wg_offset + wi_id]; // read host input for subgroup
                    device_result =
                        my[wg_offset
                           + wi_id]; // read device outputs for subgroup
                    if (!compare(device_result, expected_result))
                    {
                        log_error("ERROR:  get_sub_group_%s_mask... mismatch "
                                  "for local id %d in sub group %d in group "
                                  "%d, obtained %d, expected %d\n",
                                  operation_names(operation), wi_id, sb_id,
                                  wg_id, device_result, expected_result);
                        return TEST_FAIL;
                    }
                }
            }
            x += lws;
            y += lws;
            m += 4 * lws;
        }
        return TEST_PASS;
    }
};

std::string sub_group_non_uniform_broadcast_source = R"(
__kernel void test_sub_group_non_uniform_broadcast(const __global Type *in, __global int4 *xy, __global Type *out) {
    int gid = get_global_id(0);
    XY(xy,gid);
    Type x = in[gid];
    if (xy[gid].x < NR_OF_ACTIVE_WORK_ITEMS) {
        out[gid] = sub_group_non_uniform_broadcast(x, xy[gid].z);
    } else {
        out[gid] = sub_group_non_uniform_broadcast(x, xy[gid].w);
    }
}
)";
std::string sub_group_broadcast_first_source = R"(
__kernel void test_sub_group_broadcast_first(const __global Type *in, __global int4 *xy, __global Type *out) {
    int gid = get_global_id(0);
    XY(xy,gid);
    Type x = in[gid];
    if (xy[gid].x < NR_OF_ACTIVE_WORK_ITEMS) {
        out[gid] = sub_group_broadcast_first(x);;
    } else {
        out[gid] = sub_group_broadcast_first(x);;
    }
}
)";
std::string sub_group_ballot_bit_scan_find_source = R"(
__kernel void test_%s(const __global Type *in, __global int4 *xy, __global Type *out) {
    int gid = get_global_id(0);
    XY(xy,gid);
    Type x = in[gid];
    uint4 value = (uint4)(0,0,0,0);
    value = (uint4)(%s(x),0,0,0);
    out[gid] = value;
}
)";
std::string sub_group_ballot_mask_source = R"(
__kernel void test_%s(const __global Type *in, __global int4 *xy, __global Type *out) {
    int gid = get_global_id(0);
    XY(xy,gid);
    xy[gid].z = get_max_sub_group_size();
    Type x = in[gid];
    uint4 mask = %s();
    out[gid] = mask;
}
)";
std::string sub_group_ballot_source = R"(
__kernel void test_sub_group_ballot(const __global Type *in, __global int4 *xy, __global Type *out, uint4 work_item_mask_vector) {
    uint gid = get_global_id(0);
    XY(xy,gid);
    uint subgroup_local_id = get_sub_group_local_id();
    uint elect_work_item = 1 << (subgroup_local_id % 32);
    uint work_item_mask;
    if (subgroup_local_id < 32) {
        work_item_mask = work_item_mask_vector.x;
    } else if(subgroup_local_id < 64) {
        work_item_mask = work_item_mask_vector.y;
    } else if(subgroup_local_id < 96) {
        work_item_mask = work_item_mask_vector.z;
    } else if(subgroup_local_id < 128) {
        work_item_mask = work_item_mask_vector.w;
    }
    uint4 value = (uint4)(0, 0, 0, 0);
    if (elect_work_item & work_item_mask) {
        value = sub_group_ballot(in[gid].s0);
    }
    out[gid] = value;
}
)";
std::string sub_group_inverse_ballot_source = R"(
__kernel void test_sub_group_inverse_ballot(const __global Type *in, __global int4 *xy, __global Type *out) {
    int gid = get_global_id(0);
    XY(xy,gid);
    Type x = in[gid];
    uint4 value = (uint4)(10,0,0,0);
    if (get_sub_group_local_id() & 1) {
        uint4 partial_ballot_mask = (uint4)(0xAAAAAAAA,0xAAAAAAAA,0xAAAAAAAA,0xAAAAAAAA);
        if (sub_group_inverse_ballot(partial_ballot_mask)) {
            value = (uint4)(1,0,0,1);
        } else {
            value = (uint4)(0,0,0,1);
        }
    } else {
        uint4 partial_ballot_mask = (uint4)(0x55555555,0x55555555,0x55555555,0x55555555);
        if (sub_group_inverse_ballot(partial_ballot_mask)) {
            value = (uint4)(1,0,0,2);
        } else {
            value = (uint4)(0,0,0,2);
        }
    }
    out[gid] = value;
}
)";
std::string sub_group_ballot_bit_extract_source = R"(
 __kernel void test_sub_group_ballot_bit_extract(const __global Type *in, __global int4 *xy, __global Type *out) {
    int gid = get_global_id(0);
    XY(xy,gid);
    Type x = in[gid];
    uint index = xy[gid].z;
    uint4 value = (uint4)(10,0,0,0);
    if (get_sub_group_local_id() & 1) {
        if (sub_group_ballot_bit_extract(x, xy[gid].z)) {
            value = (uint4)(1,0,0,1);
        } else {
            value = (uint4)(0,0,0,1);
        }
    } else {
        if (sub_group_ballot_bit_extract(x, xy[gid].w)) {
            value = (uint4)(1,0,0,2);
        } else {
            value = (uint4)(0,0,0,2);
        }
    }
    out[gid] = value;
}
)";

template <typename T> int run_non_uniform_broadcast_for_type(RunTestForType rft)
{
    int error =
        rft.run_impl<T, BC<T, SubgroupsBroadcastOp::non_uniform_broadcast>>(
            "sub_group_non_uniform_broadcast");
    return error;
}


}

int test_subgroup_functions_ballot(cl_device_id device, cl_context context,
                                   cl_command_queue queue, int num_elements)
{
    if (!is_extension_available(device, "cl_khr_subgroup_ballot"))
    {
        log_info("cl_khr_subgroup_ballot is not supported on this device, "
                 "skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    constexpr size_t global_work_size = 170;
    constexpr size_t local_work_size = 64;
    WorkGroupParams test_params(global_work_size, local_work_size);
    test_params.save_kernel_source(sub_group_ballot_mask_source);
    test_params.save_kernel_source(sub_group_non_uniform_broadcast_source,
                                   "sub_group_non_uniform_broadcast");
    test_params.save_kernel_source(sub_group_broadcast_first_source,
                                   "sub_group_broadcast_first");
    RunTestForType rft(device, context, queue, num_elements, test_params);

    // non uniform broadcast functions
    int error = run_non_uniform_broadcast_for_type<cl_int>(rft);
    error |= run_non_uniform_broadcast_for_type<cl_int2>(rft);
    error |= run_non_uniform_broadcast_for_type<subgroups::cl_int3>(rft);
    error |= run_non_uniform_broadcast_for_type<cl_int4>(rft);
    error |= run_non_uniform_broadcast_for_type<cl_int8>(rft);
    error |= run_non_uniform_broadcast_for_type<cl_int16>(rft);

    error |= run_non_uniform_broadcast_for_type<cl_uint>(rft);
    error |= run_non_uniform_broadcast_for_type<cl_uint2>(rft);
    error |= run_non_uniform_broadcast_for_type<subgroups::cl_uint3>(rft);
    error |= run_non_uniform_broadcast_for_type<cl_uint4>(rft);
    error |= run_non_uniform_broadcast_for_type<cl_uint8>(rft);
    error |= run_non_uniform_broadcast_for_type<cl_uint16>(rft);

    error |= run_non_uniform_broadcast_for_type<cl_char>(rft);
    error |= run_non_uniform_broadcast_for_type<cl_char2>(rft);
    error |= run_non_uniform_broadcast_for_type<subgroups::cl_char3>(rft);
    error |= run_non_uniform_broadcast_for_type<cl_char4>(rft);
    error |= run_non_uniform_broadcast_for_type<cl_char8>(rft);
    error |= run_non_uniform_broadcast_for_type<cl_char16>(rft);

    error |= run_non_uniform_broadcast_for_type<cl_uchar>(rft);
    error |= run_non_uniform_broadcast_for_type<cl_uchar2>(rft);
    error |= run_non_uniform_broadcast_for_type<subgroups::cl_uchar3>(rft);
    error |= run_non_uniform_broadcast_for_type<cl_uchar4>(rft);
    error |= run_non_uniform_broadcast_for_type<cl_uchar8>(rft);
    error |= run_non_uniform_broadcast_for_type<cl_uchar16>(rft);

    error |= run_non_uniform_broadcast_for_type<cl_short>(rft);
    error |= run_non_uniform_broadcast_for_type<cl_short2>(rft);
    error |= run_non_uniform_broadcast_for_type<subgroups::cl_short3>(rft);
    error |= run_non_uniform_broadcast_for_type<cl_short4>(rft);
    error |= run_non_uniform_broadcast_for_type<cl_short8>(rft);
    error |= run_non_uniform_broadcast_for_type<cl_short16>(rft);

    error |= run_non_uniform_broadcast_for_type<cl_ushort>(rft);
    error |= run_non_uniform_broadcast_for_type<cl_ushort2>(rft);
    error |= run_non_uniform_broadcast_for_type<subgroups::cl_ushort3>(rft);
    error |= run_non_uniform_broadcast_for_type<cl_ushort4>(rft);
    error |= run_non_uniform_broadcast_for_type<cl_ushort8>(rft);
    error |= run_non_uniform_broadcast_for_type<cl_ushort16>(rft);

    error |= run_non_uniform_broadcast_for_type<cl_long>(rft);
    error |= run_non_uniform_broadcast_for_type<cl_long2>(rft);
    error |= run_non_uniform_broadcast_for_type<subgroups::cl_long3>(rft);
    error |= run_non_uniform_broadcast_for_type<cl_long4>(rft);
    error |= run_non_uniform_broadcast_for_type<cl_long8>(rft);
    error |= run_non_uniform_broadcast_for_type<cl_long16>(rft);

    error |= run_non_uniform_broadcast_for_type<cl_ulong>(rft);
    error |= run_non_uniform_broadcast_for_type<cl_ulong2>(rft);
    error |= run_non_uniform_broadcast_for_type<subgroups::cl_ulong3>(rft);
    error |= run_non_uniform_broadcast_for_type<cl_ulong4>(rft);
    error |= run_non_uniform_broadcast_for_type<cl_ulong8>(rft);
    error |= run_non_uniform_broadcast_for_type<cl_ulong16>(rft);

    error |= run_non_uniform_broadcast_for_type<cl_float>(rft);
    error |= run_non_uniform_broadcast_for_type<cl_float2>(rft);
    error |= run_non_uniform_broadcast_for_type<subgroups::cl_float3>(rft);
    error |= run_non_uniform_broadcast_for_type<cl_float4>(rft);
    error |= run_non_uniform_broadcast_for_type<cl_float8>(rft);
    error |= run_non_uniform_broadcast_for_type<cl_float16>(rft);

    error |= run_non_uniform_broadcast_for_type<cl_double>(rft);
    error |= run_non_uniform_broadcast_for_type<cl_double2>(rft);
    error |= run_non_uniform_broadcast_for_type<subgroups::cl_double3>(rft);
    error |= run_non_uniform_broadcast_for_type<cl_double4>(rft);
    error |= run_non_uniform_broadcast_for_type<cl_double8>(rft);
    error |= run_non_uniform_broadcast_for_type<cl_double16>(rft);

    error |= run_non_uniform_broadcast_for_type<subgroups::cl_half>(rft);
    error |= run_non_uniform_broadcast_for_type<subgroups::cl_half2>(rft);
    error |= run_non_uniform_broadcast_for_type<subgroups::cl_half3>(rft);
    error |= run_non_uniform_broadcast_for_type<subgroups::cl_half4>(rft);
    error |= run_non_uniform_broadcast_for_type<subgroups::cl_half8>(rft);
    error |= run_non_uniform_broadcast_for_type<subgroups::cl_half16>(rft);

    // broadcast first functions
    error |=
        rft.run_impl<cl_int, BC<cl_int, SubgroupsBroadcastOp::broadcast_first>>(
            "sub_group_broadcast_first");
    error |= rft.run_impl<cl_uint,
                          BC<cl_uint, SubgroupsBroadcastOp::broadcast_first>>(
        "sub_group_broadcast_first");
    error |= rft.run_impl<cl_long,
                          BC<cl_long, SubgroupsBroadcastOp::broadcast_first>>(
        "sub_group_broadcast_first");
    error |= rft.run_impl<cl_ulong,
                          BC<cl_ulong, SubgroupsBroadcastOp::broadcast_first>>(
        "sub_group_broadcast_first");
    error |= rft.run_impl<cl_short,
                          BC<cl_short, SubgroupsBroadcastOp::broadcast_first>>(
        "sub_group_broadcast_first");
    error |= rft.run_impl<cl_ushort,
                          BC<cl_ushort, SubgroupsBroadcastOp::broadcast_first>>(
        "sub_group_broadcast_first");
    error |= rft.run_impl<cl_char,
                          BC<cl_char, SubgroupsBroadcastOp::broadcast_first>>(
        "sub_group_broadcast_first");
    error |= rft.run_impl<cl_uchar,
                          BC<cl_uchar, SubgroupsBroadcastOp::broadcast_first>>(
        "sub_group_broadcast_first");
    error |= rft.run_impl<cl_float,
                          BC<cl_float, SubgroupsBroadcastOp::broadcast_first>>(
        "sub_group_broadcast_first");
    error |= rft.run_impl<cl_double,
                          BC<cl_double, SubgroupsBroadcastOp::broadcast_first>>(
        "sub_group_broadcast_first");
    error |= rft.run_impl<
        subgroups::cl_half,
        BC<subgroups::cl_half, SubgroupsBroadcastOp::broadcast_first>>(
        "sub_group_broadcast_first");

    // mask functions
    error |= rft.run_impl<cl_uint4, SMASK<cl_uint4, BallotOp::eq_mask>>(
        "get_sub_group_eq_mask");
    error |= rft.run_impl<cl_uint4, SMASK<cl_uint4, BallotOp::ge_mask>>(
        "get_sub_group_ge_mask");
    error |= rft.run_impl<cl_uint4, SMASK<cl_uint4, BallotOp::gt_mask>>(
        "get_sub_group_gt_mask");
    error |= rft.run_impl<cl_uint4, SMASK<cl_uint4, BallotOp::le_mask>>(
        "get_sub_group_le_mask");
    error |= rft.run_impl<cl_uint4, SMASK<cl_uint4, BallotOp::lt_mask>>(
        "get_sub_group_lt_mask");

    // sub_group_ballot function
    WorkGroupParams test_params_ballot(global_work_size, local_work_size, 3);
    test_params_ballot.save_kernel_source(sub_group_ballot_source);
    RunTestForType rft_ballot(device, context, queue, num_elements,
                              test_params_ballot);
    error |=
        rft_ballot.run_impl<cl_uint4, BALLOT<cl_uint4>>("sub_group_ballot");

    // ballot arithmetic functions
    WorkGroupParams test_params_arith(global_work_size, local_work_size);
    test_params_arith.save_kernel_source(sub_group_ballot_bit_scan_find_source);
    test_params_arith.save_kernel_source(sub_group_inverse_ballot_source,
                                         "sub_group_inverse_ballot");
    test_params_arith.save_kernel_source(sub_group_ballot_bit_extract_source,
                                         "sub_group_ballot_bit_extract");
    RunTestForType rft_arith(device, context, queue, num_elements,
                             test_params_arith);
    error |=
        rft_arith.run_impl<cl_uint4,
                           BALLOT_INVERSE<cl_uint4, BallotOp::inverse_ballot>>(
            "sub_group_inverse_ballot");
    error |= rft_arith.run_impl<
        cl_uint4, BALLOT_BIT_EXTRACT<cl_uint4, BallotOp::ballot_bit_extract>>(
        "sub_group_ballot_bit_extract");
    error |= rft_arith.run_impl<
        cl_uint4, BALLOT_COUNT_SCAN_FIND<cl_uint4, BallotOp::ballot_bit_count>>(
        "sub_group_ballot_bit_count");
    error |= rft_arith.run_impl<
        cl_uint4,
        BALLOT_COUNT_SCAN_FIND<cl_uint4, BallotOp::ballot_inclusive_scan>>(
        "sub_group_ballot_inclusive_scan");
    error |= rft_arith.run_impl<
        cl_uint4,
        BALLOT_COUNT_SCAN_FIND<cl_uint4, BallotOp::ballot_exclusive_scan>>(
        "sub_group_ballot_exclusive_scan");
    error |= rft_arith.run_impl<
        cl_uint4, BALLOT_COUNT_SCAN_FIND<cl_uint4, BallotOp::ballot_find_lsb>>(
        "sub_group_ballot_find_lsb");
    error |= rft_arith.run_impl<
        cl_uint4, BALLOT_COUNT_SCAN_FIND<cl_uint4, BallotOp::ballot_find_msb>>(
        "sub_group_ballot_find_msb");

    return error;
}
