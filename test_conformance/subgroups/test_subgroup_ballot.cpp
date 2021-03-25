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
    static void gen(Ty *x, Ty *t, cl_int *m, const WorkGroupParams &test_params)
    {
        // no work here
        int gws = test_params.global_workgroup_size;
        int lws = test_params.local_workgroup_size;
        int sbs = test_params.subgroup_size;
        int non_uniform_size = gws % lws;
        log_info("  sub_group_ballot...\n");
        if (non_uniform_size)
        {
            log_info("  non uniform work group size mode ON\n");
        }
    }

    static int chk(Ty *x, Ty *y, Ty *mx, Ty *my, cl_int *m,
                   const WorkGroupParams &test_params)
    {
        int wi_id, wg_id, sb_id;
        int gws = test_params.global_workgroup_size;
        int lws = test_params.local_workgroup_size;
        int sbs = test_params.subgroup_size;
        int sb_number = (lws + sbs - 1) / sbs;
        int current_sbs = 0;
        cl_uint expected_result, device_result;
        int non_uniform_size = gws % lws;
        int wg_number = gws / lws;
        wg_number = non_uniform_size ? wg_number + 1 : wg_number;
        int last_subgroup_size = 0;

        for (wg_id = 0; wg_id < wg_number; ++wg_id)
        { // for each work_group
            if (non_uniform_size && wg_id == wg_number - 1)
            {
                set_last_workgroup_params(non_uniform_size, sb_number, sbs, lws,
                                          last_subgroup_size);
            }

            for (wi_id = 0; wi_id < lws; ++wi_id)
            { // inside the work_group
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
                for (wi_id = 0; wi_id < current_sbs; ++wi_id)
                {
                    device_result = my[wg_offset + wi_id];
                    expected_result = 1;
                    if (!compare(device_result, expected_result))
                    {
                        log_error(
                            "ERROR: sub_group_ballot mismatch for local id "
                            "%d in sub group %d in group %d obtained {%d}, "
                            "expected {%d} \n",
                            wi_id, sb_id, wg_id, device_result,
                            expected_result);
                        return TEST_FAIL;
                    }
                }
            }
            y += lws;
            m += 4 * lws;
        }
        log_info("  sub_group_ballot... passed\n");
        return TEST_PASS;
    }
};

// Test for bit extract ballot functions
template <typename Ty, BallotOp operation> struct BALLOT_BIT_EXTRACT
{
    static void gen(Ty *x, Ty *t, cl_int *m, const WorkGroupParams &test_params)
    {
        int wi_id, sb_id, wg_id, l;
        int gws = test_params.global_workgroup_size;
        int lws = test_params.local_workgroup_size;
        int sbs = test_params.subgroup_size;
        int sb_number = (lws + sbs - 1) / sbs;
        int wg_number = gws / lws;
        int limit_sbs = sbs > 100 ? 100 : sbs;
        int non_uniform_size = gws % lws;
        log_info("  sub_group_%s(%s)...\n", operation_names(operation),
                 TypeManager<Ty>::name());

        if (non_uniform_size)
        {
            log_info("  non uniform work group size mode ON\n");
        }

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

    static int chk(Ty *x, Ty *y, Ty *mx, Ty *my, cl_int *m,
                   const WorkGroupParams &test_params)
    {
        int wi_id, wg_id, l, sb_id;
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
        log_info("  sub_group_%s(%s)... passed\n", operation_names(operation),
                 TypeManager<Ty>::name());
        return TEST_PASS;
    }
};

template <typename Ty, BallotOp operation> struct BALLOT_INVERSE
{
    static void gen(Ty *x, Ty *t, cl_int *m, const WorkGroupParams &test_params)
    {
        int gws = test_params.global_workgroup_size;
        int lws = test_params.local_workgroup_size;
        int sbs = test_params.subgroup_size;
        int non_uniform_size = gws % lws;
        log_info("  sub_group_inverse_ballot...\n");
        if (non_uniform_size)
        {
            log_info("  non uniform work group size mode ON\n");
        }
        // no work here
    }

    static int chk(Ty *x, Ty *y, Ty *mx, Ty *my, cl_int *m,
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
                // take index of array where info which work_item will
                // be broadcast its value is stored
                int midx = 4 * wg_offset + 2;
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

        log_info("  sub_group_inverse_ballot... passed\n");
        return TEST_PASS;
    }
};


// Test for bit count/inclusive and exclusive scan/ find lsb msb ballot function
template <typename Ty, BallotOp operation> struct BALLOT_COUNT_SCAN_FIND
{
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

        log_info("  sub_group_%s(%s)...\n", operation_names(operation),
                 TypeManager<Ty>::name());
        if (non_uniform_size)
        {
            log_info("  non uniform work group size mode ON\n");
            wg_number++;
        }
        int e;
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
                    log_error("Unknown operation...");
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
            for (cl_uint i = 0; i <= sub_group_local_id; ++i) mask.set(i);
            if (operation == BallotOp::ballot_exclusive_scan)
                mask.reset(sub_group_local_id);
        }
        return mask;
    }

    static int chk(Ty *x, Ty *y, Ty *mx, Ty *my, cl_int *m,
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
        cl_uint4 expected_result, device_result;
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
                expected_result = { 0, 0, 0, 0 };
                for (wi_id = 0; wi_id < current_sbs; ++wi_id)
                { // for subgroup element
                    bs128 bs;
                    // convert cl_uint4 input into std::bitset<128>
                    bs |= bs128(mx[wg_offset + wi_id].s0)
                        | (bs128(mx[wg_offset + wi_id].s1) << 32)
                        | (bs128(mx[wg_offset + wi_id].s2) << 64)
                        | (bs128(mx[wg_offset + wi_id].s3) << 96);
                    bs &= getImportantBits(wi_id, current_sbs);
                    device_result = my[wg_offset + wi_id];
                    if (operation == BallotOp::ballot_inclusive_scan
                        || operation == BallotOp::ballot_exclusive_scan
                        || operation == BallotOp::ballot_bit_count)
                    {
                        expected_result.s0 = bs.count();
                        if (!compare(device_result, expected_result))
                        {
                            log_error("ERROR: sub_group_%s "
                                      "mismatch for local id %d in sub group "
                                      "%d in group %d obtained {%d, %d, %d, "
                                      "%d}, expected {%d, %d, %d, %d}\n",
                                      operation_names(operation), wi_id, sb_id,
                                      wg_id, device_result.s0, device_result.s1,
                                      device_result.s2, device_result.s3,
                                      expected_result.s0, expected_result.s1,
                                      expected_result.s2, expected_result.s3);
                            return TEST_FAIL;
                        }
                    }
                    else if (operation == BallotOp::ballot_find_lsb)
                    {
                        for (int id = 0; id < current_sbs; ++id)
                        {
                            if (bs.test(id))
                            {
                                expected_result.s0 = id;
                                break;
                            }
                        }
                        if (!compare(device_result, expected_result))
                        {
                            log_error("ERROR: sub_group_ballot_find_lsb "
                                      "mismatch for local id %d in sub group "
                                      "%d in group %d obtained {%d, %d, %d, "
                                      "%d}, expected {%d, %d, %d, %d}\n",
                                      wi_id, sb_id, wg_id, device_result.s0,
                                      device_result.s1, device_result.s2,
                                      device_result.s3, expected_result.s0,
                                      expected_result.s1, expected_result.s2,
                                      expected_result.s3);
                            return TEST_FAIL;
                        }
                    }
                    else if (operation == BallotOp::ballot_find_msb)
                    {
                        for (int id = current_sbs - 1; id >= 0; --id)
                        {
                            if (bs.test(id))
                            {
                                expected_result.s0 = id;
                                break;
                            }
                        }
                        if (!compare(device_result, expected_result))
                        {
                            log_error("ERROR: sub_group_ballot_find_msb "
                                      "mismatch for local id %d in sub group "
                                      "%d in group %d obtained {%d, %d, %d, "
                                      "%d}, expected {%d, %d, %d, %d}\n",
                                      wi_id, sb_id, wg_id, device_result.s0,
                                      device_result.s1, device_result.s2,
                                      device_result.s3, expected_result.s0,
                                      expected_result.s1, expected_result.s2,
                                      expected_result.s3);
                            return TEST_FAIL;
                        }
                    }
                }
            }
            x += lws;
            y += lws;
            m += 4 * lws;
        }
        log_info("  sub_group_ballot_%s(%s)... passed\n",
                 operation_names(operation), TypeManager<Ty>::name());
        return TEST_PASS;
    }
};

// test mask functions
template <typename Ty, BallotOp operation> struct SMASK
{
    static void gen(Ty *x, Ty *t, cl_int *m, const WorkGroupParams &test_params)
    {
        int wi_id, wg_id, l, sb_id;
        int gws = test_params.global_workgroup_size;
        int lws = test_params.local_workgroup_size;
        int sbs = test_params.subgroup_size;
        int sb_number = (lws + sbs - 1) / sbs;
        int wg_number = gws / lws;
        log_info("  get_sub_group_%s_mask...\n", operation_names(operation));
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

    static int chk(Ty *x, Ty *y, Ty *mx, Ty *my, cl_int *m,
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
        log_info("  get_sub_group_%s_mask... passed\n",
                 operation_names(operation));
        return TEST_PASS;
    }
};

static const char *bcast_non_uniform_source =
    "__kernel void test_bcast_non_uniform(const __global Type *in, __global "
    "int4 *xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    "    Type x = in[gid];\n"
    "    if (xy[gid].x < NR_OF_ACTIVE_WORK_ITEMS) {\n"
    "        out[gid] = sub_group_non_uniform_broadcast(x, xy[gid].z);\n"
    "    } else {\n"
    "       out[gid] = sub_group_non_uniform_broadcast(x, xy[gid].w);\n"
    "    }\n"
    "}\n";

static const char *bcast_first_source =
    "__kernel void test_bcast_first(const __global Type *in, __global int4 "
    "*xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    "    Type x = in[gid];\n"
    "    if (xy[gid].x < NR_OF_ACTIVE_WORK_ITEMS) {\n"
    "       out[gid] = sub_group_broadcast_first(x);\n"
    "    } else {\n"
    "       out[gid] = sub_group_broadcast_first(x);\n"
    "    }\n"
    "}\n";

static const char *ballot_bit_count_source =
    "__kernel void test_sub_group_ballot_bit_count(const __global Type *in, "
    "__global int4 *xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    "    Type x = in[gid];\n"
    "    uint4 value = (uint4)(0,0,0,0);\n"
    "    value = (uint4)(sub_group_ballot_bit_count(x),0,0,0);\n"
    "    out[gid] = value;\n"
    "}\n";

static const char *ballot_inclusive_scan_source =
    "__kernel void test_sub_group_ballot_inclusive_scan(const __global Type "
    "*in, __global int4 *xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    "    Type x = in[gid];\n"
    "    uint4 value = (uint4)(0,0,0,0);\n"
    "    value = (uint4)(sub_group_ballot_inclusive_scan(x),0,0,0);\n"
    "    out[gid] = value;\n"
    "}\n";

static const char *ballot_exclusive_scan_source =
    "__kernel void test_sub_group_ballot_exclusive_scan(const __global Type "
    "*in, __global int4 *xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    "    Type x = in[gid];\n"
    "    uint4 value = (uint4)(0,0,0,0);\n"
    "    value = (uint4)(sub_group_ballot_exclusive_scan(x),0,0,0);\n"
    "    out[gid] = value;\n"
    "}\n";

static const char *ballot_find_lsb_source =
    "__kernel void test_sub_group_ballot_find_lsb(const __global Type *in, "
    "__global int4 *xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    "    Type x = in[gid];\n"
    "    uint4 value = (uint4)(0,0,0,0);\n"
    "    value = (uint4)(sub_group_ballot_find_lsb(x),0,0,0);\n"
    "    out[gid] = value;\n"
    "}\n";

static const char *ballot_find_msb_source =
    "__kernel void test_sub_group_ballot_find_msb(const __global Type *in, "
    "__global int4 *xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    "    Type x = in[gid];\n"
    "    uint4 value = (uint4)(0,0,0,0);"
    "    value = (uint4)(sub_group_ballot_find_msb(x),0,0,0);"
    "    out[gid] = value ;"
    "}\n";

static const char *get_subgroup_ge_mask_source =
    "__kernel void test_get_sub_group_ge_mask(const __global Type *in, "
    "__global int4 *xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    "    xy[gid].z = get_max_sub_group_size();\n"
    "    Type x = in[gid];\n"
    "    uint4 mask = get_sub_group_ge_mask();"
    "    out[gid] = mask;\n"
    "}\n";

static const char *get_subgroup_gt_mask_source =
    "__kernel void test_get_sub_group_gt_mask(const __global Type *in, "
    "__global int4 *xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    "    xy[gid].z = get_max_sub_group_size();\n"
    "    Type x = in[gid];\n"
    "    uint4 mask = get_sub_group_gt_mask();"
    "    out[gid] = mask;\n"
    "}\n";

static const char *get_subgroup_le_mask_source =
    "__kernel void test_get_sub_group_le_mask(const __global Type *in, "
    "__global int4 *xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    "    xy[gid].z = get_max_sub_group_size();\n"
    "    Type x = in[gid];\n"
    "    uint4 mask = get_sub_group_le_mask();"
    "    out[gid] = mask;\n"
    "}\n";

static const char *get_subgroup_lt_mask_source =
    "__kernel void test_get_sub_group_lt_mask(const __global Type *in, "
    "__global int4 *xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    "    xy[gid].z = get_max_sub_group_size();\n"
    "    Type x = in[gid];\n"
    "    uint4 mask = get_sub_group_lt_mask();"
    "    out[gid] = mask;\n"
    "}\n";

static const char *get_subgroup_eq_mask_source =
    "__kernel void test_get_sub_group_eq_mask(const __global Type *in, "
    "__global int4 *xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    "    xy[gid].z = get_max_sub_group_size();\n"
    "    Type x = in[gid];\n"
    "    uint4 mask = get_sub_group_eq_mask();"
    "    out[gid] = mask;\n"
    "}\n";

static const char *ballot_source =
    "__kernel void test_sub_group_ballot(const __global Type *in, "
    "__global int4 *xy, __global Type *out)\n"
    "{\n"
    "uint4 full_ballot = sub_group_ballot(1);\n"
    "uint divergence_mask;\n"
    "uint4 partial_ballot;\n"
    "uint gid = get_global_id(0);"
    "XY(xy,gid);\n"
    "if (get_sub_group_local_id() & 1) {\n"
    "    divergence_mask = 0xaaaaaaaa;\n"
    "    partial_ballot = sub_group_ballot(1);\n"
    "} else {\n"
    "    divergence_mask = 0x55555555;\n"
    "    partial_ballot = sub_group_ballot(1);\n"
    "}\n"
    " size_t lws = get_local_size(0);\n"
    "uint4 masked_ballot = full_ballot;\n"
    "masked_ballot.x &= divergence_mask;\n"
    "masked_ballot.y &= divergence_mask;\n"
    "masked_ballot.z &= divergence_mask;\n"
    "masked_ballot.w &= divergence_mask;\n"
    "out[gid] = all(masked_ballot == partial_ballot);\n"

    "} \n";

static const char *ballot_source_inverse =
    "__kernel void test_sub_group_ballot_inverse(const __global "
    "Type *in, "
    "__global int4 *xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    "    Type x = in[gid];\n"
    "    uint4 value = (uint4)(10,0,0,0);\n"
    "    if (get_sub_group_local_id() & 1) {"
    "        uint4 partial_ballot_mask = "
    "(uint4)(0xAAAAAAAA,0xAAAAAAAA,0xAAAAAAAA,0xAAAAAAAA);"
    "        if (sub_group_inverse_ballot(partial_ballot_mask)) {\n"
    "            value = (uint4)(1,0,0,1);\n"
    "        } else {\n"
    "            value = (uint4)(0,0,0,1);\n"
    "        }\n"
    "    } else {\n"
    "       uint4 partial_ballot_mask = "
    "(uint4)(0x55555555,0x55555555,0x55555555,0x55555555);"
    "        if (sub_group_inverse_ballot(partial_ballot_mask)) {\n"
    "            value = (uint4)(1,0,0,2);\n"
    "        } else {\n"
    "            value = (uint4)(0,0,0,2);\n"
    "        }\n"
    "    }\n"
    "    out[gid] = value;\n"
    "}\n";

static const char *ballot_bit_extract_source =
    "__kernel void test_sub_group_ballot_bit_extract(const __global Type *in, "
    "__global int4 *xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    "    Type x = in[gid];\n"
    "    uint index = xy[gid].z;\n"
    "    uint4 value = (uint4)(10,0,0,0);\n"
    "    if (get_sub_group_local_id() & 1) {"
    "       if (sub_group_ballot_bit_extract(x, xy[gid].z)) {\n"
    "           value = (uint4)(1,0,0,1);\n"
    "       } else {\n"
    "           value = (uint4)(0,0,0,1);\n"
    "       }\n"
    "    } else {\n"
    "       if (sub_group_ballot_bit_extract(x, xy[gid].w)) {\n"
    "           value = (uint4)(1,0,0,2);\n"
    "       } else {\n"
    "           value = (uint4)(0,0,0,2);\n"
    "       }\n"
    "    }\n"
    "    out[gid] = value;\n"
    "}\n";

template <typename T> int run_non_uniform_broadcast_for_type(RunTestForType rft)
{
    int error =
        rft.run_impl<T, BC<T, SubgroupsBroadcastOp::non_uniform_broadcast>>(
            "test_bcast_non_uniform", bcast_non_uniform_source);
    return error;
}


}

int test_subgroup_functions_ballot(cl_device_id device, cl_context context,
                                   cl_command_queue queue, int num_elements)
{
    std::vector<std::string> required_extensions = { "cl_khr_subgroup_ballot" };
    constexpr size_t global_work_size = 170;
    constexpr size_t local_work_size = 64;
    WorkGroupParams test_params(global_work_size, local_work_size,
                                required_extensions);
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
            "test_bcast_first", bcast_first_source);
    error |= rft.run_impl<cl_uint,
                          BC<cl_uint, SubgroupsBroadcastOp::broadcast_first>>(
        "test_bcast_first", bcast_first_source);
    error |= rft.run_impl<cl_long,
                          BC<cl_long, SubgroupsBroadcastOp::broadcast_first>>(
        "test_bcast_first", bcast_first_source);
    error |= rft.run_impl<cl_ulong,
                          BC<cl_ulong, SubgroupsBroadcastOp::broadcast_first>>(
        "test_bcast_first", bcast_first_source);
    error |= rft.run_impl<cl_short,
                          BC<cl_short, SubgroupsBroadcastOp::broadcast_first>>(
        "test_bcast_first", bcast_first_source);
    error |= rft.run_impl<cl_ushort,
                          BC<cl_ushort, SubgroupsBroadcastOp::broadcast_first>>(
        "test_bcast_first", bcast_first_source);
    error |= rft.run_impl<cl_char,
                          BC<cl_char, SubgroupsBroadcastOp::broadcast_first>>(
        "test_bcast_first", bcast_first_source);
    error |= rft.run_impl<cl_uchar,
                          BC<cl_uchar, SubgroupsBroadcastOp::broadcast_first>>(
        "test_bcast_first", bcast_first_source);
    error |= rft.run_impl<cl_float,
                          BC<cl_float, SubgroupsBroadcastOp::broadcast_first>>(
        "test_bcast_first", bcast_first_source);
    error |= rft.run_impl<cl_double,
                          BC<cl_double, SubgroupsBroadcastOp::broadcast_first>>(
        "test_bcast_first", bcast_first_source);
    error |= rft.run_impl<
        subgroups::cl_half,
        BC<subgroups::cl_half, SubgroupsBroadcastOp::broadcast_first>>(
        "test_bcast_first", bcast_first_source);

    // mask functions
    error |= rft.run_impl<cl_uint4, SMASK<cl_uint4, BallotOp::eq_mask>>(
        "test_get_sub_group_eq_mask", get_subgroup_eq_mask_source);
    error |= rft.run_impl<cl_uint4, SMASK<cl_uint4, BallotOp::ge_mask>>(
        "test_get_sub_group_ge_mask", get_subgroup_ge_mask_source);
    error |= rft.run_impl<cl_uint4, SMASK<cl_uint4, BallotOp::gt_mask>>(
        "test_get_sub_group_gt_mask", get_subgroup_gt_mask_source);
    error |= rft.run_impl<cl_uint4, SMASK<cl_uint4, BallotOp::le_mask>>(
        "test_get_sub_group_le_mask", get_subgroup_le_mask_source);
    error |= rft.run_impl<cl_uint4, SMASK<cl_uint4, BallotOp::lt_mask>>(
        "test_get_sub_group_lt_mask", get_subgroup_lt_mask_source);

    // ballot functions
    error |= rft.run_impl<cl_uint, BALLOT<cl_uint>>("test_sub_group_ballot",
                                                    ballot_source);
    error |= rft.run_impl<cl_uint4,
                          BALLOT_INVERSE<cl_uint4, BallotOp::inverse_ballot>>(
        "test_sub_group_ballot_inverse", ballot_source_inverse);
    error |= rft.run_impl<
        cl_uint4, BALLOT_BIT_EXTRACT<cl_uint4, BallotOp::ballot_bit_extract>>(
        "test_sub_group_ballot_bit_extract", ballot_bit_extract_source);
    error |= rft.run_impl<
        cl_uint4, BALLOT_COUNT_SCAN_FIND<cl_uint4, BallotOp::ballot_bit_count>>(
        "test_sub_group_ballot_bit_count", ballot_bit_count_source);
    error |= rft.run_impl<
        cl_uint4,
        BALLOT_COUNT_SCAN_FIND<cl_uint4, BallotOp::ballot_inclusive_scan>>(
        "test_sub_group_ballot_inclusive_scan", ballot_inclusive_scan_source);
    error |= rft.run_impl<
        cl_uint4,
        BALLOT_COUNT_SCAN_FIND<cl_uint4, BallotOp::ballot_exclusive_scan>>(
        "test_sub_group_ballot_exclusive_scan", ballot_exclusive_scan_source);
    error |= rft.run_impl<
        cl_uint4, BALLOT_COUNT_SCAN_FIND<cl_uint4, BallotOp::ballot_find_lsb>>(
        "test_sub_group_ballot_find_lsb", ballot_find_lsb_source);
    error |= rft.run_impl<
        cl_uint4, BALLOT_COUNT_SCAN_FIND<cl_uint4, BallotOp::ballot_find_msb>>(
        "test_sub_group_ballot_find_msb", ballot_find_msb_source);
    return error;
}
