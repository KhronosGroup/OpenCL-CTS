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
#include "workgroup_common_templates.h"
#include "harness/typeWrappers.h"
#include <bitset>

// Global/local work group sizes
// Adjust these individually below if desired/needed
#define GWS 2000
#define LWS 200

#define GWS_NON_UNIFORM 170
#define LWS_NON_UNIFORM 64

namespace {
// Test for ballot functions
template <typename Ty> struct BALLOT
{
    static void gen(Ty *x, Ty *t, cl_int *m, int sbs, int lws, int gws)
    {
        // no work here
        int non_uniform_size = gws % lws;
        log_info("  sub_group_ballot...\n");
        if (non_uniform_size)
        {
            log_info("  non uniform work group size mode ON\n");
        }
    }

    static int chk(Ty *x, Ty *y, Ty *mx, Ty *my, cl_int *m, int sbs, int lws,
                   int gws)
    {
        int wi_id, wg_id, sb_id;
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
                set_last_worgroup_params(non_uniform_size, sb_number, sbs, lws,
                                         last_subgroup_size);
            }

            for (wi_id = 0; wi_id < lws; ++wi_id)
            { // inside the work_group
                int offset = m[4 * wi_id + 1] * sbs + m[4 * wi_id];
                // read device outputs for work_group
                my[wi_id] = y[offset];
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
    static void gen(Ty *x, Ty *t, cl_int *m, int sbs, int lws, int gws)
    {
        int wi_id, sb_id, wg_id, l;
        int sb_number = (lws + sbs - 1) / sbs;
        int wg_number = gws / lws;
        int limit_sbs = sbs > 100 ? 100 : sbs;
        log_info("  sub_group_%s(%s)...\n", operation_names(operation),
                 TypeManager<Ty>::name());

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
                int randomize_data = (int)(genrand_int32(gMTdata) % 3);
                // Initialize data matrix indexed by local id and sub group id
                switch (randomize_data)
                {
                    case 0:
                        memset(&t[wg_offset], 0, current_sbs * sizeof(Ty));
                        break;
                    case 1:
                        memset(&t[wg_offset], 0, current_sbs * sizeof(Ty));
                        wi_id = (int)(genrand_int32(gMTdata)
                                      % (cl_uint)current_sbs);
                        set_value(t[wg_offset + wi_id], 41);
                        break;
                    case 2:
                        memset(&t[wg_offset], 0xff, current_sbs * sizeof(Ty));
                        break;
                }
            }

            // Now map into work group using map from device
            for (wi_id = 0; wi_id < lws; ++wi_id)
            {
                int offset = m[4 * wi_id + 1] * sbs + m[4 * wi_id];
                x[wi_id] = t[offset];
            }

            x += lws;
            m += 4 * lws;
        }
    }

    static int chk(Ty *x, Ty *y, Ty *mx, Ty *my, cl_int *m, int sbs, int lws,
                   int gws)
    {
        int wi_id, wg_id, l, sb_id;
        int sb_number = (lws + sbs - 1) / sbs;
        int wg_number = gws / lws;
        cl_uint4 expected_result, device_result;

        for (wg_id = 0; wg_id < wg_number; ++wg_id)
        { // for each work_group
            // Map to array indexed to array indexed by local ID and sub group
            for (wi_id = 0; wi_id < lws; ++wi_id)
            { // inside the work_group
                int offset = m[4 * wi_id + 1] * sbs + m[4 * wi_id];
                // read host inputs for work_group
                mx[wi_id] = x[offset];
                // read device outputs for work_group
                my[wi_id] = y[offset];
            }

            for (sb_id = 0; sb_id < sb_number; ++sb_id)
            { // for each subgroup
                int wg_offset = sb_id * sbs;
                int current_sbs = wg_offset + sbs > lws ? lws - wg_offset : sbs;
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
    static void gen(Ty *x, Ty *t, cl_int *m, int sbs, int lws, int gws)
    {
        int non_uniform_size = gws % lws;
        log_info("  sub_group_inverse_ballot...\n");
        if (non_uniform_size)
        {
            log_info("  non uniform work group size mode ON\n");
        }
        // no work here
    }

    static int chk(Ty *x, Ty *y, Ty *mx, Ty *my, cl_int *m, int sbs, int lws,
                   int gws)
    {
        int wi_id, wg_id, sb_id;
        int sb_number = (lws + sbs - 1) / sbs;
        cl_uint4 expected_result, device_result;
        int non_uniform_size = gws % lws;
        int wg_number = gws / lws;
        if (non_uniform_size) wg_number++;

        for (wg_id = 0; wg_id < wg_number; ++wg_id)
        { // for each work_group
            // Map to array indexed to array indexed by local ID and sub group
            for (wi_id = 0; wi_id < lws; ++wi_id)
            { // inside the work_group
                int offset = m[4 * wi_id + 1] * sbs + m[4 * wi_id];
                mx[wi_id] = x[offset]; // read host inputs for work_group
                my[wi_id] = y[offset]; // read device outputs for work_group
            }

            for (sb_id = 0; sb_id < sb_number; ++sb_id)
            { // for each subgroup
                int wg_offset = sb_id * sbs;
                int current_sbs = wg_offset + sbs > lws ? lws - wg_offset : sbs;
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
    static void gen(Ty *x, Ty *t, cl_int *m, int sbs, int lws, int gws)
    {
        int wi_id, wg_id, sb_id;
        int sb_number = (lws + sbs - 1) / sbs;
        int non_uniform_size = gws % lws;
        int wg_number = gws / lws;

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
            for (sb_id = 0; sb_id < sb_number; ++sb_id)
            { // for each subgroup
                int wg_offset = sb_id * sbs;
                int current_sbs = wg_offset + sbs > lws ? lws - wg_offset : sbs;
                log_info("wg_offset = %d current_sbs = %d sbs = %d\n",
                         wg_offset, current_sbs, sbs);
                if (operation == BallotOp::ballot_bit_count
                    || operation == BallotOp::ballot_inclusive_scan
                    || operation == BallotOp::ballot_exclusive_scan)
                {
                    // Initialize data matrix indexed by local id and sub group
                    // id
                    int randomize_data = (int)(genrand_int32(gMTdata) % 3);
                    switch (randomize_data)
                    {
                        case 0:
                            memset(&t[wg_offset], 0, current_sbs * sizeof(Ty));
                            break;
                        case 1:
                            memset(&t[wg_offset], 0, current_sbs * sizeof(Ty));
                            wi_id = (int)(genrand_int32(gMTdata)
                                          % (cl_uint)current_sbs);
                            set_value(t[wg_offset + wi_id], 41);
                            break;
                        case 2:
                            memset(&t[wg_offset], 0xff,
                                   current_sbs * sizeof(Ty));
                            break;
                    }
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
                int offset = m[4 * wi_id + 1] * sbs + m[4 * wi_id];
                x[wi_id] = t[offset];
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

    static int chk(Ty *x, Ty *y, Ty *mx, Ty *my, cl_int *m, int sbs, int lws,
                   int gws)
    {
        int wi_id, wg_id, sb_id;
        int sb_number = (lws + sbs - 1) / sbs;
        int non_uniform_size = gws % lws;
        int wg_number = gws / lws;
        wg_number = non_uniform_size ? wg_number + 1 : wg_number;
        cl_uint4 expected_result, device_result;

        for (wg_id = 0; wg_id < wg_number; ++wg_id)
        { // for each work_group
            // Map to array indexed to array indexed by local ID and sub group
            for (wi_id = 0; wi_id < lws; ++wi_id)
            { // inside the work_group
                int offset = m[4 * wi_id + 1] * sbs + m[4 * wi_id];
                // read host inputs for work_group
                mx[wi_id] = x[offset];
                // read device outputs for work_group
                my[wi_id] = y[offset];
            }

            for (sb_id = 0; sb_id < sb_number; ++sb_id)
            { // for each subgroup
                int wg_offset = sb_id * sbs;
                int current_sbs = wg_offset + sbs > lws ? lws - wg_offset : sbs;
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
    static void gen(Ty *x, Ty *t, cl_int *m, int sbs, int lws, int gws)
    {
        int wi_id, wg_id, l, sb_id;
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
                int offset = m[4 * wi_id + 1] * sbs + m[4 * wi_id];
                x[wi_id] = t[offset];
            }
            x += lws;
            m += 4 * lws;
        }
    }

    static int chk(Ty *x, Ty *y, Ty *mx, Ty *my, cl_int *m, int sbs, int lws,
                   int gws)
    {
        int wi_id, wg_id, sb_id;
        int sb_number = (lws + sbs - 1) / sbs;
        Ty expected_result, device_result;
        int wg_number = gws / lws;

        for (wg_id = 0; wg_id < wg_number; ++wg_id)
        { // for each work_group
            for (wi_id = 0; wi_id < lws; ++wi_id)
            { // inside the work_group
                int offset = m[4 * wi_id + 1] * sbs + m[4 * wi_id];
                mx[wi_id] = x[offset]; // read host inputs for work_group
                my[wi_id] = y[offset]; // read device outputs for work_group
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


    template <typename T> int run_nu_bc()
    {
        int error =
            test<T, BC<T, SubgroupsBroadcastOp::non_uniform_broadcast>,
                 GWS_NON_UNIFORM,
                 LWS_NON_UNIFORM>::run(device_, context_, queue_, num_elements_,
                                       "test_bcast_non_uniform",
                                       bcast_non_uniform_source, 0,
                                       useCoreSubgroups_, required_extensions_);
        return error;
    }

    template <typename T> int run_bc_first()
    {
        int error =
            test<T, BC<T, SubgroupsBroadcastOp::broadcast_first>, 64, 32>::run(
                device_, context_, queue_, num_elements_, "test_bcast_first",
                bcast_first_source, 0, useCoreSubgroups_, required_extensions_);
        return error;
    }


    int run_smask()
    {
        int error =
            test<cl_uint4, SMASK<cl_uint4, BallotOp::eq_mask>, GWS, LWS>::run(
                device_, context_, queue_, num_elements_,
                "test_get_sub_group_eq_mask", get_subgroup_eq_mask_source, 0,
                useCoreSubgroups_, required_extensions_);
        error |=
            test<cl_uint4, SMASK<cl_uint4, BallotOp::ge_mask>, GWS, LWS>::run(
                device_, context_, queue_, num_elements_,
                "test_get_sub_group_ge_mask", get_subgroup_ge_mask_source, 0,
                useCoreSubgroups_, required_extensions_);
        error |=
            test<cl_uint4, SMASK<cl_uint4, BallotOp::gt_mask>, GWS, LWS>::run(
                device_, context_, queue_, num_elements_,
                "test_get_sub_group_gt_mask", get_subgroup_gt_mask_source, 0,
                useCoreSubgroups_, required_extensions_);
        error |=
            test<cl_uint4, SMASK<cl_uint4, BallotOp::le_mask>, GWS, LWS>::run(
                device_, context_, queue_, num_elements_,
                "test_get_sub_group_le_mask", get_subgroup_le_mask_source, 0,
                useCoreSubgroups_, required_extensions_);
        error |=
            test<cl_uint4, SMASK<cl_uint4, BallotOp::lt_mask>, GWS, LWS>::run(
                device_, context_, queue_, num_elements_,
                "test_get_sub_group_lt_mask", get_subgroup_lt_mask_source, 0,
                useCoreSubgroups_, required_extensions_);
        return error;
    }

    int run_ballot()
    {
        int error =
            test<cl_uint, BALLOT<cl_uint>, GWS_NON_UNIFORM,
                 LWS_NON_UNIFORM>::run(device_, context_, queue_, num_elements_,
                                       "test_sub_group_ballot", ballot_source,
                                       0, useCoreSubgroups_,
                                       required_extensions_);

        error |=
            test<cl_uint4, BALLOT_INVERSE<cl_uint4, BallotOp::inverse_ballot>,
                 GWS, LWS>::run(device_, context_, queue_, num_elements_,
                                "test_sub_group_ballot_inverse",
                                ballot_source_inverse, 0, useCoreSubgroups_,
                                required_extensions_);
        error |=
            test<cl_uint4,
                 BALLOT_BIT_EXTRACT<cl_uint4, BallotOp::ballot_bit_extract>,
                 GWS, LWS>::run(device_, context_, queue_, num_elements_,
                                "test_sub_group_ballot_bit_extract",
                                ballot_bit_extract_source, 0, useCoreSubgroups_,
                                required_extensions_);

        error |=
            test<cl_uint4,
                 BALLOT_COUNT_SCAN_FIND<cl_uint4, BallotOp::ballot_bit_count>,
                 GWS, LWS>::run(device_, context_, queue_, num_elements_,
                                "test_sub_group_ballot_bit_count",
                                ballot_bit_count_source, 0, useCoreSubgroups_,
                                required_extensions_);
        error |= test<
            cl_uint4,
            BALLOT_COUNT_SCAN_FIND<cl_uint4, BallotOp::ballot_inclusive_scan>,
            GWS, LWS>::run(device_, context_, queue_, num_elements_,
                           "test_sub_group_ballot_inclusive_scan",
                           ballot_inclusive_scan_source, 0, useCoreSubgroups_,
                           required_extensions_);
        error |= test<
            cl_uint4,
            BALLOT_COUNT_SCAN_FIND<cl_uint4, BallotOp::ballot_exclusive_scan>,
            GWS, LWS>::run(device_, context_, queue_, num_elements_,
                           "test_sub_group_ballot_exclusive_scan",
                           ballot_exclusive_scan_source, 0, useCoreSubgroups_,
                           required_extensions_);
        error |=
            test<cl_uint4,
                 BALLOT_COUNT_SCAN_FIND<cl_uint4, BallotOp::ballot_find_lsb>,
                 GWS, LWS>::run(device_, context_, queue_, num_elements_,
                                "test_sub_group_ballot_find_lsb",
                                ballot_find_lsb_source, 0, useCoreSubgroups_,
                                required_extensions_);
        error |=
            test<cl_uint4,
                 BALLOT_COUNT_SCAN_FIND<cl_uint4, BallotOp::ballot_find_msb>,
                 GWS, LWS>::run(device_, context_, queue_, num_elements_,
                                "test_sub_group_ballot_find_msb",
                                ballot_find_msb_source, 0, useCoreSubgroups_,
                                required_extensions_);
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

int test_work_group_functions_ballot(cl_device_id device, cl_context context,
                                     cl_command_queue queue, int num_elements)
{
    std::vector<std::string> required_extensions = { "cl_khr_subgroup_ballot" };
    run_for_type rft(device, context, queue, num_elements, true,
                     required_extensions);

    int error = rft.run_nu_bc<cl_int>();
    error |= rft.run_nu_bc<cl_int2>();
    error |= rft.run_nu_bc<subgroups::cl_int3>();
    error |= rft.run_nu_bc<cl_int4>();
    error |= rft.run_nu_bc<cl_int8>();
    error |= rft.run_nu_bc<cl_int16>();

    error |= rft.run_nu_bc<cl_uint>();
    error |= rft.run_nu_bc<cl_uint2>();
    error |= rft.run_nu_bc<subgroups::cl_uint3>();
    error |= rft.run_nu_bc<cl_uint4>();
    error |= rft.run_nu_bc<cl_uint8>();
    error |= rft.run_nu_bc<cl_uint16>();

    error |= rft.run_nu_bc<cl_long>();
    error |= rft.run_nu_bc<cl_long2>();
    error |= rft.run_nu_bc<subgroups::cl_long3>();
    error |= rft.run_nu_bc<cl_long4>();
    error |= rft.run_nu_bc<cl_long8>();
    error |= rft.run_nu_bc<cl_long16>();

    error |= rft.run_nu_bc<cl_ulong>();
    error |= rft.run_nu_bc<cl_ulong2>();
    error |= rft.run_nu_bc<subgroups::cl_ulong3>();
    error |= rft.run_nu_bc<cl_ulong4>();
    error |= rft.run_nu_bc<cl_ulong8>();
    error |= rft.run_nu_bc<cl_ulong16>();

    error |= rft.run_nu_bc<cl_short>();
    error |= rft.run_nu_bc<cl_short2>();
    error |= rft.run_nu_bc<subgroups::cl_short3>();
    error |= rft.run_nu_bc<cl_short4>();
    error |= rft.run_nu_bc<cl_short8>();
    error |= rft.run_nu_bc<cl_short16>();

    error |= rft.run_nu_bc<cl_ushort>();
    error |= rft.run_nu_bc<cl_ushort2>();
    error |= rft.run_nu_bc<subgroups::cl_ushort3>();
    error |= rft.run_nu_bc<cl_ushort4>();
    error |= rft.run_nu_bc<cl_ushort8>();
    error |= rft.run_nu_bc<cl_ushort16>();

    error |= rft.run_nu_bc<cl_char>();
    error |= rft.run_nu_bc<cl_char2>();
    error |= rft.run_nu_bc<subgroups::cl_char3>();
    error |= rft.run_nu_bc<cl_char4>();
    error |= rft.run_nu_bc<cl_char8>();
    error |= rft.run_nu_bc<cl_char16>();

    error |= rft.run_nu_bc<cl_uchar>();
    error |= rft.run_nu_bc<cl_uchar2>();
    error |= rft.run_nu_bc<subgroups::cl_uchar3>();
    error |= rft.run_nu_bc<cl_uchar4>();
    error |= rft.run_nu_bc<cl_uchar8>();
    error |= rft.run_nu_bc<cl_uchar16>();

    error |= rft.run_nu_bc<cl_float>();
    error |= rft.run_nu_bc<cl_float2>();
    error |= rft.run_nu_bc<subgroups::cl_float3>();
    error |= rft.run_nu_bc<cl_float4>();
    error |= rft.run_nu_bc<cl_float8>();
    error |= rft.run_nu_bc<cl_float16>();

    error |= rft.run_nu_bc<cl_double>();
    error |= rft.run_nu_bc<cl_double2>();
    error |= rft.run_nu_bc<subgroups::cl_double3>();
    error |= rft.run_nu_bc<cl_double4>();
    error |= rft.run_nu_bc<cl_double8>();
    error |= rft.run_nu_bc<cl_double16>();

    error |= rft.run_nu_bc<subgroups::cl_half>();
    error |= rft.run_nu_bc<subgroups::cl_half2>();
    error |= rft.run_nu_bc<subgroups::cl_float3>();
    error |= rft.run_nu_bc<subgroups::cl_half4>();
    error |= rft.run_nu_bc<subgroups::cl_half8>();
    error |= rft.run_nu_bc<subgroups::cl_half16>();

    error |= rft.run_bc_first<cl_int>();
    error |= rft.run_bc_first<cl_uint>();
    error |= rft.run_bc_first<cl_long>();
    error |= rft.run_bc_first<cl_ulong>();
    error |= rft.run_bc_first<cl_short>();
    error |= rft.run_bc_first<cl_ushort>();
    error |= rft.run_bc_first<cl_char>();
    error |= rft.run_bc_first<cl_uchar>();
    error |= rft.run_bc_first<cl_float>();
    error |= rft.run_bc_first<cl_double>();
    error |= rft.run_bc_first<subgroups::cl_half>();
    error |= rft.run_smask();
    error |= rft.run_ballot();
    return error;
}
