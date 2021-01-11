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

namespace {
// Test for ballot functions
template <typename Ty> struct BALLOT
{
    static void gen(Ty *x, Ty *t, cl_int *m, int ns, int nw, int ng)
    {
        int i, ii, j, k, n;
        int nj = (nw + ns - 1) / ns;
        int e;
        ii = 0;
        for (k = 0; k < ng; ++k)
        { // for each work_group
            for (j = 0; j < nj; ++j)
            { // for each subgroup
                ii = j * ns;
                n = ii + ns > nw ? nw - ii : ns;
                e = (int)(genrand_int32(gMTdata) % 3);

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
        cl_uint4 tr, rr;

        log_info("  sub_group_ballot...\n");

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

                // Check result
                tr = { 0, 0, 0, 0 };
                bs128 bs;
                for (i = 0; i < n; ++i)
                { // for each subgroup
                    (mx[ii + i].s0 != 0) ? bs.set(i) : bs.reset(i);
                }
                // convert bs to uint4
                auto const uint_mask = bs128{ static_cast<unsigned long>(-1) };
                tr.s0 = (bs & uint_mask).to_ulong();
                bs >>= 32;
                tr.s1 = (bs & uint_mask).to_ulong();
                bs >>= 32;
                tr.s2 = (bs & uint_mask).to_ulong();
                bs >>= 32;
                tr.s3 = (bs & uint_mask).to_ulong();
                rr = my[ii];
                if (!compare(rr, tr))
                {
                    log_error("ERROR: sub_group_ballot mismatch for local id "
                              "%d in sub group %d in group %d obtained {%d, "
                              "%d, %d, %d}, expected {%d, %d, %d, %d}\n",
                              i, j, k, rr.s0, rr.s1, rr.s2, rr.s3, tr.s0, tr.s1,
                              tr.s2, tr.s3);
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

// Test for inverse/bit extract ballot functions
template <typename Ty, BallotOp operation> struct BALLOT2
{
    static void gen(Ty *x, Ty *t, cl_int *m, int ns, int nw, int ng)
    {
        int i, ii, j, k, n, l;
        int nj = (nw + ns - 1) / ns;
        int e;
        ii = 0;
        int d = ns > 100 ? 100 : ns;
        for (k = 0; k < ng; ++k)
        { // for each work_group
            for (j = 0; j < nj; ++j)
            { // for each subgroup
                ii = j * ns;
                n = ii + ns > nw ? nw - ii : ns;
                l = (int)(genrand_int32(gMTdata) & 0x7fffffff)
                    % (d > n ? n : d); // rand index to bit extract
                e = (int)(genrand_int32(gMTdata) % 3);
                for (i = 0; i < n; ++i)
                {
                    int midx = 4 * ii + 4 * i
                        + 2; // index of the third element int the vector.
                    m[midx] = (cl_int)
                        l; // storing information about index to bit extract
                }
                // Initialize data matrix indexed by local id and sub group id
                switch (e)
                {
                    case 0: memset(&t[ii], 0, n * sizeof(Ty)); break;
                    case 1:
                        // inverse ballot requires that value must be the same
                        // for all active invocations
                        if (BallotOp::inverse_ballot == operation)
                            memset(&t[ii], (int)(genrand_int32(gMTdata)) & 0xff,
                                   n * sizeof(Ty));
                        else
                        {
                            memset(&t[ii], 0, n * sizeof(Ty));
                            i = (int)(genrand_int32(gMTdata) % (cl_uint)n);
                            set_value(t[ii + i], 41);
                        }
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
        int ii, i, j, k, n, l;
        int nj = (nw + ns - 1) / ns;
        cl_uint4 tr, rr;

        log_info("  sub_group_%s(%s)...\n", operation_names(operation),
                 TypeManager<Ty>::name());

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
                int midx = 4 * ii
                    + 2; // take index of array where info which work_item will
                         // be broadcast its value is stored
                l = (int)m[midx]; // take subgroup local id of this work_item
                // Check result

                for (i = 0; i < n; ++i)
                { // for each subgroup
                    int bit_value = 0;
                    int bit_mask = 1
                        << ((BallotOp::inverse_ballot == operation)
                                ? i
                                : l % 32); // from which value of bitfield bit
                                           // verification will be done

                    if (i < 32)
                        (mx[ii + i].s0 & bit_mask) > 0 ? bit_value = 1
                                                       : bit_value = 0;
                    if (i >= 32 && i < 64)
                        (mx[ii + i].s1 & bit_mask) > 0 ? bit_value = 1
                                                       : bit_value = 0;
                    if (i >= 64 && i < 96)
                        (mx[ii + i].s2 & bit_mask) > 0 ? bit_value = 1
                                                       : bit_value = 0;
                    if (i >= 96 && i < 128)
                        (mx[ii + i].s3 & bit_mask) > 0 ? bit_value = 1
                                                       : bit_value = 0;

                    bit_value == 1 ? tr = { 1, 0, 0, 0 } : tr = { 0, 0, 0, 0 };

                    rr = my[ii + i];
                    if (!compare(rr, tr))
                    {
                        log_error(
                            "ERROR: sub_group_%s mismatch for local id %d in "
                            "sub group %d in group %d obtained {%d, %d, %d, "
                            "%d}, expected {%d, %d, %d, %d}\n",
                            operation_names(operation), i, j, k, rr.s0, rr.s1,
                            rr.s2, rr.s3, tr.s0, tr.s1, tr.s2, tr.s3);
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


// Test for bit count/inclusive and exclusive scan/ find lsb msb ballot function
template <typename Ty, BallotOp operation> struct BALLOT3
{
    static void gen(Ty *x, Ty *t, cl_int *m, int ns, int nw, int ng)
    {
        int i, ii, j, k, n;
        int nj = (nw + ns - 1) / ns;
        int e;
        ii = 0;
        for (k = 0; k < ng; ++k)
        { // for each work_group
            for (j = 0; j < nj; ++j)
            { // for each subgroup
                ii = j * ns;
                n = ii + ns > nw ? nw - ii : ns;
                if (operation == BallotOp::ballot_bit_count
                    || operation == BallotOp::ballot_inclusive_scan
                    || operation == BallotOp::ballot_exclusive_scan)
                {
                    // Initialize data matrix indexed by local id and sub group
                    // id
                    e = (int)(genrand_int32(gMTdata) % 3);
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
                else if (operation == BallotOp::ballot_find_lsb
                         || operation == BallotOp::ballot_find_msb)
                {
                    // Regarding to the spec, find lsb and find msb result is
                    // undefined behavior if input value is zero, so generate
                    // only non-zero values.
                    e = (int)(genrand_int32(gMTdata) % 2);
                    switch (e)
                    {
                        case 0: memset(&t[ii], 0xff, n * sizeof(Ty)); break;
                        case 1:
                            char x = (genrand_int32(gMTdata)) & 0xff;
                            memset(&t[ii], x ? x : 1, n * sizeof(Ty));
                            break;
                    }
                }
                else
                {
                    log_error("Unknown operation...");
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

    static int chk(Ty *x, Ty *y, Ty *mx, Ty *my, cl_int *m, int ns, int nw,
                   int ng)
    {
        int ii, i, j, k, n;
        int nj = (nw + ns - 1) / ns;
        cl_uint4 tr, rr;

        log_info("  sub_group_ballot_%s(%s)...\n", operation_names(operation),
                 TypeManager<Ty>::name());

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
                // Check result
                tr = { 0, 0, 0, 0 };
                for (i = 0; i < n; ++i)
                { // for each subgroup
                    bs128 bs;
                    // convert cl_uint4 input into std::bitset<128>
                    bs |= bs128(mx[ii + i].s0) | (bs128(mx[ii + i].s1) << 32)
                        | (bs128(mx[ii + i].s2) << 64)
                        | (bs128(mx[ii + i].s3) << 96);
                    bs &= getImportantBits(i, n);

                    rr = my[ii + i];
                    if (operation == BallotOp::ballot_bit_count)
                    {
                        tr.s0 = bs.count();
                        if (!compare(rr, tr))
                        {
                            log_error("ERROR: sub_group_ballot_bit_count "
                                      "mismatch for local id %d in sub group "
                                      "%d in group %d obtained {%d, %d, %d, "
                                      "%d}, expected {%d, %d, %d, %d}\n",
                                      i, j, k, rr.s0, rr.s1, rr.s2, rr.s3,
                                      tr.s0, tr.s1, tr.s2, tr.s3);
                            return -1;
                        }
                    }
                    else if (operation == BallotOp::ballot_inclusive_scan)
                    {
                        tr.s0 = bs.count();
                        if (!compare(rr, tr))
                        {
                            log_error("ERROR: sub_group_ballot_inclusive_scan "
                                      "mismatch for local id %d in sub group "
                                      "%d in group %d obtained {%d, %d, %d, "
                                      "%d}, expected {%d, %d, %d, %d}\n",
                                      i, j, k, rr.s0, rr.s1, rr.s2, rr.s3,
                                      tr.s0, tr.s1, tr.s2, tr.s3);
                            return -1;
                        }
                    }
                    else if (operation == BallotOp::ballot_exclusive_scan)
                    {
                        tr.s0 = bs.count();
                        if (!compare(rr, tr))
                        {
                            log_error("ERROR: sub_group_ballot_exclusive_scan "
                                      "mismatch for local id %d in sub group "
                                      "%d in group %d obtained {%d, %d, %d, "
                                      "%d}, expected {%d, %d, %d, %d}\n",
                                      i, j, k, rr.s0, rr.s1, rr.s2, rr.s3,
                                      tr.s0, tr.s1, tr.s2, tr.s3);
                            return -1;
                        }
                    }
                    else if (operation == BallotOp::ballot_find_lsb)
                    {
                        for (int id = 0; id < n; ++id)
                        {
                            if (bs.test(id))
                            {
                                tr.s0 = id;
                                break;
                            }
                        }
                        if (!compare(rr, tr))
                        {
                            log_error("ERROR: sub_group_ballot_find_lsb "
                                      "mismatch for local id %d in sub group "
                                      "%d in group %d obtained {%d, %d, %d, "
                                      "%d}, expected {%d, %d, %d, %d}\n",
                                      i, j, k, rr.s0, rr.s1, rr.s2, rr.s3,
                                      tr.s0, tr.s1, tr.s2, tr.s3);
                            return -1;
                        }
                    }
                    else if (operation == BallotOp::ballot_find_msb)
                    {
                        for (int id = n - 1; id >= 0; --id)
                        {
                            if (bs.test(id))
                            {
                                tr.s0 = id;
                                break;
                            }
                        }
                        if (!compare(rr, tr))
                        {
                            log_error("ERROR: sub_group_ballot_find_msb "
                                      "mismatch for local id %d in sub group "
                                      "%d in group %d obtained {%d, %d, %d, "
                                      "%d}, expected {%d, %d, %d, %d}\n",
                                      i, j, k, rr.s0, rr.s1, rr.s2, rr.s3,
                                      tr.s0, tr.s1, tr.s2, tr.s3);
                            return -1;
                        }
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

// test mask functions
template <typename Ty, BallotOp operation> struct SMASK
{
    static void gen(Ty *x, Ty *t, cl_int *m, int ns, int nw, int ng)
    {
        int i, ii, j, k, n, l;
        int nj = (nw + ns - 1) / ns;
        int d = ns > 100 ? 100 : ns;
        int e;

        ii = 0;
        for (k = 0; k < ng; ++k)
        { // for each work_group
            for (j = 0; j < nj; ++j)
            { // for each subgroup
                ii = j * ns;
                n = ii + ns > nw ? nw - ii : ns;
                // Produce expected masks for each work item in the subgroup
                for (i = 0; i < n; ++i)
                {
                    int midx = 4 * ii + 4 * i;
                    cl_uint max_sub_group_size = m[midx + 2];
                    cl_uint4 expected_mask = { 0 };
                    expected_mask = generate_bit_mask(
                        i, operation_names(operation), max_sub_group_size);
                    set_value(t[ii + i], expected_mask);
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
        Ty taa, raa;

        log_info("  get_sub_group_%s_mask...\n", operation_names(operation));

        for (k = 0; k < ng; ++k)
        { // for each work_group
            for (j = 0; j < nw; ++j)
            { // inside the work_group
                i = m[4 * j + 1] * ns + m[4 * j];
                mx[i] = x[j]; // read host inputs for work_group
                my[i] = y[j]; // read device outputs for work_group
            }

            for (j = 0; j < nj; ++j)
            {
                ii = j * ns;
                n = ii + ns > nw ? nw - ii : ns;

                // Check result
                for (i = 0; i < n; ++i)
                { // inside the subgroup
                    taa = mx[ii + i]; // read host input for subgroup
                    raa = my[ii + i]; // read device outputs for subgroup
                    if (!compare(raa, taa))
                    {
                        log_error("ERROR:  get_sub_group_%s_mask... mismatch "
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

static const char *bcast_non_uniform_source =
    "__kernel void test_bcast_non_uniform(const __global Type *in, __global "
    "int4 *xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    "    Type x = in[gid];\n"
    " if (xy[gid].x < NON_UNIFORM_WG_SIZE) {\n" // broadcast 4 values , other
                                                // values are
                                                // 0
    "    out[gid] = sub_group_non_uniform_broadcast(x, xy[gid].z);\n"
    " }\n"
    "}\n";
static const char *bcast_first_source =
    "__kernel void test_bcast_first(const __global Type *in, __global int4 "
    "*xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    "    Type x = in[gid];\n"
    "    out[gid] = sub_group_broadcast_first(x);\n"
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
    "__kernel void test_sub_group_ballot(const __global Type *in, __global "
    "int4 *xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    "    Type x = in[gid];\n"
    "    uint4 value = sub_group_ballot(x.s0);\n"
    "    out[gid] = value;\n"
    "}\n";
static const char *inverse_ballot_source =
    "__kernel void test_sub_group_inverse_ballot(const __global Type *in, "
    "__global int4 *xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    "    Type x = in[gid];\n"
    "    uint4 value = (uint4)(10,0,0,0);\n"
    "    if (sub_group_inverse_ballot(x)) {\n"
    "       value = (uint4)(1,0,0,0);\n"
    "    } else {\n"
    "       value = (uint4)(0,0,0,0);\n"
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
    "    if (sub_group_ballot_bit_extract(x, index)) {\n"
    "       value = (uint4)(1,0,0,0);\n"
    "    } else {\n"
    "       value = (uint4)(0,0,0,0);\n"
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
            test<T, BC<T, SubgroupsBroadcastOp::non_uniform_broadcast>, GWS,
                 LWS>::run(device_, context_, queue_, num_elements_,
                           "test_bcast_non_uniform", bcast_non_uniform_source,
                           0, useCoreSubgroups_, required_extensions_);
        return error;
    }

    template <typename T> int run_bc_first()
    {
        int error = test<T, BC<T, SubgroupsBroadcastOp::broadcast_first>, GWS,
                         LWS>::run(device_, context_, queue_, num_elements_,
                                   "test_bcast_first", bcast_first_source, 0,
                                   useCoreSubgroups_, required_extensions_);
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
        int error = test<cl_uint4, BALLOT<cl_uint4>, GWS, LWS>::run(
            device_, context_, queue_, num_elements_, "test_sub_group_ballot",
            ballot_source, 0, useCoreSubgroups_, required_extensions_);

        error |= test<cl_uint4, BALLOT2<cl_uint4, BallotOp::inverse_ballot>,
                      GWS, LWS>::run(device_, context_, queue_, num_elements_,
                                     "test_sub_group_inverse_ballot",
                                     inverse_ballot_source, 0,
                                     useCoreSubgroups_, required_extensions_);
        error |= test<cl_uint4, BALLOT2<cl_uint4, BallotOp::ballot_bit_extract>,
                      GWS, LWS>::run(device_, context_, queue_, num_elements_,
                                     "test_sub_group_ballot_bit_extract",
                                     ballot_bit_extract_source, 0,
                                     useCoreSubgroups_, required_extensions_);

        error |= test<cl_uint4, BALLOT3<cl_uint4, BallotOp::ballot_bit_count>,
                      GWS, LWS>::run(device_, context_, queue_, num_elements_,
                                     "test_sub_group_ballot_bit_count",
                                     ballot_bit_count_source, 0,
                                     useCoreSubgroups_, required_extensions_);
        error |=
            test<cl_uint4, BALLOT3<cl_uint4, BallotOp::ballot_inclusive_scan>,
                 GWS, LWS>::run(device_, context_, queue_, num_elements_,
                                "test_sub_group_ballot_inclusive_scan",
                                ballot_inclusive_scan_source, 0,
                                useCoreSubgroups_, required_extensions_);
        error |=
            test<cl_uint4, BALLOT3<cl_uint4, BallotOp::ballot_exclusive_scan>,
                 GWS, LWS>::run(device_, context_, queue_, num_elements_,
                                "test_sub_group_ballot_exclusive_scan",
                                ballot_exclusive_scan_source, 0,
                                useCoreSubgroups_, required_extensions_);
        error |= test<cl_uint4, BALLOT3<cl_uint4, BallotOp::ballot_find_lsb>,
                      GWS, LWS>::run(device_, context_, queue_, num_elements_,
                                     "test_sub_group_ballot_find_lsb",
                                     ballot_find_lsb_source, 0,
                                     useCoreSubgroups_, required_extensions_);
        error |= test<cl_uint4, BALLOT3<cl_uint4, BallotOp::ballot_find_msb>,
                      GWS, LWS>::run(device_, context_, queue_, num_elements_,
                                     "test_sub_group_ballot_find_msb",
                                     ballot_find_msb_source, 0,
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
