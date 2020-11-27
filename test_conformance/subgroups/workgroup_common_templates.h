//
// Copyright (c) 2020 The Khronos Group Inc.
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
#ifndef WORKGROUPCOMMONTEMPLATES_H
#define WORKGROUPCOMMONTEMPLATES_H

#include "typeWrappers.h"
#include <bitset>
#include "CL/cl_half.h"


typedef std::bitset<128> bs128;
inline cl_uint4 generate_bit_mask(cl_uint subgroup_local_id,
                                  std::string mask_type,
                                  cl_uint max_sub_group_size)
{
    bs128 mask128;
    cl_uint4 mask;
    cl_uint pos = subgroup_local_id;
    if (mask_type == "eq") mask128.set(pos);
    if (mask_type == "le" || mask_type == "lt")
    {
        for (cl_uint i = 0; i <= pos; i++) mask128.set(i);
        if (mask_type == "lt") mask128.reset(pos);
    }
    if (mask_type == "ge" || mask_type == "gt")
    {
        for (cl_uint i = pos; i < max_sub_group_size; i++) mask128.set(i);
        if (mask_type == "gt") mask128.reset(pos);
    }

    // convert std::bitset<128> to uint4
    auto const uint_mask = bs128{ static_cast<unsigned long>(-1) };
    mask.s0 = (mask128 & uint_mask).to_ulong();
    mask128 >>= 32;
    mask.s1 = (mask128 & uint_mask).to_ulong();
    mask128 >>= 32;
    mask.s2 = (mask128 & uint_mask).to_ulong();
    mask128 >>= 32;
    mask.s3 = (mask128 & uint_mask).to_ulong();

    return mask;
}

// DESCRIPTION :
// Which = 0 -  sub_group_broadcast - each work_item registers it's own value.
// All work_items in subgroup takes one value from only one (any) work_item
// Which = 1 -  sub_group_broadcast_first - same as type 0. All work_items in
// subgroup takes only one value from only one chosen (the smallest subgroup ID)
// work_item Which = 2 -  sub_group_non_uniform_broadcast - same as type 0 but
// only 4 work_items from subgroup enter the code (are active)

template <typename Ty, int Which = 0> struct BC
{
    static void gen(Ty *x, Ty *t, cl_int *m, int ns, int nw, int ng)
    {
        int i, ii, j, k, l, n;
        int nj = (nw + ns - 1) / ns;
        int d = ns > 100 ? 100 : ns;

        ii = 0;
        for (k = 0; k < ng; ++k)
        { // for each work_group
            for (j = 0; j < nj; ++j)
            { // for each subgroup
                ii = j * ns;
                n = ii + ns > nw ? nw - ii : ns;
                // l - calculate subgroup local id from which value will be
                // broadcasted (one the same value for whole subgroup)
                l = (int)(genrand_int32(gMTdata) & 0x7fffffff)
                    % (d > n ? n : d);
                if (Which == 2)
                {
                    // only 4 work_items in subgroup will be active
                    l = l % 4;
                }

                for (i = 0; i < n; ++i)
                {
                    int midx = 4 * ii + 4 * i
                        + 2; // index of the third element int the vector.
                    m[midx] =
                        (cl_int)l; // storing information about broadcasting
                                   // index - earlier calculated
                    cl_ulong number = genrand_int64(
                        gMTdata); // calculate value for broadcasting
                    set_value(t[ii + i], number);
                    // log_info("wg = %d ,sg = %d, inside sg = %d, number == %d,
                    // l = %d, midx = %d\n", k, j, i, number, l, midx);
                }
            }

            // Now map into work group using map from device
            for (j = 0; j < nw; ++j)
            { // for each element in work_group
                i = m[4 * j + 1] * ns
                    + m[4 * j]; // calculate index as number of subgroup plus
                                // subgroup local id
                x[j] = t[i];
            }
            x += nw;
            m += 4 * nw;
        }
    }

    static int chk(Ty *x, Ty *y, Ty *mx, Ty *my, cl_int *m, int ns, int nw,
                   int ng)
    {
        int ii, i, j, k, l, n;
        int nj = (nw + ns - 1) / ns;
        Ty tr, rr;

        if (Which == 0)
        {
            log_info("  sub_group_broadcast(%s)...\n", TypeManager<Ty>::name());
        }
        else if (Which == 1)
        {
            log_info("  sub_group_broadcast_first(%s)...\n",
                     TypeManager<Ty>::name());
        }
        else if (Which == 2)
        {
            log_info("  sub_group_non_uniform_broadcast(%s)...\n",
                     TypeManager<Ty>::name());
        }
        else
        {
            log_error("ERROR: Unknown function name...\n");
            return -1;
        }

        for (k = 0; k < ng; ++k)
        { // for each work_group
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
                tr = mx[ii
                        + l]; // take value generated on host for this work_item

                // Check result
                if (Which == 1)
                {
                    int lowest_active_id = -1;
                    for (i = 0; i < n; ++i)
                    {
                        tr = mx[ii + i];
                        rr = my[ii + i];
                        if (compare(rr, tr))
                        { // find work_item id in subgroup which value could be
                          // broadcasted
                            lowest_active_id = i;
                            break;
                        }
                    }
                    if (lowest_active_id == -1)
                    {
                        log_error(
                            "ERROR: sub_group_broadcast_first(%s) do not found "
                            "any matching values in sub group %d in group %d\n",
                            TypeManager<Ty>::name(), j, k);
                        return -1;
                    }
                    for (i = 0; i < n; ++i)
                    {
                        tr = mx[ii
                                + lowest_active_id]; //  findout if broadcasted
                                                     //  value is the same
                        rr = my[ii + i]; //  findout if broadcasted to all
                        if (!compare(rr, tr))
                        {
                            log_error(
                                "ERROR: sub_group_broadcast_first(%s) mismatch "
                                "for local id %d in sub group %d in group %d\n",
                                TypeManager<Ty>::name(), i, j, k);
                        }
                    }
                }
                else
                {
                    for (i = 0; i < n; ++i)
                    {
                        if (Which == 2 && i >= NON_UNIFORM)
                        {
                            break; // non uniform case - only first 4 workitems
                                   // should broadcast. Others are undefined.
                        }
                        rr = my[ii + i]; // read device outputs for work_item in
                                         // the subgroup
                        if (!compare(rr, tr))
                        {
                            if (Which == 0)
                            {
                                log_error("ERROR: sub_group_broadcast(%s) "
                                          "mismatch for local id %d in sub "
                                          "group %d in group %d\n",
                                          TypeManager<Ty>::name(), i, j, k);
                            }
                            if (Which == 2)
                            {
                                log_error("ERROR: "
                                          "sub_group_non_uniform_broadcast(%s) "
                                          "mismatch for local id %d in sub "
                                          "group %d in group %d\n",
                                          TypeManager<Ty>::name(), i, j, k);
                            }
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

template <typename Ty, int Which> struct OPERATION;

template <typename Ty> struct OPERATION<Ty, 0>
{
    static Ty calculate(Ty a, Ty b) { return a + b; }
};
template <typename Ty> struct OPERATION<Ty, 1>
{
    static Ty calculate(Ty a, Ty b) { return a > b ? a : b; }
};
template <typename Ty> struct OPERATION<Ty, 2>
{
    static Ty calculate(Ty a, Ty b) { return a < b ? a : b; }
};
template <typename Ty> struct OPERATION<Ty, 3>
{
    static Ty calculate(Ty a, Ty b) { return a * b; }
};
template <typename Ty> struct OPERATION<Ty, 4>
{
    static Ty calculate(Ty a, Ty b) { return a & b; }
};
template <typename Ty> struct OPERATION<Ty, 5>
{
    static Ty calculate(Ty a, Ty b) { return a | b; }
};
template <typename Ty> struct OPERATION<Ty, 6>
{
    static Ty calculate(Ty a, Ty b) { return a ^ b; }
};
template <typename Ty> struct OPERATION<Ty, 7>
{
    static Ty calculate(Ty a, Ty b) { return a && b; }
};
template <typename Ty> struct OPERATION<Ty, 8>
{
    static Ty calculate(Ty a, Ty b) { return a || b; }
};
template <typename Ty> struct OPERATION<Ty, 9>
{
    static Ty calculate(Ty a, Ty b) { return !a ^ !b; }
};

static float to_float(subgroups::cl_half x) { return cl_half_to_float(x.data); }

static subgroups::cl_half to_half(float x)
{
    subgroups::cl_half value;
    value.data = cl_half_from_double(x, CL_HALF_RTE);
    return value;
}

template <> struct OPERATION<subgroups::cl_half, 0>
{
    static subgroups::cl_half calculate(subgroups::cl_half a,
                                        subgroups::cl_half b)
    {
        return to_half(to_float(a) + to_float(b));
    }
};

template <> struct OPERATION<subgroups::cl_half, 1>
{
    static subgroups::cl_half calculate(subgroups::cl_half a,
                                        subgroups::cl_half b)
    {
        return to_float(a) > to_float(b) || isnan_half(b) ? a : b;
    }
};

template <> struct OPERATION<subgroups::cl_half, 2>
{
    static subgroups::cl_half calculate(subgroups::cl_half a,
                                        subgroups::cl_half b)
    {
        return to_float(a) < to_float(b) || isnan_half(b) ? a : b;
    }
};

template <> struct OPERATION<subgroups::cl_half, 3>
{
    static subgroups::cl_half calculate(subgroups::cl_half a,
                                        subgroups::cl_half b)
    {
        return to_half(to_float(a) * to_float(b));
    }
};

static const char *const operation_names[] = {
    "add", "max", "min",         "mul",        "and",
    "or",  "xor", "logical_and", "logical_or", "logical_xor"
};

template <typename Ty> bool is_floating_point()
{
    return std::is_floating_point<Ty>::value
        || std::is_same<Ty, subgroups::cl_half>::value;
}

template <typename Ty, int Which>
void genrand(Ty *x, Ty *t, cl_int *m, int ns, int nw, int ng)
{
    int nj = (nw + ns - 1) / ns;

    for (int k = 0; k < ng; ++k)
    {
        for (int j = 0; j < nj; ++j)
        {
            int ii = j * ns;
            int n = ii + ns > nw ? nw - ii : ns;

            for (int i = 0; i < n; ++i)
            {
                cl_ulong x;
                if ((Which == 0 || Which == 3) && is_floating_point<Ty>())
                {
                    // work around different results depending on operation
                    // order by having input with little precision
                    x = genrand_int32(gMTdata) % 64;
                }
                else
                {
                    x = genrand_int64(gMTdata);
                    if (Which >= 7 && Which <= 9 && ((x >> 32) & 1) == 0)
                        x = 0; // increase probability of false
                }
                set_value(t[ii + i], x);
            }
        }

        // Now map into work group using map from device
        for (int j = 0; j < nw; ++j)
        {
            int i = m[4 * j + 1] * ns + m[4 * j];
            x[j] = t[i];
        }

        x += nw;
        m += 4 * nw;
    }
}

// Reduce functions
template <typename Ty, int Which> struct RED
{
    static void gen(Ty *x, Ty *t, cl_int *m, int ns, int nw, int ng)
    {
        genrand<Ty, Which>(x, t, m, ns, nw, ng);
    }

    static int chk(Ty *x, Ty *y, Ty *mx, Ty *my, cl_int *m, int ns, int nw,
                   int ng)
    {
        int ii, i, j, k, n;
        int nj = (nw + ns - 1) / ns;
        Ty tr, rr;

        log_info("  sub_group_reduce_%s(%s)...\n", operation_names[Which],
                 TypeManager<Ty>::name());

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
                tr = mx[ii];
                for (i = 1; i < n; ++i)
                    tr = OPERATION<Ty, Which>::calculate(tr, mx[ii + i]);

                // Check result
                for (i = 0; i < n; ++i)
                {
                    rr = my[ii + i];
                    if (rr != tr)
                    {
                        log_error("ERROR: sub_group_reduce_%s(%s) mismatch for "
                                  "local id %d in sub group %d in group %d\n",
                                  operation_names[Which],
                                  TypeManager<Ty>::name(), i, j, k);
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

// Scan Inclusive functions
template <typename Ty, int Which> struct SCIN
{
    static void gen(Ty *x, Ty *t, cl_int *m, int ns, int nw, int ng)
    {
        genrand<Ty, Which>(x, t, m, ns, nw, ng);
    }

    static int chk(Ty *x, Ty *y, Ty *mx, Ty *my, cl_int *m, int ns, int nw,
                   int ng)
    {
        int ii, i, j, k, n;
        int nj = (nw + ns - 1) / ns;
        Ty tr, rr;

        log_info("  sub_group_scan_inclusive_%s(%s)...\n",
                 operation_names[Which], TypeManager<Ty>::name());

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

                // Check result
                for (i = 0; i < n; ++i)
                {
                    tr = i == 0
                        ? mx[ii]
                        : OPERATION<Ty, Which>::calculate(tr, mx[ii + i]);

                    rr = my[ii + i];
                    if (rr != tr)
                    {
                        log_error(
                            "ERROR: sub_group_scan_inclusive_%s(%s) mismatch "
                            "for local id %d in sub group %d in group %d\n",
                            operation_names[Which], TypeManager<Ty>::name(), i,
                            j, k);
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

// Scan Exclusive functions
template <typename Ty, int Which> struct SCEX
{
    static void gen(Ty *x, Ty *t, cl_int *m, int ns, int nw, int ng)
    {
        genrand<Ty, Which>(x, t, m, ns, nw, ng);
    }

    static int chk(Ty *x, Ty *y, Ty *mx, Ty *my, cl_int *m, int ns, int nw,
                   int ng)
    {
        int ii, i, j, k, n;
        int nj = (nw + ns - 1) / ns;
        Ty tr, trt, rr;

        log_info("  sub_group_scan_exclusive_%s(%s)...\n",
                 operation_names[Which], TypeManager<Ty>::name());

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

                // Check result
                for (i = 0; i < n; ++i)
                {
                    if (Which == 0)
                    {
                        tr = i == 0 ? TypeManager<Ty>::identify_limits(Which)
                                    : tr + trt;
                    }
                    else if (Which == 1)
                    {
                        tr = i == 0 ? TypeManager<Ty>::identify_limits(Which)
                                    : (trt > tr ? trt : tr);
                    }
                    else
                    {
                        tr = i == 0 ? TypeManager<Ty>::identify_limits(Which)
                                    : (trt > tr ? tr : trt);
                    }
                    trt = mx[ii + i];
                    rr = my[ii + i];

                    if (rr != tr)
                    {
                        log_error(
                            "ERROR: sub_group_scan_exclusive_%s(%s) mismatch "
                            "for local id %d in sub group %d in group %d\n",
                            operation_names[Which], TypeManager<Ty>::name(), i,
                            j, k);
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

template <typename Ty, int Which = 0> struct SHF
{
    static void gen(Ty *x, Ty *t, cl_int *m, int ns, int nw, int ng)
    {
        int i, ii, j, k, l, n, delta;
        int nj = (nw + ns - 1) / ns;
        int d = ns > 100 ? 100 : ns;
        ii = 0;
        for (k = 0; k < ng; ++k)
        { // for each work_group
            for (j = 0; j < nj; ++j)
            { // for each subgroup
                ii = j * ns;
                n = ii + ns > nw ? nw - ii : ns;
                for (i = 0; i < n; ++i)
                {
                    int midx = 4 * ii + 4 * i + 2;
                    l = (int)(genrand_int32(gMTdata) & 0x7fffffff)
                        % (d > n ? n : d);
                    if (Which == 0 || Which == 3)
                    {
                        // storing information about shuffle index
                        m[midx] = (cl_int)l;
                    }

                    if (Which == 1)
                    {
                        delta = l; // calculate delta for shuffle up
                        if (i - delta < 0)
                        {
                            delta = i;
                        }
                        m[midx] = (cl_int)delta;
                    }

                    if (Which == 2)
                    {
                        delta = l; // calculate delta for shuffle down
                        if (i + delta >= n)
                        {
                            delta = n - 1 - i;
                        }
                        m[midx] = (cl_int)delta;
                    }

                    cl_ulong number = genrand_int64(gMTdata);
                    set_value(t[ii + i], number);

                }
            }
            // Now map into work group using map from device
            for (j = 0; j < nw; ++j)
            { // for each element in work_group
                i = m[4 * j + 1] * ns
                    + m[4 * j]; // calculate index as number of subgroup plus
                                // subgroup local id
                x[j] = t[i];
            }
            x += nw;
            m += 4 * nw;
        }
    }

    static int chk(Ty *x, Ty *y, Ty *mx, Ty *my, cl_int *m, int ns, int nw,
                   int ng)
    {
        int ii, i, j, k, l, n;
        int nj = (nw + ns - 1) / ns;
        Ty tr, rr;


        log_info("  sub_group_shuffle%s(%s)...\n",
                 Which == 0
                     ? ""
                     : (Which == 1 ? "_up" : Which == 2 ? "_down" : "_xor"),
                 TypeManager<Ty>::name());

        for (k = 0; k < ng; ++k)
        { // for each work_group
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

                for (i = 0; i < n; ++i)
                { // inside the subgroup
                    int midx = 4 * ii + 4 * i + 2; // shuffle index storage
                    l = (int)m[midx];
                    rr = my[ii + i];
                    if (Which == 0)
                    {
                        tr = mx[ii + l]; // shuffle basic - treat l as index
                    }
                    if (Which == 1)
                    {
                        tr = mx[ii + i - l]; // shuffle up - treat l as delta
                    }
                    if (Which == 2)
                    {
                        tr = mx[ii + i + l]; // shuffle down - treat l as delta
                    }
                    if (Which == 3)
                    {
                        tr = mx[ii + (i ^ l)]; // shuffle xor - treat l as mask
                    }

                    if (!compare(rr, tr))
                    {
                        log_error("ERROR: sub_group_shuffle%s(%s) mismatch for "
                                  "local id %d in sub group %d in group %d\n",
                                  Which == 0
                                      ? ""
                                      : (Which == 1
                                             ? "_up"
                                             : (Which == 2 ? "_down" : "_xor")),
                                  TypeManager<Ty>::name(), i, j, k);
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

#endif
