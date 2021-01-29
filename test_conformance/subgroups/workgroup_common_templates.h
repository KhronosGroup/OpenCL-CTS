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
#include "subhelpers.h"


typedef std::bitset<128> bs128;
static cl_uint4 generate_bit_mask(cl_uint subgroup_local_id,
                                  const std::string &mask_type,
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
// sub_group_broadcast - each work_item registers it's own value.
// All work_items in subgroup takes one value from only one (any) work_item
// sub_group_broadcast_first - same as type 0. All work_items in
// subgroup takes only one value from only one chosen (the smallest subgroup ID)
// work_item
// sub_group_non_uniform_broadcast - same as type 0 but
// only 4 work_items from subgroup enter the code (are active)
template <typename Ty, SubgroupsBroadcastOp operation> struct BC
{
    static void gen(Ty *x, Ty *t, cl_int *m, int ns, int nw, int ng)
    {
        int i, ii, j, k, n;
        int nj = (nw + ns - 1) / ns;
        int d = ns > 100 ? 100 : ns;
        int non_uniform_size = ng % nw;
        ng = ng / nw;
        int last_subgroup_size = 0;
        ii = 0;

        log_info("  sub_group_%s(%s)...\n", operation_names(operation),
                 TypeManager<Ty>::name());
        if (non_uniform_size)
        {
            log_info("  non uniform work group size mode ON\n");
            ng++;
        }
        for (k = 0; k < ng; ++k)
        { // for each work_group
            if (non_uniform_size && k == ng - 1)
            {
                set_last_worgroup_params(non_uniform_size, nj, ns, nw,
                                         last_subgroup_size);
            }
            for (j = 0; j < nj; ++j)
            { // for each subgroup
                ii = j * ns;
                if (last_subgroup_size && j == nj - 1)
                {
                    n = last_subgroup_size;
                }
                else
                {
                    n = ii + ns > nw ? nw - ii : ns;
                }
                int bcast_if = 0;
                int bcast_elseif = 0;
                int bcast_index = (int)(genrand_int32(gMTdata) & 0x7fffffff)
                    % (d > n ? n : d);
                // l - calculate subgroup local id from which value will be
                // broadcasted (one the same value for whole subgroup)
                if (operation != SubgroupsBroadcastOp::broadcast)
                {
                    // reduce brodcasting index in case of non_uniform and
                    // last workgroup last subgroup
                    if (last_subgroup_size && j == nj - 1
                        && last_subgroup_size < NR_OF_ACTIVE_WORK_ITEMS)
                    {
                        bcast_if = bcast_index % last_subgroup_size;
                        bcast_elseif = bcast_if;
                    }
                    else
                    {
                        bcast_if = bcast_index % NR_OF_ACTIVE_WORK_ITEMS;
                        bcast_elseif = NR_OF_ACTIVE_WORK_ITEMS
                            + bcast_index % (n - NR_OF_ACTIVE_WORK_ITEMS);
                    }
                }

                for (i = 0; i < n; ++i)
                {
                    if (operation == SubgroupsBroadcastOp::broadcast)
                    {
                        int midx = 4 * ii + 4 * i + 2;
                        m[midx] = (cl_int)bcast_index;
                    }
                    else
                    {
                        if (i < NR_OF_ACTIVE_WORK_ITEMS)
                        {
                            // index of the third
                            // element int the vector.
                            int midx = 4 * ii + 4 * i + 2;
                            // storing information about
                            // broadcasting index -
                            // earlier calculated
                            m[midx] = (cl_int)bcast_if;
                        }
                        else
                        { // index of the third
                          // element int the vector.
                            int midx = 4 * ii + 4 * i + 3;
                            m[midx] = (cl_int)bcast_elseif;
                        }
                    }

                    // calculate value for broadcasting
                    cl_ulong number = genrand_int64(gMTdata);
                    set_value(t[ii + i], number);
                }
            }
            // Now map into work group using map from device
            for (j = 0; j < nw; ++j)
            { // for each element in work_group
                // calculate index as number of subgroup
                // plus subgroup local id
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
        int ii, i, j, k, l, n;
        int nj = (nw + ns - 1) / ns;
        Ty tr, rr;
        int non_uniform_size = ng % nw;
        ng = ng / nw;
        int last_subgroup_size = 0;
        if (non_uniform_size) ng++;

        for (k = 0; k < ng; ++k)
        { // for each work_group
            if (non_uniform_size && k == ng - 1)
            {
                set_last_worgroup_params(non_uniform_size, nj, ns, nw,
                                         last_subgroup_size);
            }
            for (j = 0; j < nw; ++j)
            { // inside the work_group
                i = m[4 * j + 1] * ns + m[4 * j];
                mx[i] = x[j]; // read host inputs for work_group
                my[i] = y[j]; // read device outputs for work_group
            }

            for (j = 0; j < nj; ++j)
            { // for each subgroup
                ii = j * ns;
                if (last_subgroup_size && j == nj - 1)
                {
                    n = last_subgroup_size;
                }
                else
                {
                    n = ii + ns > nw ? nw - ii : ns;
                }

                // Check result
                if (operation == SubgroupsBroadcastOp::broadcast_first)
                {
                    int lowest_active_id = -1;
                    for (i = 0; i < n; ++i)
                    {

                        lowest_active_id = i < NR_OF_ACTIVE_WORK_ITEMS
                            ? 0
                            : NR_OF_ACTIVE_WORK_ITEMS;
                        //  findout if broadcasted
                        //  value is the same
                        tr = mx[ii + lowest_active_id];
                        //  findout if broadcasted to all
                        rr = my[ii + i];

                        if (!compare(rr, tr))
                        {
                            log_error(
                                "ERROR: sub_group_broadcast_first(%s) "
                                "mismatch "
                                "for local id %d in sub group %d in group "
                                "%d\n",
                                TypeManager<Ty>::name(), i, j, k);
                            return TEST_FAIL;
                        }
                    }
                }
                else
                {
                    for (i = 0; i < n; ++i)
                    {
                        if (operation == SubgroupsBroadcastOp::broadcast)
                        {
                            int midx = 4 * ii + 4 * i + 2;
                            l = (int)m[midx];
                            tr = mx[ii + l];
                        }
                        else
                        {
                            if (i < NR_OF_ACTIVE_WORK_ITEMS)
                            { // take index of array where info
                              // which work_item will be
                              // broadcast its value is stored
                                int midx = 4 * ii + 4 * i + 2;
                                // take subgroup local id of
                                // this work_item
                                l = (int)m[midx];
                                // take value generated on host
                                // for this work_item
                                tr = mx[ii + l];
                            }
                            else
                            {
                                int midx = 4 * ii + 4 * i + 3;
                                l = (int)m[midx];
                                tr = mx[ii + l];
                            }
                        }
                        rr = my[ii + i]; // read device outputs for
                                         // work_item in the subgroup

                        if (!compare(rr, tr))
                        {
                            log_error("ERROR: sub_group_%s(%s) "
                                      "mismatch for local id %d in sub "
                                      "group %d in group %d - got %lu "
                                      "expected %lu\n",
                                      operation_names(operation),
                                      TypeManager<Ty>::name(), i, j, k, rr, tr);
                            return TEST_FAIL;
                        }
                    }
                }
            }
            x += nw;
            y += nw;
            m += 4 * nw;
        }
        log_info("  sub_group_%s(%s)... passed\n", operation_names(operation),
                 TypeManager<Ty>::name());
        return TEST_PASS;
    }
};

static float to_float(subgroups::cl_half x) { return cl_half_to_float(x.data); }

static subgroups::cl_half to_half(float x)
{
    subgroups::cl_half value;
    value.data = cl_half_from_double(x, CL_HALF_RTE);
    return value;
}

// for integer types
template <typename Ty> inline Ty calculate(Ty a, Ty b, ArithmeticOp operation)
{
    switch (operation)
    {
        case ArithmeticOp::add_: return a + b;
        case ArithmeticOp::max_: return a > b ? a : b;
        case ArithmeticOp::min_: return a < b ? a : b;
        case ArithmeticOp::mul_: return a * b;
        case ArithmeticOp::and_: return a & b;
        case ArithmeticOp::or_: return a | b;
        case ArithmeticOp::xor_: return a ^ b;
        case ArithmeticOp::logical_and: return a && b;
        case ArithmeticOp::logical_or: return a || b;
        case ArithmeticOp::logical_xor: return !a ^ !b;
        default: log_error("Unknown operation request"); break;
    }
    return 0;
}
// Specialize for floating points.
template <>
inline cl_double calculate(cl_double a, cl_double b, ArithmeticOp operation)
{
    switch (operation)
    {
        case ArithmeticOp::add_: {
            return a + b;
        }
        case ArithmeticOp::max_: {
            return a > b ? a : b;
        }
        case ArithmeticOp::min_: {
            return a < b ? a : b;
        }
        case ArithmeticOp::mul_: {
            return a * b;
        }
        default: log_error("Unknown operation request"); break;
    }
    return 0;
}

template <>
inline cl_float calculate(cl_float a, cl_float b, ArithmeticOp operation)
{
    switch (operation)
    {
        case ArithmeticOp::add_: {
            return a + b;
        }
        case ArithmeticOp::max_: {
            return a > b ? a : b;
        }
        case ArithmeticOp::min_: {
            return a < b ? a : b;
        }
        case ArithmeticOp::mul_: {
            return a * b;
        }
        default: log_error("Unknown operation request"); break;
    }
    return 0;
}

template <>
inline subgroups::cl_half calculate(subgroups::cl_half a, subgroups::cl_half b,
                                    ArithmeticOp operation)
{
    switch (operation)
    {
        case ArithmeticOp::add_: return to_half(to_float(a) + to_float(b));
        case ArithmeticOp::max_:
            return to_float(a) > to_float(b) || is_half_nan(b.data) ? a : b;
        case ArithmeticOp::min_:
            return to_float(a) < to_float(b) || is_half_nan(b.data) ? a : b;
        case ArithmeticOp::mul_: return to_half(to_float(a) * to_float(b));
        default: log_error("Unknown operation request"); break;
    }
    return to_half(0);
}

template <typename Ty> bool is_floating_point()
{
    return std::is_floating_point<Ty>::value
        || std::is_same<Ty, subgroups::cl_half>::value;
}

template <typename Ty, ArithmeticOp operation>
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
                if ((operation == ArithmeticOp::add_
                     || operation == ArithmeticOp::mul_)
                    && is_floating_point<Ty>())
                {
                    // work around different results depending on operation
                    // order by having input with little precision
                    x = genrand_int32(gMTdata) % 64;
                }
                else
                {
                    x = genrand_int64(gMTdata);
                    if ((operation == ArithmeticOp::logical_and
                         || operation == ArithmeticOp::logical_or
                         || operation == ArithmeticOp::logical_xor)
                        && ((x >> 32) & 1) == 0)
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
template <typename Ty, ArithmeticOp operation> struct RED
{
    static void gen(Ty *x, Ty *t, cl_int *m, int ns, int nw, int ng)
    {
        ng = ng / nw;
        log_info("  sub_group_reduce_%s(%s)... \n", operation_names(operation),
                 TypeManager<Ty>::name());
        genrand<Ty, operation>(x, t, m, ns, nw, ng);
    }

    static int chk(Ty *x, Ty *y, Ty *mx, Ty *my, cl_int *m, int ns, int nw,
                   int ng)
    {
        int ii, i, j, k, n;
        int nj = (nw + ns - 1) / ns;
        Ty tr, rr;
        ng = ng / nw;

        for (k = 0; k < ng; ++k)
        {
            // Map to array indexed to array indexed by local ID and sub
            // group
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
                    tr = calculate<Ty>(tr, mx[ii + i], operation);

                // Check result
                for (i = 0; i < n; ++i)
                {
                    rr = my[ii + i];
                    if (rr != tr)
                    {
                        log_error("ERROR: sub_group_reduce_%s(%s) mismatch for "
                                  "local id %d in sub group %d in group %d\n",
                                  operation_names(operation),
                                  TypeManager<Ty>::name(), i, j, k);
                        return TEST_FAIL;
                    }
                }
            }

            x += nw;
            y += nw;
            m += 4 * nw;
        }
        log_info("  sub_group_reduce_%s(%s)... passed\n",
                 operation_names(operation), TypeManager<Ty>::name());
        return TEST_PASS;
    }
};

// Scan Inclusive functions
template <typename Ty, ArithmeticOp operation> struct SCIN
{
    static void gen(Ty *x, Ty *t, cl_int *m, int ns, int nw, int ng)
    {
        ng = ng / nw;
        log_info("  sub_group_scan_inclusive_%s(%s)...\n",
                 operation_names(operation), TypeManager<Ty>::name());
        genrand<Ty, operation>(x, t, m, ns, nw, ng);
    }

    static int chk(Ty *x, Ty *y, Ty *mx, Ty *my, cl_int *m, int ns, int nw,
                   int ng)
    {
        int ii, i, j, k, n;
        int nj = (nw + ns - 1) / ns;
        Ty tr, rr;
        ng = ng / nw;

        for (k = 0; k < ng; ++k)
        {
            // Map to array indexed to array indexed by local ID and sub
            // group
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
                    tr = i == 0 ? mx[ii] : calculate(tr, mx[ii + i], operation);

                    rr = my[ii + i];
                    if (rr != tr)
                    {
                        log_error(
                            "ERROR: sub_group_scan_inclusive_%s(%s) "
                            "mismatch "
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
        log_info("  sub_group_scan_inclusive_%s(%s)... passed\n",
                 operation_names(operation), TypeManager<Ty>::name());
        return TEST_PASS;
    }
};

// Scan Exclusive functions
template <typename Ty, ArithmeticOp operation> struct SCEX
{
    static void gen(Ty *x, Ty *t, cl_int *m, int ns, int nw, int ng)
    {
        ng = ng / nw;
        log_info("  sub_group_scan_exclusive_%s(%s)...\n",
                 operation_names(operation), TypeManager<Ty>::name());
        genrand<Ty, operation>(x, t, m, ns, nw, ng);
    }

    static int chk(Ty *x, Ty *y, Ty *mx, Ty *my, cl_int *m, int ns, int nw,
                   int ng)
    {
        int ii, i, j, k, n;
        int nj = (nw + ns - 1) / ns;
        Ty tr, trt, rr;
        ng = ng / nw;

        for (k = 0; k < ng; ++k)
        {
            // Map to array indexed to array indexed by local ID and sub
            // group
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
                    switch (operation)
                    {
                        case ArithmeticOp::add_:
                            tr = i == 0 ? TypeManager<Ty>::identify_limits(
                                     ArithmeticOp::add_)
                                        : tr + trt;
                            break;
                        case ArithmeticOp::max_:
                            tr = i == 0 ? TypeManager<Ty>::identify_limits(
                                     ArithmeticOp::max_)
                                        : (trt > tr ? trt : tr);
                            break;
                        default:
                            tr = i == 0
                                ? TypeManager<Ty>::identify_limits(operation)
                                : (trt > tr ? tr : trt);
                            break;
                    }
                    trt = mx[ii + i];
                    rr = my[ii + i];

                    if (rr != tr)
                    {
                        log_error(
                            "ERROR: sub_group_scan_exclusive_%s(%s) "
                            "mismatch "
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
        log_info("  sub_group_scan_exclusive_%s(%s)... passed\n",
                 operation_names(operation), TypeManager<Ty>::name());
        return TEST_PASS;
    }
};

template <typename Ty, ShuffleOp operation> struct SHF
{
    static void gen(Ty *x, Ty *t, cl_int *m, int ns, int nw, int ng)
    {
        int i, ii, j, k, l, n, delta;
        int nj = (nw + ns - 1) / ns;
        int d = ns > 100 ? 100 : ns;
        ii = 0;
        ng = ng / nw;
        log_info("  sub_group_%s(%s)...\n", operation_names(operation),
                 TypeManager<Ty>::name());
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
                    switch (operation)
                    {
                        case ShuffleOp::shuffle:
                        case ShuffleOp::shuffle_xor:
                            // storing information about shuffle index
                            m[midx] = (cl_int)l;
                            break;
                        case ShuffleOp::shuffle_up:
                            delta = l; // calculate delta for shuffle up
                            if (i - delta < 0)
                            {
                                delta = i;
                            }
                            m[midx] = (cl_int)delta;
                            break;
                        case ShuffleOp::shuffle_down:
                            delta = l; // calculate delta for shuffle down
                            if (i + delta >= n)
                            {
                                delta = n - 1 - i;
                            }
                            m[midx] = (cl_int)delta;
                            break;
                        default: break;
                    }
                    cl_ulong number = genrand_int64(gMTdata);
                    set_value(t[ii + i], number);
                }
            }
            // Now map into work group using map from device
            for (j = 0; j < nw; ++j)
            { // for each element in work_group
                // calculate index as number of subgroup
                // plus subgroup local id
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
        int ii, i, j, k, l, n;
        int nj = (nw + ns - 1) / ns;
        Ty tr, rr;
        ng = ng / nw;

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
                  // shuffle index storage
                    int midx = 4 * ii + 4 * i + 2;
                    l = (int)m[midx];
                    rr = my[ii + i];
                    switch (operation)
                    {
                        // shuffle basic - treat l as index
                        case ShuffleOp::shuffle: tr = mx[ii + l]; break;
                        // shuffle up - treat l as delta
                        case ShuffleOp::shuffle_up: tr = mx[ii + i - l]; break;
                        // shuffle up - treat l as delta
                        case ShuffleOp::shuffle_down:
                            tr = mx[ii + i + l];
                            break;
                        // shuffle xor - treat l as mask
                        case ShuffleOp::shuffle_xor:
                            tr = mx[ii + (i ^ l)];
                            break;
                        default: break;
                    }

                    if (!compare(rr, tr))
                    {
                        log_error("ERROR: sub_group_%s(%s) mismatch for "
                                  "local id %d in sub group %d in group %d\n",
                                  operation_names(operation),
                                  TypeManager<Ty>::name(), i, j, k);
                        return TEST_FAIL;
                    }
                }
            }
            x += nw;
            y += nw;
            m += 4 * nw;
        }
        log_info("  sub_group_%s(%s)... passed\n", operation_names(operation),
                 TypeManager<Ty>::name());
        return TEST_PASS;
    }
};

#endif
