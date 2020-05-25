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

static const char *any_source = "__kernel void test_any(const __global Type "
                                "*in, __global int2 *xy, __global Type *out)\n"
                                "{\n"
                                "    int gid = get_global_id(0);\n"
                                "    XY(xy,gid);\n"
                                "    out[gid] = sub_group_any(in[gid]);\n"
                                "}\n";

static const char *all_source = "__kernel void test_all(const __global Type "
                                "*in, __global int2 *xy, __global Type *out)\n"
                                "{\n"
                                "    int gid = get_global_id(0);\n"
                                "    XY(xy,gid);\n"
                                "    out[gid] = sub_group_all(in[gid]);\n"
                                "}\n";

static const char *bcast_source =
    "__kernel void test_bcast(const __global Type *in, __global int2 *xy, "
    "__global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    "    Type x = in[gid];\n"
    "    size_t loid = (size_t)((int)x % 100);\n"
    "    out[gid] = sub_group_broadcast(x, loid);\n"
    "}\n";

static const char *redadd_source =
    "__kernel void test_redadd(const __global Type *in, __global int2 *xy, "
    "__global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    "    out[gid] = sub_group_reduce_add(in[gid]);\n"
    "}\n";

static const char *redmax_source =
    "__kernel void test_redmax(const __global Type *in, __global int2 *xy, "
    "__global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    "    out[gid] = sub_group_reduce_max(in[gid]);\n"
    "}\n";

static const char *redmin_source =
    "__kernel void test_redmin(const __global Type *in, __global int2 *xy, "
    "__global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    "    out[gid] = sub_group_reduce_min(in[gid]);\n"
    "}\n";

static const char *scinadd_source =
    "__kernel void test_scinadd(const __global Type *in, __global int2 *xy, "
    "__global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    "    out[gid] = sub_group_scan_inclusive_add(in[gid]);\n"
    "}\n";

static const char *scinmax_source =
    "__kernel void test_scinmax(const __global Type *in, __global int2 *xy, "
    "__global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    "    out[gid] = sub_group_scan_inclusive_max(in[gid]);\n"
    "}\n";

static const char *scinmin_source =
    "__kernel void test_scinmin(const __global Type *in, __global int2 *xy, "
    "__global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    "    out[gid] = sub_group_scan_inclusive_min(in[gid]);\n"
    "}\n";

static const char *scexadd_source =
    "__kernel void test_scexadd(const __global Type *in, __global int2 *xy, "
    "__global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    "    out[gid] = sub_group_scan_exclusive_add(in[gid]);\n"
    "}\n";

static const char *scexmax_source =
    "__kernel void test_scexmax(const __global Type *in, __global int2 *xy, "
    "__global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    "    out[gid] = sub_group_scan_exclusive_max(in[gid]);\n"
    "}\n";

static const char *scexmin_source =
    "__kernel void test_scexmin(const __global Type *in, __global int2 *xy, "
    "__global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    "    out[gid] = sub_group_scan_exclusive_min(in[gid]);\n"
    "}\n";


// Any/All test functions
template <int Which> struct AA
{
    static void gen(cl_int *x, cl_int *t, cl_int *m, int ns, int nw, int ng)
    {
        int i, ii, j, k, n;
        int nj = (nw + ns - 1) / ns;
        int e;

        ii = 0;
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
                i = m[2 * j + 1] * ns + m[2 * j];
                x[j] = t[i];
            }

            x += nw;
            m += 2 * nw;
        }
    }

    static int chk(cl_int *x, cl_int *y, cl_int *mx, cl_int *my, cl_int *m,
                   int ns, int nw, int ng)
    {
        int ii, i, j, k, n;
        int nj = (nw + ns - 1) / ns;
        cl_int taa, raa;

        log_info("  sub_group_%s...\n", Which == 0 ? "any" : "all");

        for (k = 0; k < ng; ++k)
        {
            // Map to array indexed to array indexed by local ID and sub group
            for (j = 0; j < nw; ++j)
            {
                i = m[2 * j + 1] * ns + m[2 * j];
                mx[i] = x[j];
                my[i] = y[j];
            }

            for (j = 0; j < nj; ++j)
            {
                ii = j * ns;
                n = ii + ns > nw ? nw - ii : ns;

                // Compute target
                if (Which == 0)
                {
                    taa = 0;
                    for (i = 0; i < n; ++i) taa |= mx[ii + i] != 0;
                }
                else
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
                                  Which == 0 ? "any" : "all", i, j, k);
                        return -1;
                    }
                }
            }

            x += nw;
            y += nw;
            m += 2 * nw;
        }

        return 0;
    }
};

// Reduce functions
template <typename Ty, int Which> struct RED
{
    static void gen(Ty *x, Ty *t, cl_int *m, int ns, int nw, int ng)
    {
        int i, ii, j, k, n;
        int nj = (nw + ns - 1) / ns;

        ii = 0;
        for (k = 0; k < ng; ++k)
        {
            for (j = 0; j < nj; ++j)
            {
                ii = j * ns;
                n = ii + ns > nw ? nw - ii : ns;

                for (i = 0; i < n; ++i)
                    t[ii + i] = (Ty)(
                        (int)(genrand_int32(gMTdata) & 0x7fffffff) % ns + 1);
            }

            // Now map into work group using map from device
            for (j = 0; j < nw; ++j)
            {
                i = m[2 * j + 1] * ns + m[2 * j];
                x[j] = t[i];
            }

            x += nw;
            m += 2 * nw;
        }
    }

    static int chk(Ty *x, Ty *y, Ty *mx, Ty *my, cl_int *m, int ns, int nw,
                   int ng)
    {
        int ii, i, j, k, n;
        int nj = (nw + ns - 1) / ns;
        Ty tr, rr;

        log_info("  sub_group_reduce_%s(%s)...\n",
                 Which == 0 ? "add" : (Which == 1 ? "max" : "min"),
                 TypeName<Ty>::val());

        for (k = 0; k < ng; ++k)
        {
            // Map to array indexed to array indexed by local ID and sub group
            for (j = 0; j < nw; ++j)
            {
                i = m[2 * j + 1] * ns + m[2 * j];
                mx[i] = x[j];
                my[i] = y[j];
            }

            for (j = 0; j < nj; ++j)
            {
                ii = j * ns;
                n = ii + ns > nw ? nw - ii : ns;

                // Compute target
                if (Which == 0)
                {
                    // add
                    tr = mx[ii];
                    for (i = 1; i < n; ++i) tr += mx[ii + i];
                }
                else if (Which == 1)
                {
                    // max
                    tr = mx[ii];
                    for (i = 1; i < n; ++i)
                        tr = tr > mx[ii + i] ? tr : mx[ii + i];
                }
                else if (Which == 2)
                {
                    // min
                    tr = mx[ii];
                    for (i = 1; i < n; ++i)
                        tr = tr > mx[ii + i] ? mx[ii + i] : tr;
                }

                // Check result
                for (i = 0; i < n; ++i)
                {
                    rr = my[ii + i];
                    if (rr != tr)
                    {
                        log_error("ERROR: sub_group_reduce_%s(%s) mismatch for "
                                  "local id %d in sub group %d in group %d\n",
                                  Which == 0 ? "add"
                                             : (Which == 1 ? "max" : "min"),
                                  TypeName<Ty>::val(), i, j, k);
                        return -1;
                    }
                }
            }

            x += nw;
            y += nw;
            m += 2 * nw;
        }

        return 0;
    }
};

// Scan Inclusive functions
template <typename Ty, int Which> struct SCIN
{
    static void gen(Ty *x, Ty *t, cl_int *m, int ns, int nw, int ng)
    {
        int i, ii, j, k, n;
        int nj = (nw + ns - 1) / ns;

        ii = 0;
        for (k = 0; k < ng; ++k)
        {
            for (j = 0; j < nj; ++j)
            {
                ii = j * ns;
                n = ii + ns > nw ? nw - ii : ns;

                for (i = 0; i < n; ++i)
                    // t[ii+i] = (Ty)((int)(genrand_int32(gMTdata) & 0x7fffffff)
                    // % ns + 1);
                    t[ii + i] = (Ty)i;
            }

            // Now map into work group using map from device
            for (j = 0; j < nw; ++j)
            {
                i = m[2 * j + 1] * ns + m[2 * j];
                x[j] = t[i];
            }

            x += nw;
            m += 2 * nw;
        }
    }

    static int chk(Ty *x, Ty *y, Ty *mx, Ty *my, cl_int *m, int ns, int nw,
                   int ng)
    {
        int ii, i, j, k, n;
        int nj = (nw + ns - 1) / ns;
        Ty tr, rr;

        log_info("  sub_group_scan_inclusive_%s(%s)...\n",
                 Which == 0 ? "add" : (Which == 1 ? "max" : "min"),
                 TypeName<Ty>::val());

        for (k = 0; k < ng; ++k)
        {
            // Map to array indexed to array indexed by local ID and sub group
            for (j = 0; j < nw; ++j)
            {
                i = m[2 * j + 1] * ns + m[2 * j];
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
                        tr = i == 0 ? mx[ii] : tr + mx[ii + i];
                    }
                    else if (Which == 1)
                    {
                        tr = i == 0 ? mx[ii]
                                    : (tr > mx[ii + i] ? tr : mx[ii + i]);
                    }
                    else
                    {
                        tr = i == 0 ? mx[ii]
                                    : (tr > mx[ii + i] ? mx[ii + i] : tr);
                    }

                    rr = my[ii + i];
                    if (rr != tr)
                    {
                        log_error(
                            "ERROR: sub_group_scan_inclusive_%s(%s) mismatch "
                            "for local id %d in sub group %d in group %d\n",
                            Which == 0 ? "add" : (Which == 1 ? "max" : "min"),
                            TypeName<Ty>::val(), i, j, k);
                        return -1;
                    }
                }
            }

            x += nw;
            y += nw;
            m += 2 * nw;
        }

        return 0;
    }
};

// Scan Exclusive functions
template <typename Ty, int Which> struct SCEX
{
    static void gen(Ty *x, Ty *t, cl_int *m, int ns, int nw, int ng)
    {
        int i, ii, j, k, n;
        int nj = (nw + ns - 1) / ns;

        ii = 0;
        for (k = 0; k < ng; ++k)
        {
            for (j = 0; j < nj; ++j)
            {
                ii = j * ns;
                n = ii + ns > nw ? nw - ii : ns;

                for (i = 0; i < n; ++i)
                    t[ii + i] = (Ty)(
                        (int)(genrand_int32(gMTdata) & 0x7fffffff) % ns + 1);
            }

            // Now map into work group using map from device
            for (j = 0; j < nw; ++j)
            {
                i = m[2 * j + 1] * ns + m[2 * j];
                x[j] = t[i];
            }

            x += nw;
            m += 2 * nw;
        }
    }

    static int chk(Ty *x, Ty *y, Ty *mx, Ty *my, cl_int *m, int ns, int nw,
                   int ng)
    {
        int ii, i, j, k, n;
        int nj = (nw + ns - 1) / ns;
        Ty tr, trt, rr;

        log_info("  sub_group_scan_exclusive_%s(%s)...\n",
                 Which == 0 ? "add" : (Which == 1 ? "max" : "min"),
                 TypeName<Ty>::val());

        for (k = 0; k < ng; ++k)
        {
            // Map to array indexed to array indexed by local ID and sub group
            for (j = 0; j < nw; ++j)
            {
                i = m[2 * j + 1] * ns + m[2 * j];
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
                        tr = i == 0 ? TypeIdentity<Ty, Which>::val() : tr + trt;
                    }
                    else if (Which == 1)
                    {
                        tr = i == 0 ? TypeIdentity<Ty, Which>::val()
                                    : (trt > tr ? trt : tr);
                    }
                    else
                    {
                        tr = i == 0 ? TypeIdentity<Ty, Which>::val()
                                    : (trt > tr ? tr : trt);
                    }
                    trt = mx[ii + i];
                    rr = my[ii + i];

                    if (rr != tr)
                    {
                        log_error(
                            "ERROR: sub_group_scan_exclusive_%s(%s) mismatch "
                            "for local id %d in sub group %d in group %d\n",
                            Which == 0 ? "add" : (Which == 1 ? "max" : "min"),
                            TypeName<Ty>::val(), i, j, k);
                        return -1;
                    }
                }
            }

            x += nw;
            y += nw;
            m += 2 * nw;
        }

        return 0;
    }
};

// Broadcast functios
template <typename Ty> struct BC
{
    static void gen(Ty *x, Ty *t, cl_int *m, int ns, int nw, int ng)
    {
        int i, ii, j, k, l, n;
        int nj = (nw + ns - 1) / ns;
        int d = ns > 100 ? 100 : ns;

        ii = 0;
        for (k = 0; k < ng; ++k)
        {
            for (j = 0; j < nj; ++j)
            {
                ii = j * ns;
                n = ii + ns > nw ? nw - ii : ns;
                l = (int)(genrand_int32(gMTdata) & 0x7fffffff)
                    % (d > n ? n : d);

                for (i = 0; i < n; ++i)
                    t[ii + i] = (Ty)((int)(genrand_int32(gMTdata) & 0x7fffffff)
                                         % 100 * 100
                                     + l);
            }

            // Now map into work group using map from device
            for (j = 0; j < nw; ++j)
            {
                i = m[2 * j + 1] * ns + m[2 * j];
                x[j] = t[i];
            }

            x += nw;
            m += 2 * nw;
        }
    }

    static int chk(Ty *x, Ty *y, Ty *mx, Ty *my, cl_int *m, int ns, int nw,
                   int ng)
    {
        int ii, i, j, k, l, n;
        int nj = (nw + ns - 1) / ns;
        Ty tr, rr;

        log_info("  sub_group_broadcast(%s)...\n", TypeName<Ty>::val());

        for (k = 0; k < ng; ++k)
        {
            // Map to array indexed to array indexed by local ID and sub group
            for (j = 0; j < nw; ++j)
            {
                i = m[2 * j + 1] * ns + m[2 * j];
                mx[i] = x[j];
                my[i] = y[j];
            }

            for (j = 0; j < nj; ++j)
            {
                ii = j * ns;
                n = ii + ns > nw ? nw - ii : ns;
                l = (int)mx[ii] % 100;
                tr = mx[ii + l];

                // Check result
                for (i = 0; i < n; ++i)
                {
                    rr = my[ii + i];
                    if (rr != tr)
                    {
                        log_error("ERROR: sub_group_broadcast(%s) mismatch for "
                                  "local id %d in sub group %d in group %d\n",
                                  TypeName<Ty>::val(), i, j, k);
                        return -1;
                    }
                }
            }

            x += nw;
            y += nw;
            m += 2 * nw;
        }

        return 0;
    }
};


// Entry point from main
int test_work_group_functions(cl_device_id device, cl_context context,
                              cl_command_queue queue, int num_elements,
                              bool useCoreSubgroups)
{
    int error;

    // Adjust these individually below if desired/needed
#define G 2000
#define L 200

    error = test<int, AA<0>, G, L>::run(device, context, queue, num_elements,
                                        "test_any", any_source, 0,
                                        useCoreSubgroups);
    error |= test<int, AA<1>, G, L>::run(device, context, queue, num_elements,
                                         "test_all", all_source, 0,
                                         useCoreSubgroups);

    // error |= test<cl_half, BC<cl_half>, G, L>::run(device, context, queue,
    // num_elements, "test_bcast", bcast_source);
    error |= test<cl_uint, BC<cl_uint>, G, L>::run(
        device, context, queue, num_elements, "test_bcast", bcast_source, 0,
        useCoreSubgroups);
    error |= test<cl_int, BC<cl_int>, G, L>::run(
        device, context, queue, num_elements, "test_bcast", bcast_source, 0,
        useCoreSubgroups);
    error |= test<cl_ulong, BC<cl_ulong>, G, L>::run(
        device, context, queue, num_elements, "test_bcast", bcast_source, 0,
        useCoreSubgroups);
    error |= test<cl_long, BC<cl_long>, G, L>::run(
        device, context, queue, num_elements, "test_bcast", bcast_source, 0,
        useCoreSubgroups);
    error |= test<float, BC<float>, G, L>::run(
        device, context, queue, num_elements, "test_bcast", bcast_source, 0,
        useCoreSubgroups);
    error |= test<double, BC<double>, G, L>::run(
        device, context, queue, num_elements, "test_bcast", bcast_source, 0,
        useCoreSubgroups);

    // error |= test<cl_half, RED<cl_half,0>, G, L>::run(device, context, queue,
    // num_elements, "test_redadd", redadd_source);
    error |= test<cl_uint, RED<cl_uint, 0>, G, L>::run(
        device, context, queue, num_elements, "test_redadd", redadd_source, 0,
        useCoreSubgroups);
    error |= test<cl_int, RED<cl_int, 0>, G, L>::run(
        device, context, queue, num_elements, "test_redadd", redadd_source, 0,
        useCoreSubgroups);
    error |= test<cl_ulong, RED<cl_ulong, 0>, G, L>::run(
        device, context, queue, num_elements, "test_redadd", redadd_source, 0,
        useCoreSubgroups);
    error |= test<cl_long, RED<cl_long, 0>, G, L>::run(
        device, context, queue, num_elements, "test_redadd", redadd_source, 0,
        useCoreSubgroups);
    error |= test<float, RED<float, 0>, G, L>::run(
        device, context, queue, num_elements, "test_redadd", redadd_source, 0,
        useCoreSubgroups);
    error |= test<double, RED<double, 0>, G, L>::run(
        device, context, queue, num_elements, "test_redadd", redadd_source, 0,
        useCoreSubgroups);

    // error |= test<cl_half, RED<cl_half,1>, G, L>::run(device, context, queue,
    // num_elements, "test_redmax", redmax_source);
    error |= test<cl_uint, RED<cl_uint, 1>, G, L>::run(
        device, context, queue, num_elements, "test_redmax", redmax_source, 0,
        useCoreSubgroups);
    error |= test<cl_int, RED<cl_int, 1>, G, L>::run(
        device, context, queue, num_elements, "test_redmax", redmax_source, 0,
        useCoreSubgroups);
    error |= test<cl_ulong, RED<cl_ulong, 1>, G, L>::run(
        device, context, queue, num_elements, "test_redmax", redmax_source, 0,
        useCoreSubgroups);
    error |= test<cl_long, RED<cl_long, 1>, G, L>::run(
        device, context, queue, num_elements, "test_redmax", redmax_source, 0,
        useCoreSubgroups);
    error |= test<float, RED<float, 1>, G, L>::run(
        device, context, queue, num_elements, "test_redmax", redmax_source, 0,
        useCoreSubgroups);
    error |= test<double, RED<double, 1>, G, L>::run(
        device, context, queue, num_elements, "test_redmax", redmax_source, 0,
        useCoreSubgroups);

    // error |= test<cl_half, RED<cl_half,2>, G, L>::run(device, context, queue,
    // num_elements, "test_redmin", redmin_source);
    error |= test<cl_uint, RED<cl_uint, 2>, G, L>::run(
        device, context, queue, num_elements, "test_redmin", redmin_source, 0,
        useCoreSubgroups);
    error |= test<cl_int, RED<cl_int, 2>, G, L>::run(
        device, context, queue, num_elements, "test_redmin", redmin_source, 0,
        useCoreSubgroups);
    error |= test<cl_ulong, RED<cl_ulong, 2>, G, L>::run(
        device, context, queue, num_elements, "test_redmin", redmin_source, 0,
        useCoreSubgroups);
    error |= test<cl_long, RED<cl_long, 2>, G, L>::run(
        device, context, queue, num_elements, "test_redmin", redmin_source, 0,
        useCoreSubgroups);
    error |= test<float, RED<float, 2>, G, L>::run(
        device, context, queue, num_elements, "test_redmin", redmin_source, 0,
        useCoreSubgroups);
    error |= test<double, RED<double, 2>, G, L>::run(
        device, context, queue, num_elements, "test_redmin", redmin_source, 0,
        useCoreSubgroups);

    // error |= test<cl_half, SCIN<cl_half,0>, G, L>::run(device, context,
    // queue, num_elements, "test_scinadd", scinadd_source);
    error |= test<cl_uint, SCIN<cl_uint, 0>, G, L>::run(
        device, context, queue, num_elements, "test_scinadd", scinadd_source, 0,
        useCoreSubgroups);
    error |= test<cl_int, SCIN<cl_int, 0>, G, L>::run(
        device, context, queue, num_elements, "test_scinadd", scinadd_source, 0,
        useCoreSubgroups);
    error |= test<cl_ulong, SCIN<cl_ulong, 0>, G, L>::run(
        device, context, queue, num_elements, "test_scinadd", scinadd_source, 0,
        useCoreSubgroups);
    error |= test<cl_long, SCIN<cl_long, 0>, G, L>::run(
        device, context, queue, num_elements, "test_scinadd", scinadd_source, 0,
        useCoreSubgroups);
    error |= test<float, SCIN<float, 0>, G, L>::run(
        device, context, queue, num_elements, "test_scinadd", scinadd_source, 0,
        useCoreSubgroups);
    error |= test<double, SCIN<double, 0>, G, L>::run(
        device, context, queue, num_elements, "test_scinadd", scinadd_source, 0,
        useCoreSubgroups);

    // error |= test<cl_half, SCIN<cl_half,1>, G, L>::run(device, context,
    // queue, num_elements, "test_scinmax", scinmax_source);
    error |= test<cl_uint, SCIN<cl_uint, 1>, G, L>::run(
        device, context, queue, num_elements, "test_scinmax", scinmax_source, 0,
        useCoreSubgroups);
    error |= test<cl_int, SCIN<cl_int, 1>, G, L>::run(
        device, context, queue, num_elements, "test_scinmax", scinmax_source, 0,
        useCoreSubgroups);
    error |= test<cl_ulong, SCIN<cl_ulong, 1>, G, L>::run(
        device, context, queue, num_elements, "test_scinmax", scinmax_source, 0,
        useCoreSubgroups);
    error |= test<cl_long, SCIN<cl_long, 1>, G, L>::run(
        device, context, queue, num_elements, "test_scinmax", scinmax_source, 0,
        useCoreSubgroups);
    error |= test<float, SCIN<float, 1>, G, L>::run(
        device, context, queue, num_elements, "test_scinmax", scinmax_source, 0,
        useCoreSubgroups);
    error |= test<double, SCIN<double, 1>, G, L>::run(
        device, context, queue, num_elements, "test_scinmax", scinmax_source, 0,
        useCoreSubgroups);

    // error |= test<cl_half, SCIN<cl_half,2>, G, L>::run(device, context,
    // queue, num_elements, "test_scinmin", scinmin_source);
    error |= test<cl_uint, SCIN<cl_uint, 2>, G, L>::run(
        device, context, queue, num_elements, "test_scinmin", scinmin_source, 0,
        useCoreSubgroups);
    error |= test<cl_int, SCIN<cl_int, 2>, G, L>::run(
        device, context, queue, num_elements, "test_scinmin", scinmin_source, 0,
        useCoreSubgroups);
    error |= test<cl_ulong, SCIN<cl_ulong, 2>, G, L>::run(
        device, context, queue, num_elements, "test_scinmin", scinmin_source, 0,
        useCoreSubgroups);
    error |= test<cl_long, SCIN<cl_long, 2>, G, L>::run(
        device, context, queue, num_elements, "test_scinmin", scinmin_source, 0,
        useCoreSubgroups);
    error |= test<float, SCIN<float, 2>, G, L>::run(
        device, context, queue, num_elements, "test_scinmin", scinmin_source, 0,
        useCoreSubgroups);
    error |= test<double, SCIN<double, 2>, G, L>::run(
        device, context, queue, num_elements, "test_scinmin", scinmin_source, 0,
        useCoreSubgroups);

    // error |= test<cl_half, SCEX<cl_half,0>, G, L>::run(device, context,
    // queue, num_elements, "test_scexadd", scexadd_source);
    error |= test<cl_uint, SCEX<cl_uint, 0>, G, L>::run(
        device, context, queue, num_elements, "test_scexadd", scexadd_source, 0,
        useCoreSubgroups);
    error |= test<cl_int, SCEX<cl_int, 0>, G, L>::run(
        device, context, queue, num_elements, "test_scexadd", scexadd_source, 0,
        useCoreSubgroups);
    error |= test<cl_ulong, SCEX<cl_ulong, 0>, G, L>::run(
        device, context, queue, num_elements, "test_scexadd", scexadd_source, 0,
        useCoreSubgroups);
    error |= test<cl_long, SCEX<cl_long, 0>, G, L>::run(
        device, context, queue, num_elements, "test_scexadd", scexadd_source, 0,
        useCoreSubgroups);
    error |= test<float, SCEX<float, 0>, G, L>::run(
        device, context, queue, num_elements, "test_scexadd", scexadd_source, 0,
        useCoreSubgroups);
    error |= test<double, SCEX<double, 0>, G, L>::run(
        device, context, queue, num_elements, "test_scexadd", scexadd_source, 0,
        useCoreSubgroups);

    // error |= test<cl_half, SCEX<cl_half,1>, G, L>::run(device, context,
    // queue, num_elements, "test_scexmax", scexmax_source);
    error |= test<cl_uint, SCEX<cl_uint, 1>, G, L>::run(
        device, context, queue, num_elements, "test_scexmax", scexmax_source, 0,
        useCoreSubgroups);
    error |= test<cl_int, SCEX<cl_int, 1>, G, L>::run(
        device, context, queue, num_elements, "test_scexmax", scexmax_source, 0,
        useCoreSubgroups);
    error |= test<cl_ulong, SCEX<cl_ulong, 1>, G, L>::run(
        device, context, queue, num_elements, "test_scexmax", scexmax_source, 0,
        useCoreSubgroups);
    error |= test<cl_long, SCEX<cl_long, 1>, G, L>::run(
        device, context, queue, num_elements, "test_scexmax", scexmax_source, 0,
        useCoreSubgroups);
    error |= test<float, SCEX<float, 1>, G, L>::run(
        device, context, queue, num_elements, "test_scexmax", scexmax_source, 0,
        useCoreSubgroups);
    error |= test<double, SCEX<double, 1>, G, L>::run(
        device, context, queue, num_elements, "test_scexmax", scexmax_source, 0,
        useCoreSubgroups);

    // error |= test<cl_half, SCEX<cl_half,2>, G, L>::run(device, context,
    // queue, num_elements, "test_scexmin", scexmin_source);
    error |= test<cl_uint, SCEX<cl_uint, 2>, G, L>::run(
        device, context, queue, num_elements, "test_scexmin", scexmin_source, 0,
        useCoreSubgroups);
    error |= test<cl_int, SCEX<cl_int, 2>, G, L>::run(
        device, context, queue, num_elements, "test_scexmin", scexmin_source, 0,
        useCoreSubgroups);
    error |= test<cl_ulong, SCEX<cl_ulong, 2>, G, L>::run(
        device, context, queue, num_elements, "test_scexmin", scexmin_source, 0,
        useCoreSubgroups);
    error |= test<cl_long, SCEX<cl_long, 2>, G, L>::run(
        device, context, queue, num_elements, "test_scexmin", scexmin_source, 0,
        useCoreSubgroups);
    error |= test<float, SCEX<float, 2>, G, L>::run(
        device, context, queue, num_elements, "test_scexmin", scexmin_source, 0,
        useCoreSubgroups);
    error |= test<double, SCEX<double, 2>, G, L>::run(
        device, context, queue, num_elements, "test_scexmin", scexmin_source, 0,
        useCoreSubgroups);
    return error;
}

int test_work_group_functions_core(cl_device_id device, cl_context context,
                                   cl_command_queue queue, int num_elements)
{
    return test_work_group_functions(device, context, queue, num_elements,
                                     true);
}

int test_work_group_functions_ext(cl_device_id device, cl_context context,
                                  cl_command_queue queue, int num_elements)
{
    bool hasExtension = is_extension_available(device, "cl_khr_subgroups");

    if (!hasExtension)
    {
        log_info(
            "Device does not support 'cl_khr_subgroups'. Skipping the test.\n");
        return TEST_SKIPPED_ITSELF;
    }
    return test_work_group_functions(device, context, queue, num_elements,
                                     false);
}
