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
#include "procs.h"
#include "subhelpers.h"
#include "harness/typeWrappers.h"
#include "workgroup_common_templates.h"


// DESCRIPTION:
// Test for scan inclusive non uniform functions
// Which: 0 - add, 1 - max, 2 - min, 3 - mul, 4 - and, 5 - or, 6 - xor, 7 -
// logical and, 8 - logical or, 9 - logical xor
template <typename Ty, int Which> struct SCIN_NU
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

        log_info("  sub_group_non_uniform_scan_inclusive_%s(%s)...\n",
                 operation_names[Which], TypeManager<Ty>::name());

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
                tr = TypeManager<Ty>::identify_limits(Which);
                for (i = 0; i < n && i < NON_UNIFORM; ++i)
                { // inside the subgroup
                    tr = OPERATION<Ty, Which>::calculate(tr, mx[ii + i]);
                    rr = my[ii + i];
                    if (!compare(rr, tr))
                    {
                        log_error("ERROR: "
                                  "sub_group_non_uniform_scan_inclusive_%s(%s) "
                                  "mismatch for local id %d in sub group %d in "
                                  "group %d\n",
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


// Scan Exclusive non uniform functions
// Which: 0 - add, 1 - max, 2 - min, 3 - mul, 4 - and, 5 - or, 6 - xor, 7 -
// logical and, 8 - logical or, 9 - logical xor
template <typename Ty, int Which> struct SCEX_NU
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

        log_info("  sub_group_non_uniform_scan_exclusive_%s(%s)...\n",
                 operation_names[Which], TypeManager<Ty>::name());

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
                tr = TypeManager<Ty>::identify_limits(Which);
                for (i = 0; i < n && i < NON_UNIFORM; ++i)
                { // inside the subgroup
                    rr = my[ii + i];

                    if (!compare(rr, tr))
                    {
                        log_error("ERROR: "
                                  "sub_group_non_uniform_scan_exclusive_%s(%s) "
                                  "mismatch for local id %d in sub group %d in "
                                  "group %d\n",
                                  operation_names[Which],
                                  TypeManager<Ty>::name(), i, j, k);
                        return -1;
                    }

                    tr = OPERATION<Ty, Which>::calculate(tr, mx[ii + i]);
                }
            }
            x += nw;
            y += nw;
            m += 4 * nw;
        }

        return 0;
    }
};

// DESCRIPTION:
// Test for reduce non uniform functions
// Which: 0 - add, 1 - max, 2 - min, 3 - mul, 4 - and, 5 - or, 6 - xor, 7 -
// logical and, 8 - logical or, 9 - logical xor
template <typename Ty, int Which> struct RED_NU
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

        log_info("  sub_group_non_uniform_reduce_%s(%s)...\n",
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
                if (n > NON_UNIFORM) n = NON_UNIFORM;

                // Compute target
                tr = mx[ii];
                for (i = 1; i < n; ++i)
                {
                    tr = OPERATION<Ty, Which>::calculate(tr, mx[ii + i]);
                }
                // Check result
                for (i = 0; i < n; ++i)
                {
                    rr = my[ii + i];
                    if (!compare(rr, tr))
                    {
                        log_error("ERROR: sub_group_non_uniform_reduce_%s(%s) "
                                  "mismatch for local id %d in sub group %d in "
                                  "group %d\n",
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

static const char *scinadd_non_uniform_source =
    "__kernel void test_scinadd_non_uniform(const __global Type *in, __global "
    "int4 *xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    " if (xy[gid].x < NON_UNIFORM) {"
    "    out[gid] = sub_group_non_uniform_scan_inclusive_add(in[gid]);\n"
    " }"
    //"printf(\"gid = %d, sub group local id = %d, sub group id = %d, x form in
    //= %d, new_set = %d, out[gid] = %d\\n\",gid,xy[gid].x, xy[gid].y, x,
    // xy[gid].z, out[gid]);"
    "}\n";
static const char *scinmax_non_uniform_source =
    "__kernel void test_scinmax_non_uniform(const __global Type *in, __global "
    "int4 *xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    " if (xy[gid].x < NON_UNIFORM) {"
    "    out[gid] = sub_group_non_uniform_scan_inclusive_max(in[gid]);\n"
    " }"
    "}\n";
static const char *scinmin_non_uniform_source =
    "__kernel void test_scinmin_non_uniform(const __global Type *in, __global "
    "int4 *xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    " if (xy[gid].x < NON_UNIFORM) {"
    "    out[gid] = sub_group_non_uniform_scan_inclusive_min(in[gid]);\n"
    " }"
    "}\n";
static const char *scinmul_non_uniform_source =
    "__kernel void test_scinmul_non_uniform(const __global Type *in, __global "
    "int4 *xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    " if (xy[gid].x < NON_UNIFORM) {"
    "    out[gid] = sub_group_non_uniform_scan_inclusive_mul(in[gid]);\n"
    " }"
    "}\n";
static const char *scinand_non_uniform_source =
    "__kernel void test_scinand_non_uniform(const __global Type *in, __global "
    "int4 *xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    " if (xy[gid].x < NON_UNIFORM) {"
    "    out[gid] = sub_group_non_uniform_scan_inclusive_and(in[gid]);\n"
    " }"
    "}\n";
static const char *scinor_non_uniform_source =
    "__kernel void test_scinor_non_uniform(const __global Type *in, __global "
    "int4 *xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    " if (xy[gid].x < NON_UNIFORM) {"
    "    out[gid] = sub_group_non_uniform_scan_inclusive_or(in[gid]);\n"
    " }"
    "}\n";
static const char *scinxor_non_uniform_source =
    "__kernel void test_scinxor_non_uniform(const __global Type *in, __global "
    "int4 *xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    " if (xy[gid].x < NON_UNIFORM) {"
    "    out[gid] = sub_group_non_uniform_scan_inclusive_xor(in[gid]);\n"
    " }"
    "}\n";
static const char *scinand_non_uniform_logical_source =
    "__kernel void test_scinand_non_uniform_logical(const __global Type *in, "
    "__global int4 *xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    " if (xy[gid].x < NON_UNIFORM) {"
    "    out[gid] = "
    "sub_group_non_uniform_scan_inclusive_logical_and(in[gid]);\n"
    " }"
    "}\n";
static const char *scinor_non_uniform_logical_source =
    "__kernel void test_scinor_non_uniform_logical(const __global Type *in, "
    "__global int4 *xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    " if (xy[gid].x < NON_UNIFORM) {"
    "    out[gid] = sub_group_non_uniform_scan_inclusive_logical_or(in[gid]);\n"
    " }"
    "}\n";
static const char *scinxor_non_uniform_logical_source =
    "__kernel void test_scinxor_non_uniform_logical(const __global Type *in, "
    "__global int4 *xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    " if (xy[gid].x < NON_UNIFORM) {"
    "    out[gid] = "
    "sub_group_non_uniform_scan_inclusive_logical_xor(in[gid]);\n"
    " }"
    "}\n";

static const char *scexadd_non_uniform_source =
    "__kernel void test_scexadd_non_uniform(const __global Type *in, __global "
    "int4 *xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    " if (xy[gid].x < NON_UNIFORM) {"
    "    out[gid] = sub_group_non_uniform_scan_exclusive_add(in[gid]);\n"
    " }"
    //"printf(\"gid = %d, sub group local id = %d, sub group id = %d, x form in
    //= %d, new_set = %d, out[gid] = %d , x = %d\\n\",gid,xy[gid].x, xy[gid].y,
    // x, xy[gid].z, out[gid]);"
    "}\n";

static const char *scexmax_non_uniform_source =
    "__kernel void test_scexmax_non_uniform(const __global Type *in, __global "
    "int4 *xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    " if (xy[gid].x < NON_UNIFORM) {"
    "    out[gid] = sub_group_non_uniform_scan_exclusive_max(in[gid]);\n"
    " }"
    "}\n";

static const char *scexmin_non_uniform_source =
    "__kernel void test_scexmin_non_uniform(const __global Type *in, __global "
    "int4 *xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    " if (xy[gid].x < NON_UNIFORM) {"
    "    out[gid] = sub_group_non_uniform_scan_exclusive_min(in[gid]);\n"
    " }"
    "}\n";

static const char *scexmul_non_uniform_source =
    "__kernel void test_scexmul_non_uniform(const __global Type *in, __global "
    "int4 *xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    " if (xy[gid].x < NON_UNIFORM) {"
    "    out[gid] = sub_group_non_uniform_scan_exclusive_mul(in[gid]);\n"
    " }"
    "}\n";

static const char *scexand_non_uniform_source =
    "__kernel void test_scexand_non_uniform(const __global Type *in, __global "
    "int4 *xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    " if (xy[gid].x < NON_UNIFORM) {"
    "    out[gid] = sub_group_non_uniform_scan_exclusive_and(in[gid]);\n"
    " }"
    "}\n";

static const char *scexor_non_uniform_source =
    "__kernel void test_scexor_non_uniform(const __global Type *in, __global "
    "int4 *xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    " if (xy[gid].x < NON_UNIFORM) {"
    "    out[gid] = sub_group_non_uniform_scan_exclusive_or(in[gid]);\n"
    " }"
    "}\n";

static const char *scexxor_non_uniform_source =
    "__kernel void test_scexxor_non_uniform(const __global Type *in, __global "
    "int4 *xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    " if (xy[gid].x < NON_UNIFORM) {"
    "    out[gid] = sub_group_non_uniform_scan_exclusive_xor(in[gid]);\n"
    " }"
    "}\n";

static const char *scexand_non_uniform_logical_source =
    "__kernel void test_scexand_non_uniform_logical(const __global Type *in, "
    "__global int4 *xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    " if (xy[gid].x < NON_UNIFORM) {"
    "    out[gid] = "
    "sub_group_non_uniform_scan_exclusive_logical_and(in[gid]);\n"
    " }"
    "}\n";

static const char *scexor_non_uniform_logical_source =
    "__kernel void test_scexor_non_uniform_logical(const __global Type *in, "
    "__global int4 *xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    " if (xy[gid].x < NON_UNIFORM) {"
    "    out[gid] = sub_group_non_uniform_scan_exclusive_logical_or(in[gid]);\n"
    " }"
    "}\n";

static const char *scexxor_non_uniform_logical_source =
    "__kernel void test_scexxor_non_uniform_logical(const __global Type *in, "
    "__global int4 *xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    " if (xy[gid].x < NON_UNIFORM) {"
    "    out[gid] = "
    "sub_group_non_uniform_scan_exclusive_logical_xor(in[gid]);\n"
    " }"
    "}\n";

static const char *redadd_non_uniform_source =
    "__kernel void test_redadd_non_uniform(const __global Type *in, __global "
    "int4 *xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    " if (xy[gid].x < NON_UNIFORM) {"
    "    out[gid] = sub_group_non_uniform_reduce_add(in[gid]);\n"
    " }"
    //"printf(\"gid = %d, sub group local id = %d, sub group id = %d, x form in
    //= %d, new_set = %d, out[gid] = %d\\n\",gid,xy[gid].x, xy[gid].y, x,
    // xy[gid].z, out[gid]);"
    "}\n";

static const char *redmax_non_uniform_source =
    "__kernel void test_redmax_non_uniform(const __global Type *in, __global "
    "int4 *xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    " if (xy[gid].x < NON_UNIFORM) {"
    "    out[gid] = sub_group_non_uniform_reduce_max(in[gid]);\n"
    " }"
    "}\n";

static const char *redmin_non_uniform_source =
    "__kernel void test_redmin_non_uniform(const __global Type *in, __global "
    "int4 *xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    " if (xy[gid].x < NON_UNIFORM) {"
    "    out[gid] = sub_group_non_uniform_reduce_min(in[gid]);\n"
    " }"
    "}\n";

static const char *redmul_non_uniform_source =
    "__kernel void test_redmul_non_uniform(const __global Type *in, __global "
    "int4 *xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    " if (xy[gid].x < NON_UNIFORM) {"
    "    out[gid] = sub_group_non_uniform_reduce_mul(in[gid]);\n"
    " }"
    "}\n";

static const char *redand_non_uniform_source =
    "__kernel void test_redand_non_uniform(const __global Type *in, __global "
    "int4 *xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    " if (xy[gid].x < NON_UNIFORM) {"
    "    out[gid] = sub_group_non_uniform_reduce_and(in[gid]);\n"
    " }"
    "}\n";

static const char *redor_non_uniform_source =
    "__kernel void test_redor_non_uniform(const __global Type *in, __global "
    "int4 *xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    " if (xy[gid].x < NON_UNIFORM) {"
    "    out[gid] = sub_group_non_uniform_reduce_or(in[gid]);\n"
    " }"
    "}\n";

static const char *redxor_non_uniform_source =
    "__kernel void test_redxor_non_uniform(const __global Type *in, __global "
    "int4 *xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    " if (xy[gid].x < NON_UNIFORM) {"
    "    out[gid] = sub_group_non_uniform_reduce_xor(in[gid]);\n"
    " }"
    "}\n";

static const char *redand_non_uniform_logical_source =
    "__kernel void test_redand_non_uniform_logical(const __global Type *in, "
    "__global int4 *xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    " if (xy[gid].x < NON_UNIFORM) {"
    "    out[gid] = sub_group_non_uniform_reduce_logical_and(in[gid]);\n"
    " }"
    "}\n";

static const char *redor_non_uniform_logical_source =
    "__kernel void test_redor_non_uniform_logical(const __global Type *in, "
    "__global int4 *xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    " if (xy[gid].x < NON_UNIFORM) {"
    "    out[gid] = sub_group_non_uniform_reduce_logical_or(in[gid]);\n"
    " }"
    "}\n";

static const char *redxor_non_uniform_logical_source =
    "__kernel void test_redxor_non_uniform_logical(const __global Type *in, "
    "__global int4 *xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    " if (xy[gid].x < NON_UNIFORM) {"
    "    out[gid] = sub_group_non_uniform_reduce_logical_xor(in[gid]);\n"
    " }"
    "}\n";


struct run_for_type
{
    run_for_type(cl_device_id device, cl_context context,
                 cl_command_queue queue, int num_elements,
                 bool useCoreSubgroups,
                 std::vector<std::string> required_extensions = {})
    {
        device_ = device;
        context_ = context;
        queue_ = queue;
        num_elements_ = num_elements;
        useCoreSubgroups_ = useCoreSubgroups;
        required_extensions_ = required_extensions;
    }

    template <typename T> cl_int run_nu_scin_scex_red_not_logical_funcs()
    {
        cl_int error;
        error = test<T, SCIN_NU<T, 0>, G, L>::run(
            device_, context_, queue_, num_elements_,
            "test_scinadd_non_uniform", scinadd_non_uniform_source, 0,
            useCoreSubgroups_, required_extensions_);

        error |= test<T, SCIN_NU<T, 1>, G, L>::run(
            device_, context_, queue_, num_elements_,
            "test_scinmax_non_uniform", scinmax_non_uniform_source, 0,
            useCoreSubgroups_, required_extensions_);

        error |= test<T, SCIN_NU<T, 2>, G, L>::run(
            device_, context_, queue_, num_elements_,
            "test_scinmin_non_uniform", scinmin_non_uniform_source, 0,
            useCoreSubgroups_, required_extensions_);

        error |= test<T, SCIN_NU<T, 3>, G, L>::run(
            device_, context_, queue_, num_elements_,
            "test_scinmul_non_uniform", scinmul_non_uniform_source, 0,
            useCoreSubgroups_, required_extensions_);

        error |= test<T, SCEX_NU<T, 0>, G, L>::run(
            device_, context_, queue_, num_elements_,
            "test_scexadd_non_uniform", scexadd_non_uniform_source, 0,
            useCoreSubgroups_, required_extensions_);

        error |= test<T, SCEX_NU<T, 1>, G, L>::run(
            device_, context_, queue_, num_elements_,
            "test_scexmax_non_uniform", scexmax_non_uniform_source, 0,
            useCoreSubgroups_, required_extensions_);

        error |= test<T, SCEX_NU<T, 2>, G, L>::run(
            device_, context_, queue_, num_elements_,
            "test_scexmin_non_uniform", scexmin_non_uniform_source, 0,
            useCoreSubgroups_, required_extensions_);

        error |= test<T, SCEX_NU<T, 3>, G, L>::run(
            device_, context_, queue_, num_elements_,
            "test_scexmul_non_uniform", scexmul_non_uniform_source, 0,
            useCoreSubgroups_, required_extensions_);

        error |= test<T, RED_NU<T, 0>, G, L>::run(
            device_, context_, queue_, num_elements_, "test_redadd_non_uniform",
            redadd_non_uniform_source, 0, useCoreSubgroups_,
            required_extensions_);

        error |= test<T, RED_NU<T, 1>, G, L>::run(
            device_, context_, queue_, num_elements_, "test_redmax_non_uniform",
            redmax_non_uniform_source, 0, useCoreSubgroups_,
            required_extensions_);

        error |= test<T, RED_NU<T, 2>, G, L>::run(
            device_, context_, queue_, num_elements_, "test_redmin_non_uniform",
            redmin_non_uniform_source, 0, useCoreSubgroups_,
            required_extensions_);

        error |= test<T, RED_NU<T, 3>, G, L>::run(
            device_, context_, queue_, num_elements_, "test_redmul_non_uniform",
            redmul_non_uniform_source, 0, useCoreSubgroups_,
            required_extensions_);

        return error;
    }

    template <typename T> cl_int run_nu_scin_scex_red_all_funcs()
    {
        cl_int error;
        error = run_nu_scin_scex_red_not_logical_funcs<T>();
        error |= test<T, SCIN_NU<T, 4>, G, L>::run(
            device_, context_, queue_, num_elements_,
            "test_scinand_non_uniform", scinand_non_uniform_source, 0,
            useCoreSubgroups_, required_extensions_);

        error |= test<T, SCIN_NU<T, 5>, G, L>::run(
            device_, context_, queue_, num_elements_, "test_scinor_non_uniform",
            scinor_non_uniform_source, 0, useCoreSubgroups_,
            required_extensions_);

        error |= test<T, SCIN_NU<T, 6>, G, L>::run(
            device_, context_, queue_, num_elements_,
            "test_scinxor_non_uniform", scinxor_non_uniform_source, 0,
            useCoreSubgroups_, required_extensions_);

        error |= test<T, SCEX_NU<T, 4>, G, L>::run(
            device_, context_, queue_, num_elements_,
            "test_scexand_non_uniform", scexand_non_uniform_source, 0,
            useCoreSubgroups_, required_extensions_);

        error |= test<T, SCEX_NU<T, 5>, G, L>::run(
            device_, context_, queue_, num_elements_, "test_scexor_non_uniform",
            scexor_non_uniform_source, 0, useCoreSubgroups_,
            required_extensions_);

        error |= test<T, SCEX_NU<T, 6>, G, L>::run(
            device_, context_, queue_, num_elements_,
            "test_scexxor_non_uniform", scexxor_non_uniform_source, 0,
            useCoreSubgroups_, required_extensions_);

        error |= test<T, RED_NU<T, 4>, G, L>::run(
            device_, context_, queue_, num_elements_, "test_redand_non_uniform",
            redand_non_uniform_source, 0, useCoreSubgroups_,
            required_extensions_);

        error |= test<T, RED_NU<T, 5>, G, L>::run(
            device_, context_, queue_, num_elements_, "test_redor_non_uniform",
            redor_non_uniform_source, 0, useCoreSubgroups_,
            required_extensions_);

        error |= test<T, RED_NU<T, 6>, G, L>::run(
            device_, context_, queue_, num_elements_, "test_redxor_non_uniform",
            redxor_non_uniform_source, 0, useCoreSubgroups_,
            required_extensions_);
        return error;
    }

    cl_int run_nu_logical()
    {
        cl_int error;
        error = test<cl_int, SCIN_NU<cl_int, 7>, G, L>::run(
            device_, context_, queue_, num_elements_,
            "test_scinand_non_uniform_logical",
            scinand_non_uniform_logical_source, 0, useCoreSubgroups_,
            required_extensions_);

        error |= test<cl_int, SCIN_NU<cl_int, 8>, G, L>::run(
            device_, context_, queue_, num_elements_,
            "test_scinor_non_uniform_logical",
            scinor_non_uniform_logical_source, 0, useCoreSubgroups_,
            required_extensions_);

        error |= test<cl_int, SCIN_NU<cl_int, 9>, G, L>::run(
            device_, context_, queue_, num_elements_,
            "test_scinxor_non_uniform_logical",
            scinxor_non_uniform_logical_source, 0, useCoreSubgroups_,
            required_extensions_);

        error |= test<cl_int, SCEX_NU<cl_int, 7>, G, L>::run(
            device_, context_, queue_, num_elements_,
            "test_scexand_non_uniform_logical",
            scexand_non_uniform_logical_source, 0, useCoreSubgroups_,
            required_extensions_);

        error |= test<cl_int, SCEX_NU<cl_int, 8>, G, L>::run(
            device_, context_, queue_, num_elements_,
            "test_scexor_non_uniform_logical",
            scexor_non_uniform_logical_source, 0, useCoreSubgroups_,
            required_extensions_);

        error |= test<cl_int, SCEX_NU<cl_int, 9>, G, L>::run(
            device_, context_, queue_, num_elements_,
            "test_scexxor_non_uniform_logical",
            scexxor_non_uniform_logical_source, 0, useCoreSubgroups_,
            required_extensions_);

        error |= test<cl_int, RED_NU<cl_int, 7>, G, L>::run(
            device_, context_, queue_, num_elements_,
            "test_redand_non_uniform_logical",
            redand_non_uniform_logical_source, 0, useCoreSubgroups_,
            required_extensions_);

        error |= test<cl_int, RED_NU<cl_int, 8>, G, L>::run(
            device_, context_, queue_, num_elements_,
            "test_redor_non_uniform_logical", redor_non_uniform_logical_source,
            0, useCoreSubgroups_, required_extensions_);

        error |= test<cl_int, RED_NU<cl_int, 9>, G, L>::run(
            device_, context_, queue_, num_elements_,
            "test_redxor_non_uniform_logical",
            redxor_non_uniform_logical_source, 0, useCoreSubgroups_,
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

int test_work_group_functions_non_uniform_arithmetic(cl_device_id device,
                                                     cl_context context,
                                                     cl_command_queue queue,
                                                     int num_elements)
{
    int error;
    std::vector<std::string> required_extensions;
    required_extensions = { "cl_khr_subgroup_non_uniform_arithmetic" };
    run_for_type rft(device, context, queue, num_elements, true,
                     required_extensions);

    error = rft.run_nu_scin_scex_red_all_funcs<cl_int>();
    error |= rft.run_nu_scin_scex_red_all_funcs<cl_uint>();
    error |= rft.run_nu_scin_scex_red_all_funcs<cl_long>();
    error |= rft.run_nu_scin_scex_red_all_funcs<cl_ulong>();
    error |= rft.run_nu_scin_scex_red_all_funcs<cl_short>();
    error |= rft.run_nu_scin_scex_red_all_funcs<cl_ushort>();
    error |= rft.run_nu_scin_scex_red_all_funcs<cl_char>();
    error |= rft.run_nu_scin_scex_red_all_funcs<cl_uchar>();
    error |= rft.run_nu_scin_scex_red_not_logical_funcs<cl_float>();
    error |= rft.run_nu_scin_scex_red_not_logical_funcs<cl_double>();
    error |= rft.run_nu_scin_scex_red_not_logical_funcs<subgroups::cl_half>();
    error |= rft.run_nu_logical();
    return error;
}