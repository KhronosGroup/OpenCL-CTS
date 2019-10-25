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

static const char * any_source =
"__kernel void test_any(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    out[gid] = sub_group_any(in[gid]);\n"
"}\n";

static const char * all_source =
"__kernel void test_all(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    out[gid] = sub_group_all(in[gid]);\n"
"}\n";

static const char * bcast_source =
"__kernel void test_bcast(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    Type x = in[gid];\n"
"    out[gid] = sub_group_broadcast(x, xy[gid].z);\n"
"}\n";

static const char * redadd_source =
"__kernel void test_redadd(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    out[gid] = sub_group_reduce_add(in[gid]);\n"
"}\n";

static const char * redmax_source =
"__kernel void test_redmax(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    out[gid] = sub_group_reduce_max(in[gid]);\n"
"}\n";

static const char * redmin_source =
"__kernel void test_redmin(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    out[gid] = sub_group_reduce_min(in[gid]);\n"
"}\n";

static const char * scinadd_source =
"__kernel void test_scinadd(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    out[gid] = sub_group_scan_inclusive_add(in[gid]);\n"
"}\n";

static const char * scinmax_source =
"__kernel void test_scinmax(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    out[gid] = sub_group_scan_inclusive_max(in[gid]);\n"
"}\n";

static const char * scinmin_source =
"__kernel void test_scinmin(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    out[gid] = sub_group_scan_inclusive_min(in[gid]);\n"
"}\n";

static const char * scexadd_source =
"__kernel void test_scexadd(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    out[gid] = sub_group_scan_exclusive_add(in[gid]);\n"
"}\n";

static const char * scexmax_source =
"__kernel void test_scexmax(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    out[gid] = sub_group_scan_exclusive_max(in[gid]);\n"
"}\n";

static const char * scexmin_source =
"__kernel void test_scexmin(const __global Type *in, __global int4 *xy, __global Type *out)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    XY(xy,gid);\n"
"    out[gid] = sub_group_scan_exclusive_min(in[gid]);\n"
"}\n";

// These need to stay in sync with the kernel source below
#define NUM_LOC 49
#define INST_LOC_MASK 0x7f
#define INST_OP_SHIFT 0
#define INST_OP_MASK 0xf
#define INST_LOC_SHIFT 4
#define INST_VAL_SHIFT 12
#define INST_VAL_MASK 0x7ffff
#define INST_END 0x0
#define INST_STORE 0x1
#define INST_WAIT 0x2
#define INST_COUNT 0x3

static const char * ifp_source =
"#define NUM_LOC 49\n"
"#define INST_LOC_MASK 0x7f\n"
"#define INST_OP_SHIFT 0\n"
"#define INST_OP_MASK 0xf\n"
"#define INST_LOC_SHIFT 4\n"
"#define INST_VAL_SHIFT 12\n"
"#define INST_VAL_MASK 0x7ffff\n"
"#define INST_END 0x0\n"
"#define INST_STORE 0x1\n"
"#define INST_WAIT 0x2\n"
"#define INST_COUNT 0x3\n"
"\n"
"__kernel void\n"
"test_ifp(const __global int *in, __global int4 *xy, __global int *out)\n"
"{\n"
"    __local atomic_int loc[NUM_LOC];\n"
"\n"
"    // Don't run if there is only one sub group\n"
"    if (get_num_sub_groups() == 1)\n"
"        return;\n"
"\n"
"    // First initialize loc[]\n"
"    int lid = (int)get_local_id(0);\n"
"\n"
"    if (lid < NUM_LOC)\n"
"        atomic_init(loc+lid, 0);\n"
"\n"
"    work_group_barrier(CLK_LOCAL_MEM_FENCE);\n"
"\n"
"    // Compute pointer to this sub group's \"instructions\"\n"
"    const __global int *pc = in +\n"
"        ((int)get_group_id(0)*(int)get_enqueued_num_sub_groups() +\n"
"         (int)get_sub_group_id()) *\n"
"        (NUM_LOC+1);\n"
"\n"
"    // Set up to \"run\"\n"
"    bool ok = (int)get_sub_group_local_id() == 0;\n"
"    bool run = true;\n"
"\n"
"    while (run) {\n"
"        int inst = *pc++;\n"
"        int iop = (inst >> INST_OP_SHIFT) & INST_OP_MASK;\n"
"        int iloc = (inst >> INST_LOC_SHIFT) & INST_LOC_MASK;\n"
"        int ival = (inst >> INST_VAL_SHIFT) & INST_VAL_MASK;\n"
"\n"
"        switch (iop) {\n"
"        case INST_STORE:\n"
"            if (ok)\n"
"                atomic_store(loc+iloc, ival);\n"
"            break;\n"
"        case INST_WAIT:\n"
"            if (ok) {\n"
"                while (atomic_load(loc+iloc) != ival)\n"
"                    ;\n"
"            }\n"
"            break;\n"
"        case INST_COUNT:\n"
"            if (ok) {\n"
"                int i;\n"
"                for (i=0;i<ival;++i)\n"
"                    atomic_fetch_add(loc+iloc, 1);\n"
"            }\n"
"            break;\n"
"        case INST_END:\n"
"            run = false;\n"
"            break;\n"
"        }\n"
"\n"
"        sub_group_barrier(CLK_LOCAL_MEM_FENCE);\n"
"    }\n"
"\n"
"    work_group_barrier(CLK_LOCAL_MEM_FENCE);\n"
"\n"
"    // Save this group's result\n"
"    __global int *op = out + (int)get_group_id(0)*NUM_LOC;\n"
"    if (lid < NUM_LOC)\n"
"        op[lid] = atomic_load(loc+lid);\n"
"}\n";

// Any/All test functions
template <int Which>
struct AA {
    static void gen(cl_int *x, cl_int *t, cl_int *m, int ns, int nw, int ng)
    {
        int i, ii, j, k, n;
        int nj = (nw + ns - 1)/ns;
        int e;

        ii = 0;
        for (k=0; k<ng; ++k) {
            for (j=0; j<nj; ++j) {
                ii = j*ns;
                n = ii + ns > nw ? nw - ii : ns;
                e = (int)(genrand_int32(gMTdata) % 3);

                // Initialize data matrix indexed by local id and sub group id
                switch (e) {
                case 0:
                    memset(&t[ii], 0, n*sizeof(cl_int));
                    break;
                case 1:
                    memset(&t[ii], 0, n*sizeof(cl_int));
                    i = (int)(genrand_int32(gMTdata) % (cl_uint)n);
                    t[ii + i] = 41;
                    break;
                case 2:
                    memset(&t[ii], 0xff, n*sizeof(cl_int));
                    break;
                }
            }

            // Now map into work group using map from device
            for (j=0;j<nw;++j) {
                i = m[4*j+1]*ns + m[4*j];
                x[j] = t[i];
            }

            x += nw;
        m += 4*nw;
        }
    }

    static int chk(cl_int *x, cl_int *y, cl_int *mx, cl_int *my, cl_int *m, int ns, int nw, int ng)
    {
        int ii, i, j, k, n;
        int nj = (nw + ns - 1)/ns;
        cl_int taa, raa;

        log_info("  sub_group_%s...\n", Which == 0 ? "any" : "all");

        for (k=0; k<ng; ++k) {
            // Map to array indexed to array indexed by local ID and sub group
            for (j=0; j<nw; ++j) {
                i = m[4*j+1]*ns + m[4*j];
                mx[i] = x[j];
                my[i] = y[j];
            }

            for (j=0; j<nj; ++j) {
                ii = j*ns;
                n = ii + ns > nw ? nw - ii : ns;

                // Compute target
                if (Which == 0) {
                    taa = 0;
                    for (i=0; i<n; ++i)
                        taa |=  mx[ii + i] != 0;
                } else {
                    taa = 1;
                    for (i=0; i<n; ++i)
                        taa &=  mx[ii + i] != 0;
                }

                // Check result
                for (i=0; i<n; ++i) {
                    raa = my[ii+i] != 0;
                    if (raa != taa) {
                        log_error("ERROR: sub_group_%s mismatch for local id %d in sub group %d in group %d\n",
                                   Which == 0 ? "any" : "all", i, j, k);
                        return -1;
                    }
                }
            }

            x += nw;
            y += nw;
            m += 4*nw;
        }

        return 0;
    }
};

// Reduce functions
template <typename Ty, int Which>
struct RED {
    static void gen(Ty *x, Ty *t, cl_int *m, int ns, int nw, int ng)
    {
        int i, ii, j, k, n;
        int nj = (nw + ns - 1)/ns;

        ii = 0;
        for (k=0; k<ng; ++k) {
            for (j=0; j<nj; ++j) {
                ii = j*ns;
                n = ii + ns > nw ? nw - ii : ns;

                for (i=0; i<n; ++i)
                    t[ii+i] = (Ty)((int)(genrand_int32(gMTdata) & 0x7fffffff) % ns + 1);
            }

            // Now map into work group using map from device
            for (j=0;j<nw;++j) {
                i = m[4*j+1]*ns + m[4*j];
                x[j] = t[i];
            }

            x += nw;
        m += 4*nw;
        }
    }

    static int chk(Ty *x, Ty *y, Ty *mx, Ty *my, cl_int *m, int ns, int nw, int ng)
    {
        int ii, i, j, k, n;
        int nj = (nw + ns - 1)/ns;
        Ty tr, rr;

        log_info("  sub_group_reduce_%s(%s)...\n", Which == 0 ? "add" : (Which == 1 ? "max" : "min"), TypeName<Ty>::val());

        for (k=0; k<ng; ++k) {
            // Map to array indexed to array indexed by local ID and sub group
            for (j=0; j<nw; ++j) {
                i = m[4*j+1]*ns + m[4*j];
                mx[i] = x[j];
                my[i] = y[j];
            }

            for (j=0; j<nj; ++j) {
                ii = j*ns;
                n = ii + ns > nw ? nw - ii : ns;

                // Compute target
                if (Which == 0) {
                    // add
                    tr = mx[ii];
                    for (i=1; i<n; ++i)
                        tr +=  mx[ii + i];
                } else if (Which == 1) {
                    // max
                    tr = mx[ii];
                    for (i=1; i<n; ++i)
                        tr = tr > mx[ii + i] ? tr : mx[ii + i];
                } else if (Which == 2) {
                    // min
                    tr = mx[ii];
                    for (i=1; i<n; ++i)
                        tr = tr > mx[ii + i] ? mx[ii + i] : tr;
                }

                // Check result
                for (i=0; i<n; ++i) {
                    rr = my[ii+i];
                    if (rr != tr) {
                        log_error("ERROR: sub_group_reduce_%s(%s) mismatch for local id %d in sub group %d in group %d\n",
                                   Which == 0 ? "add" : (Which == 1 ? "max" : "min"), TypeName<Ty>::val(), i, j, k);
                        return -1;
                    }
                }
            }

            x += nw;
            y += nw;
            m += 4*nw;
        }

        return 0;
    }
};

// Scan Inclusive functions
template <typename Ty, int Which>
struct SCIN {
    static void gen(Ty *x, Ty *t, cl_int *m, int ns, int nw, int ng)
    {
        int i, ii, j, k, n;
        int nj = (nw + ns - 1)/ns;

        ii = 0;
        for (k=0; k<ng; ++k) {
            for (j=0; j<nj; ++j) {
                ii = j*ns;
                n = ii + ns > nw ? nw - ii : ns;

                for (i=0; i<n; ++i)
                    // t[ii+i] = (Ty)((int)(genrand_int32(gMTdata) & 0x7fffffff) % ns + 1);
                    t[ii+i] = (Ty)i;
            }

            // Now map into work group using map from device
            for (j=0;j<nw;++j) {
                i = m[4*j+1]*ns + m[4*j];
                x[j] = t[i];
            }

            x += nw;
        m += 4*nw;
        }
    }

    static int chk(Ty *x, Ty *y, Ty *mx, Ty *my, cl_int *m, int ns, int nw, int ng)
    {
        int ii, i, j, k, n;
        int nj = (nw + ns - 1)/ns;
        Ty tr, rr;

        log_info("  sub_group_scan_inclusive_%s(%s)...\n",  Which == 0 ? "add" : (Which == 1 ? "max" : "min"), TypeName<Ty>::val());

        for (k=0; k<ng; ++k) {
            // Map to array indexed to array indexed by local ID and sub group
            for (j=0; j<nw; ++j) {
                i = m[4*j+1]*ns + m[4*j];
                mx[i] = x[j];
                my[i] = y[j];
            }

            for (j=0; j<nj; ++j) {
                ii = j*ns;
                n = ii + ns > nw ? nw - ii : ns;

                // Check result
                for (i=0; i<n; ++i) {
                    if (Which == 0) {
                        tr = i == 0 ? mx[ii] : tr + mx[ii + i];
                    } else if (Which == 1) {
                        tr = i == 0 ? mx[ii] : (tr > mx[ii + i] ? tr : mx[ii + i]);
                    } else {
                        tr = i == 0 ? mx[ii] : (tr > mx[ii + i] ? mx[ii + i] : tr);
                    }

                    rr = my[ii+i];
                    if (rr != tr) {
                        log_error("ERROR: sub_group_scan_inclusive_%s(%s) mismatch for local id %d in sub group %d in group %d\n",
                                   Which == 0 ? "add" : (Which == 1 ? "max" : "min"), TypeName<Ty>::val(), i, j, k);
                        return -1;
                    }
                }
            }

            x += nw;
            y += nw;
            m += 4*nw;
        }

        return 0;
    }
};

// Scan Exclusive functions
template <typename Ty, int Which>
struct SCEX {
    static void gen(Ty *x, Ty *t, cl_int *m, int ns, int nw, int ng)
    {
        int i, ii, j, k, n;
        int nj = (nw + ns - 1)/ns;

        ii = 0;
        for (k=0; k<ng; ++k) {
            for (j=0; j<nj; ++j) {
                ii = j*ns;
                n = ii + ns > nw ? nw - ii : ns;

                for (i=0; i<n; ++i)
                    t[ii+i] = (Ty)((int)(genrand_int32(gMTdata) & 0x7fffffff) % ns + 1);
            }

            // Now map into work group using map from device
            for (j=0;j<nw;++j) {
                i = m[4*j+1]*ns + m[4*j];
                x[j] = t[i];
            }

            x += nw;
        m += 4*nw;
        }
    }

    static int chk(Ty *x, Ty *y, Ty *mx, Ty *my, cl_int *m, int ns, int nw, int ng)
    {
        int ii, i, j, k, n;
        int nj = (nw + ns - 1)/ns;
        Ty tr, trt, rr;

        log_info("  sub_group_scan_exclusive_%s(%s)...\n", Which == 0 ? "add" : (Which == 1 ? "max" : "min"), TypeName<Ty>::val());

        for (k=0; k<ng; ++k) {
            // Map to array indexed to array indexed by local ID and sub group
            for (j=0; j<nw; ++j) {
                i = m[4*j+1]*ns + m[4*j];
                mx[i] = x[j];
                my[i] = y[j];
            }

            for (j=0; j<nj; ++j) {
                ii = j*ns;
                n = ii + ns > nw ? nw - ii : ns;

                // Check result
                for (i=0; i<n; ++i) {
                    if (Which == 0) {
                        tr = i == 0 ? TypeIdentity<Ty,Which>::val() : tr + trt;
                    } else if (Which == 1) {
                        tr = i == 0 ? TypeIdentity<Ty,Which>::val() : (trt > tr ? trt : tr);
                    } else {
                        tr = i == 0 ? TypeIdentity<Ty,Which>::val() : (trt > tr ? tr : trt);
                    }
                    trt = mx[ii+i];
                    rr = my[ii+i];

                    if (rr != tr) {
                        log_error("ERROR: sub_group_scan_exclusive_%s(%s) mismatch for local id %d in sub group %d in group %d\n",
                                   Which == 0 ? "add" : (Which == 1 ? "max" : "min"), TypeName<Ty>::val(), i, j, k);
                        return -1;
                    }
                }
            }

            x += nw;
            y += nw;
            m += 4*nw;
        }

        return 0;
    }
};

// Broadcast functios
template <typename Ty>
struct BC {
    static void gen(Ty *x, Ty *t, cl_int *m, int ns, int nw, int ng)
    {
        int i, ii, j, k, l, n;
        int nj = (nw + ns - 1)/ns;
        int d = ns > 100 ? 100 : ns;

        ii = 0;
        for (k=0; k<ng; ++k) {
            for (j=0; j<nj; ++j) {
                ii = j*ns;
                n = ii + ns > nw ? nw - ii : ns;
                l = (int)(genrand_int32(gMTdata) & 0x7fffffff) % (d > n ? n : d);

                for (i = 0; i < n; ++i) {
                    int midx = 4*ii+ 4*i + 2 ;
                    m[midx] = (cl_int)l; // third eleement in the table
                    cl_uint number;
                    number = (int)(genrand_int32(gMTdata) & 0x7fffffff); // be sure value is no more than about 10000 plus number above l (from 0 to 100)
                    t[ii + i] = number;
                }
            }
            // Now map into work group using map from device
            for (j=0;j<nw;++j) {
                i = m[4*j+1]*ns + m[4*j]; //take number of subgroup plus take work item id in subgroup
                x[j] = t[i];
            }

            x += nw;
            m += 4*nw; // have to move becase read 2 elements above
        }
    }

    static int chk(Ty *x, Ty *y, Ty *mx, Ty *my, cl_int *m, int ns, int nw, int ng)
    {
        int ii, i, j, k, l, n;
        int nj = (nw + ns - 1)/ns;
        Ty tr, rr;

        log_info("  sub_group_broadcast(%s)...\n", TypeName<Ty>::val());

        for (k=0; k<ng; ++k) {
            // Map to array indexed to array indexed by local ID and sub group
            for (j=0; j<nw; ++j) {
                i = m[4*j+1]*ns + m[4*j];
                mx[i] = x[j];
                my[i] = y[j];
            }

            for (j=0; j<nj; ++j) {
                ii = j*ns;
                n = ii + ns > nw ? nw - ii : ns;
                int midx = 4 * ii+ 2;
                l = (int)m[midx];
                tr = mx[ii+l];

                // Check result
                for (i=0; i<n; ++i) {
                    rr = my[ii+i];
                    if (rr != tr) {
                        log_error("ERROR: sub_group_broadcast(%s) mismatch for local id %d in sub group %d in group %d\n",
                                   TypeName<Ty>::val(), i, j, k);
                        return -1;
                    }
                }
            }

            x += nw;
            y += nw;
            m += 4*nw;
        }

        return 0;
    }
};

// Independent forward progress stuff
// Note:
//   Output needs num_groups * NUM_LOC elements
//   local_size must be > NUM_LOC
//   Input needs num_groups * num_sub_groups * (NUM_LOC+1) elements

static inline int
inst(int op, int loc, int val)
{
    return (val << INST_VAL_SHIFT) | (loc << INST_LOC_SHIFT) | (op << INST_OP_SHIFT);
}

void gen_insts(cl_int *x, cl_int *p, int n)
{
    int i, j0, j1;
    int val;
    int ii[NUM_LOC];

    // Create a random permutation of 0...NUM_LOC-1
    ii[0] = 0;
    for (i=1; i<NUM_LOC;++i) {
        j0 = random_in_range(0, i, gMTdata);
        if (j0 != i)
            ii[i] = ii[j0];
        ii[j0] = i;
    }

    // Initialize "instruction pointers"
    memset(p, 0, n*4);

    for (i=0; i<NUM_LOC; ++i) {
        // Randomly choose 2 different sub groups
        // One does a random amount of work, and the other waits for it
        j0 = random_in_range(0, n-1, gMTdata);

        do
            j1 = random_in_range(0, n-1, gMTdata);
        while (j1 == j0);

        // Randomly choose a wait value and assign "instructions"
        val = random_in_range(100, 200 + 10*NUM_LOC, gMTdata);
        x[j0*(NUM_LOC+1) + p[j0]] = inst(INST_COUNT, ii[i], val);
        x[j1*(NUM_LOC+1) + p[j1]] = inst(INST_WAIT,  ii[i], val);
        ++p[j0];
        ++p[j1];
    }

    // Last "inst" for each sub group is END
    for (i=0; i<n; ++i)
        x[i*(NUM_LOC+1) + p[i]] = inst(INST_END, 0, 0);
}

// Execute one group's "instructions"
void run_insts(cl_int *x, cl_int *p, int n)
{
    int i, nend;
    bool scont;
    cl_int loc[NUM_LOC];

    // Initialize result and "instruction pointers"
    memset(loc, 0, sizeof(loc));
    memset(p, 0, 4*n);

    // Repetitively loop over subgroups with each executing "instructions" until blocked
    // The loop terminates when all subgroups have hit the "END instruction"
    do {
        nend = 0;
        for (i=0; i<n; ++i) {
            do {
                cl_int inst = x[i*(NUM_LOC+1) + p[i]];
                cl_int iop = (inst >> INST_OP_SHIFT) & INST_OP_MASK;
                cl_int iloc = (inst >> INST_LOC_SHIFT) & INST_LOC_MASK;
                cl_int ival = (inst >> INST_VAL_SHIFT) & INST_VAL_MASK;
                scont = false;

                switch (iop) {
                case INST_STORE:
                    loc[iloc] = ival;
                    ++p[i];
                    scont = true;
                    break;
                case INST_WAIT:
                    if (loc[iloc] == ival) {
                        ++p[i];
                        scont = true;
                    }
                    break;
                case INST_COUNT:
                    loc[iloc] += ival;
                    ++p[i];
                    scont = true;
                    break;
                case INST_END:
                    ++nend;
                    break;
                }
            } while (scont);
        }
    } while (nend < n);

    // Return result, reusing "p"
    memcpy(p, loc, sizeof(loc));
}


struct IFP {
    static void gen(cl_int *x, cl_int *t, cl_int *, int ns, int nw, int ng)
    {
        int k;
        int nj = (nw + ns - 1) / ns;

        // We need at least 2 sub groups per group for this test
        if (nj == 1)
            return;

        for (k=0; k<ng; ++k) {
            gen_insts(x, t, nj);
            x += nj * (NUM_LOC+1);
        }
    }

    static int chk(cl_int *x, cl_int *y, cl_int *t, cl_int *, cl_int *, int ns, int nw, int ng)
    {
        int i, k;
        int nj = (nw + ns - 1) / ns;

        // We need at least 2 sub groups per group for this tes
        if (nj == 1)
            return 0;

        log_info("  independent forward progress...\n");

        for (k=0; k<ng; ++k) {
            run_insts(x, t, nj);
            for (i=0; i<NUM_LOC; ++i) {
                if (t[i] != y[i]) {
                    log_error("ERROR: mismatch at element %d in work group %d\n", i, k);
                    return -1;
                }
            }
            x += nj * (NUM_LOC+1);
            y += NUM_LOC;
        }

        return 0;
    }
};


// Entry point from main
int
test_work_group_functions(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    int error;

    // Adjust these individually below if desired/needed
#define G 2000
#define L 200

    error = test<int, AA<0>, G, L>::run(device, context, queue, num_elements, "test_any", any_source);
    error |= test<int, AA<1>, G, L>::run(device, context, queue, num_elements, "test_all", all_source);

    // error |= test<cl_half, BC<cl_half>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source);
    error |= test<cl_uint, BC<cl_uint>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source);
    error |= test<cl_int, BC<cl_int>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source);
    error |= test<cl_ulong, BC<cl_ulong>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source);
    error |= test<cl_long, BC<cl_long>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source);
    error |= test<float, BC<float>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source);
    error |= test<double, BC<double>, G, L>::run(device, context, queue, num_elements, "test_bcast", bcast_source);

    // error |= test<cl_half, RED<cl_half,0>, G, L>::run(device, context, queue, num_elements, "test_redadd", redadd_source);
    error |= test<cl_uint, RED<cl_uint,0>, G, L>::run(device, context, queue, num_elements, "test_redadd", redadd_source);
    error |= test<cl_int, RED<cl_int,0>, G, L>::run(device, context, queue, num_elements, "test_redadd", redadd_source);
    error |= test<cl_ulong, RED<cl_ulong,0>, G, L>::run(device, context, queue, num_elements, "test_redadd", redadd_source);
    error |= test<cl_long, RED<cl_long,0>, G, L>::run(device, context, queue, num_elements, "test_redadd", redadd_source);
    error |= test<float, RED<float,0>, G, L>::run(device, context, queue, num_elements, "test_redadd", redadd_source);
    error |= test<double, RED<double,0>, G, L>::run(device, context, queue, num_elements, "test_redadd", redadd_source);

    // error |= test<cl_half, RED<cl_half,1>, G, L>::run(device, context, queue, num_elements, "test_redmax", redmax_source);
    error |= test<cl_uint, RED<cl_uint,1>, G, L>::run(device, context, queue, num_elements, "test_redmax", redmax_source);
    error |= test<cl_int, RED<cl_int,1>, G, L>::run(device, context, queue, num_elements, "test_redmax", redmax_source);
    error |= test<cl_ulong, RED<cl_ulong,1>, G, L>::run(device, context, queue, num_elements, "test_redmax", redmax_source);
    error |= test<cl_long, RED<cl_long,1>, G, L>::run(device, context, queue, num_elements, "test_redmax", redmax_source);
    error |= test<float, RED<float,1>, G, L>::run(device, context, queue, num_elements, "test_redmax", redmax_source);
    error |= test<double, RED<double,1>, G, L>::run(device, context, queue, num_elements, "test_redmax", redmax_source);

    // error |= test<cl_half, RED<cl_half,2>, G, L>::run(device, context, queue, num_elements, "test_redmin", redmin_source);
    error |= test<cl_uint, RED<cl_uint,2>, G, L>::run(device, context, queue, num_elements, "test_redmin", redmin_source);
    error |= test<cl_int, RED<cl_int,2>, G, L>::run(device, context, queue, num_elements, "test_redmin", redmin_source);
    error |= test<cl_ulong, RED<cl_ulong,2>, G, L>::run(device, context, queue, num_elements, "test_redmin", redmin_source);
    error |= test<cl_long, RED<cl_long,2>, G, L>::run(device, context, queue, num_elements, "test_redmin", redmin_source);
    error |= test<float, RED<float,2>, G, L>::run(device, context, queue, num_elements, "test_redmin", redmin_source);
    error |= test<double, RED<double,2>, G, L>::run(device, context, queue, num_elements, "test_redmin", redmin_source);

    // error |= test<cl_half, SCIN<cl_half,0>, G, L>::run(device, context, queue, num_elements, "test_scinadd", scinadd_source);
    error |= test<cl_uint, SCIN<cl_uint,0>, G, L>::run(device, context, queue, num_elements, "test_scinadd", scinadd_source);
    error |= test<cl_int, SCIN<cl_int,0>, G, L>::run(device, context, queue, num_elements, "test_scinadd", scinadd_source);
    error |= test<cl_ulong, SCIN<cl_ulong,0>, G, L>::run(device, context, queue, num_elements, "test_scinadd", scinadd_source);
    error |= test<cl_long, SCIN<cl_long,0>, G, L>::run(device, context, queue, num_elements, "test_scinadd", scinadd_source);
    error |= test<float, SCIN<float,0>, G, L>::run(device, context, queue, num_elements, "test_scinadd", scinadd_source);
    error |= test<double, SCIN<double,0>, G, L>::run(device, context, queue, num_elements, "test_scinadd", scinadd_source);

    // error |= test<cl_half, SCIN<cl_half,1>, G, L>::run(device, context, queue, num_elements, "test_scinmax", scinmax_source);
    error |= test<cl_uint, SCIN<cl_uint,1>, G, L>::run(device, context, queue, num_elements, "test_scinmax", scinmax_source);
    error |= test<cl_int, SCIN<cl_int,1>, G, L>::run(device, context, queue, num_elements, "test_scinmax", scinmax_source);
    error |= test<cl_ulong, SCIN<cl_ulong,1>, G, L>::run(device, context, queue, num_elements, "test_scinmax", scinmax_source);
    error |= test<cl_long, SCIN<cl_long,1>, G, L>::run(device, context, queue, num_elements, "test_scinmax", scinmax_source);
    error |= test<float, SCIN<float,1>, G, L>::run(device, context, queue, num_elements, "test_scinmax", scinmax_source);
    error |= test<double, SCIN<double,1>, G, L>::run(device, context, queue, num_elements, "test_scinmax", scinmax_source);

    // error |= test<cl_half, SCIN<cl_half,2>, G, L>::run(device, context, queue, num_elements, "test_scinmin", scinmin_source);
    error |= test<cl_uint, SCIN<cl_uint,2>, G, L>::run(device, context, queue, num_elements, "test_scinmin", scinmin_source);
    error |= test<cl_int, SCIN<cl_int,2>, G, L>::run(device, context, queue, num_elements, "test_scinmin", scinmin_source);
    error |= test<cl_ulong, SCIN<cl_ulong,2>, G, L>::run(device, context, queue, num_elements, "test_scinmin", scinmin_source);
    error |= test<cl_long, SCIN<cl_long,2>, G, L>::run(device, context, queue, num_elements, "test_scinmin", scinmin_source);
    error |= test<float, SCIN<float,2>, G, L>::run(device, context, queue, num_elements, "test_scinmin", scinmin_source);
    error |= test<double, SCIN<double,2>, G, L>::run(device, context, queue, num_elements, "test_scinmin", scinmin_source);

    // error |= test<cl_half, SCEX<cl_half,0>, G, L>::run(device, context, queue, num_elements, "test_scexadd", scexadd_source);
    error |= test<cl_uint, SCEX<cl_uint,0>, G, L>::run(device, context, queue, num_elements, "test_scexadd", scexadd_source);
    error |= test<cl_int, SCEX<cl_int,0>, G, L>::run(device, context, queue, num_elements, "test_scexadd", scexadd_source);
    error |= test<cl_ulong, SCEX<cl_ulong,0>, G, L>::run(device, context, queue, num_elements, "test_scexadd", scexadd_source);
    error |= test<cl_long, SCEX<cl_long,0>, G, L>::run(device, context, queue, num_elements, "test_scexadd", scexadd_source);
    error |= test<float, SCEX<float,0>, G, L>::run(device, context, queue, num_elements, "test_scexadd", scexadd_source);
    error |= test<double, SCEX<double,0>, G, L>::run(device, context, queue, num_elements, "test_scexadd", scexadd_source);

    // error |= test<cl_half, SCEX<cl_half,1>, G, L>::run(device, context, queue, num_elements, "test_scexmax", scexmax_source);
    error |= test<cl_uint, SCEX<cl_uint,1>, G, L>::run(device, context, queue, num_elements, "test_scexmax", scexmax_source);
    error |= test<cl_int, SCEX<cl_int,1>, G, L>::run(device, context, queue, num_elements, "test_scexmax", scexmax_source);
    error |= test<cl_ulong, SCEX<cl_ulong,1>, G, L>::run(device, context, queue, num_elements, "test_scexmax", scexmax_source);
    error |= test<cl_long, SCEX<cl_long,1>, G, L>::run(device, context, queue, num_elements, "test_scexmax", scexmax_source);
    error |= test<float, SCEX<float,1>, G, L>::run(device, context, queue, num_elements, "test_scexmax", scexmax_source);
    error |= test<double, SCEX<double,1>, G, L>::run(device, context, queue, num_elements, "test_scexmax", scexmax_source);

    // error |= test<cl_half, SCEX<cl_half,2>, G, L>::run(device, context, queue, num_elements, "test_scexmin", scexmin_source);
    error |= test<cl_uint, SCEX<cl_uint,2>, G, L>::run(device, context, queue, num_elements, "test_scexmin", scexmin_source);
    error |= test<cl_int, SCEX<cl_int,2>, G, L>::run(device, context, queue, num_elements, "test_scexmin", scexmin_source);
    error |= test<cl_ulong, SCEX<cl_ulong,2>, G, L>::run(device, context, queue, num_elements, "test_scexmin", scexmin_source);
    error |= test<cl_long, SCEX<cl_long,2>, G, L>::run(device, context, queue, num_elements, "test_scexmin", scexmin_source);
    error |= test<float, SCEX<float,2>, G, L>::run(device, context, queue, num_elements, "test_scexmin", scexmin_source);
    error |= test<double, SCEX<double,2>, G, L>::run(device, context, queue, num_elements, "test_scexmin", scexmin_source);

    error |= test<cl_int, IFP, G, L>::run(device, context, queue, num_elements, "test_ifp", ifp_source, NUM_LOC + 1);
    return error;
}

