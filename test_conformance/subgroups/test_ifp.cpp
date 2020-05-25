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
#include "harness/conversions.h"
#include "harness/typeWrappers.h"


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

static const char *ifp_source =
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
    "test_ifp(const __global int *in, __global int2 *xy, __global int *out)\n"
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


// Independent forward progress stuff
// Note:
//   Output needs num_groups * NUM_LOC elements
//   local_size must be > NUM_LOC
//   Input needs num_groups * num_sub_groups * (NUM_LOC+1) elements

static inline int inst(int op, int loc, int val)
{
    return (val << INST_VAL_SHIFT) | (loc << INST_LOC_SHIFT)
        | (op << INST_OP_SHIFT);
}

void gen_insts(cl_int *x, cl_int *p, int n)
{
    int i, j0, j1;
    int val;
    int ii[NUM_LOC];

    // Create a random permutation of 0...NUM_LOC-1
    ii[0] = 0;
    for (i = 1; i < NUM_LOC; ++i)
    {
        j0 = random_in_range(0, i, gMTdata);
        if (j0 != i) ii[i] = ii[j0];
        ii[j0] = i;
    }

    // Initialize "instruction pointers"
    memset(p, 0, n * 4);

    for (i = 0; i < NUM_LOC; ++i)
    {
        // Randomly choose 2 different sub groups
        // One does a random amount of work, and the other waits for it
        j0 = random_in_range(0, n - 1, gMTdata);

        do
        {
            j1 = random_in_range(0, n - 1, gMTdata);
        } while (j1 == j0);

        // Randomly choose a wait value and assign "instructions"
        val = random_in_range(100, 200 + 10 * NUM_LOC, gMTdata);
        x[j0 * (NUM_LOC + 1) + p[j0]] = inst(INST_COUNT, ii[i], val);
        x[j1 * (NUM_LOC + 1) + p[j1]] = inst(INST_WAIT, ii[i], val);
        ++p[j0];
        ++p[j1];
    }

    // Last "inst" for each sub group is END
    for (i = 0; i < n; ++i) x[i * (NUM_LOC + 1) + p[i]] = inst(INST_END, 0, 0);
}

// Execute one group's "instructions"
void run_insts(cl_int *x, cl_int *p, int n)
{
    int i, nend;
    bool scont;
    cl_int loc[NUM_LOC];

    // Initialize result and "instruction pointers"
    memset(loc, 0, sizeof(loc));
    memset(p, 0, 4 * n);

    // Repetitively loop over subgroups with each executing "instructions" until
    // blocked The loop terminates when all subgroups have hit the "END
    // instruction"
    do
    {
        nend = 0;
        for (i = 0; i < n; ++i)
        {
            do
            {
                cl_int inst = x[i * (NUM_LOC + 1) + p[i]];
                cl_int iop = (inst >> INST_OP_SHIFT) & INST_OP_MASK;
                cl_int iloc = (inst >> INST_LOC_SHIFT) & INST_LOC_MASK;
                cl_int ival = (inst >> INST_VAL_SHIFT) & INST_VAL_MASK;
                scont = false;

                switch (iop)
                {
                    case INST_STORE:
                        loc[iloc] = ival;
                        ++p[i];
                        scont = true;
                        break;
                    case INST_WAIT:
                        if (loc[iloc] == ival)
                        {
                            ++p[i];
                            scont = true;
                        }
                        break;
                    case INST_COUNT:
                        loc[iloc] += ival;
                        ++p[i];
                        scont = true;
                        break;
                    case INST_END: ++nend; break;
                }
            } while (scont);
        }
    } while (nend < n);

    // Return result, reusing "p"
    memcpy(p, loc, sizeof(loc));
}


struct IFP
{
    static void gen(cl_int *x, cl_int *t, cl_int *, int ns, int nw, int ng)
    {
        int k;
        int nj = (nw + ns - 1) / ns;

        // We need at least 2 sub groups per group for this test
        if (nj == 1) return;

        for (k = 0; k < ng; ++k)
        {
            gen_insts(x, t, nj);
            x += nj * (NUM_LOC + 1);
        }
    }

    static int chk(cl_int *x, cl_int *y, cl_int *t, cl_int *, cl_int *, int ns,
                   int nw, int ng)
    {
        int i, k;
        int nj = (nw + ns - 1) / ns;

        // We need at least 2 sub groups per group for this tes
        if (nj == 1) return 0;

        log_info("  independent forward progress...\n");

        for (k = 0; k < ng; ++k)
        {
            run_insts(x, t, nj);
            for (i = 0; i < NUM_LOC; ++i)
            {
                if (t[i] != y[i])
                {
                    log_error(
                        "ERROR: mismatch at element %d in work group %d\n", i,
                        k);
                    return -1;
                }
            }
            x += nj * (NUM_LOC + 1);
            y += NUM_LOC;
        }

        return 0;
    }
};

int test_ifp(cl_device_id device, cl_context context, cl_command_queue queue,
             int num_elements, bool useCoreSubgroups)
{
    int error;

    // Adjust these individually below if desired/needed
#define G 2000
#define L 200
    error = test<cl_int, IFP, G, L>::run(device, context, queue, num_elements,
                                         "test_ifp", ifp_source, NUM_LOC + 1,
                                         useCoreSubgroups);
    return error;
}

static test_status checkIFPSupport(cl_device_id device, bool &ifpSupport)
{
    cl_uint ifp_supported;
    cl_uint error;
    error = clGetDeviceInfo(device,
                            CL_DEVICE_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS,
                            sizeof(ifp_supported), &ifp_supported, NULL);
    if (error != CL_SUCCESS)
    {
        print_error(
            error,
            "Unable to get CL_DEVICE_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS "
            "capability");
        return TEST_FAIL;
    }
    // skip testing ifp
    if (ifp_supported != 1)
    {
        log_info("INDEPENDENT FORWARD PROGRESS not supported...\n");
        ifpSupport = false;
    }
    else
    {
        log_info("INDEPENDENT FORWARD PROGRESS supported...\n");
        ifpSupport = true;
    }
    return TEST_PASS;
}

int test_ifp_core(cl_device_id device, cl_context context,
                  cl_command_queue queue, int num_elements)
{
    bool ifpSupport = true;
    test_status error;
    error = checkIFPSupport(device, ifpSupport);
    if (error != TEST_PASS)
    {
        return error;
    }
    if (ifpSupport == false)
    {
        log_info("Independed forward progress skipped.\n");
        return TEST_SKIPPED_ITSELF;
    }

    return test_ifp(device, context, queue, num_elements, true);
}

int test_ifp_ext(cl_device_id device, cl_context context,
                 cl_command_queue queue, int num_elements)
{
    bool hasExtension = is_extension_available(device, "cl_khr_subgroups");
    bool ifpSupport = true;

    if (!hasExtension)
    {
        log_info(
            "Device does not support 'cl_khr_subgroups'. Skipping the test.\n");
        return TEST_SKIPPED_ITSELF;
    }
    // ifp only in subgroup functions tests:
    test_status error;
    error = checkIFPSupport(device, ifpSupport);
    if (error != TEST_PASS)
    {
        return error;
    }
    if (ifpSupport == false)
    {
        log_info(
            "Error reason: the extension cl_khr_subgroups requires that "
            "Independed forward progress has to be supported by device.\n");
        return TEST_FAIL;
    }
    return test_ifp(device, context, queue, num_elements, false);
}