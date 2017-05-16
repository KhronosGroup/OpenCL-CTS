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
#ifndef TEST_CONFORMANCE_CLCPP_SYNCHRONIZATION_NAMED_BARRIER_TEST_SPEC_EXAMPLE_HPP
#define TEST_CONFORMANCE_CLCPP_SYNCHRONIZATION_NAMED_BARRIER_TEST_SPEC_EXAMPLE_HPP

#include "common.hpp"

namespace named_barrier {

// ------------------------------------------------------------------------------
// ----------------------- SPECIFICATION EXAMPLE TEST----------------------------
// ------------------------------------------------------------------------------
// This test is based on the example in OpenCL C++ 1.0 specification (OpenCL C++
// Standard Library > Synchronization Functions > Named barriers > wait).
struct spec_example_work_group_named_barrier_test : public work_group_named_barrier_test_base
{
    std::string str()
    {
        return "spec_example";
    }

    // Return value that is expected to be in output_buffer[i]
    cl_uint operator()(size_t i, size_t work_group_size, size_t mas_sub_group_size)
    {
        return static_cast<cl_uint>(i);
    }

    // At the end every work-item writes its global id to ouput[work-item-global-id].
    std::string generate_program()
    {
        // -----------------------------------------------------------------------------------
        // ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
        // -----------------------------------------------------------------------------------
        #if defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
            return
                // In OpenCL C kernel we imitate subgroups by partitioning work-group (based on
                // local ids of work-items), work_group_named_barrier.wait(..) calls are replaced
                // with work_group_barriers.
                "__kernel void " + this->get_kernel_name() + "(global uint *output, "
                                                              "global uint * temp, "
                                                              "local uint * lmem)\n"
                "{\n"
                "size_t gid = get_global_id(0);\n"
                "size_t lid = get_local_id(0);\n"

                // We divide work-group into ranges:
                // [0 - e_wg)[ew_g; q_wg)[q_wg; 3 * ew_g)[3 * ew_g; h_wg)[h_wg; get_local_size(0) - 1]
                // to simulate 8 subgroups
                "size_t h_wg = get_local_size(0) / 2;\n" // half of work-group
                "size_t q_wg = get_local_size(0) / 4;\n" // quarter
                "size_t e_wg = get_local_size(0) / 8;\n" // one-eighth

                "if(lid < h_wg) lmem[lid] = gid;\n" // [0; h_wg)
                "else           temp[gid] = gid;\n" // [h_wg; get_local_size(0) - 1)
                "work_group_barrier(CLK_LOCAL_MEM_FENCE);\n"

                "size_t other_lid = (lid + q_wg) % h_wg;\n"
                "size_t value = 0;\n"
                "if(lmem[other_lid] == ((gid - lid) + other_lid)){\n"
                "     value = gid;\n"
                "}\n"
                "work_group_barrier(CLK_LOCAL_MEM_FENCE);\n"

                "if(lid < q_wg){\n" // [0; q_wg)
                "    if(lid < e_wg) lmem[lid + e_wg] = gid;\n" // [0; e_wg)
                "    else           lmem[lid - e_wg] = gid;\n" // [e_wg; q_wg)
                "}\n"
                "else if(lid < h_wg) {\n" // [q_wg; h_wg)
                "    if(lid < (3 * e_wg)) lmem[lid + e_wg] = gid;\n" // [q_ww; q_wg + e_wg)
                "    else                 lmem[lid - e_wg] = gid;\n" // [q_wg + e_wg; h_wg)
                "}\n"
                "work_group_barrier(CLK_LOCAL_MEM_FENCE);\n"

                "if(lid < q_wg){\n" // [0; q_wg)
                "    output[gid + q_wg] = lmem[lid];\n"
                "}\n"
                "else if(lid < h_wg) {\n" // [q_wg; h_wg)
                "    output[gid - q_wg] = lmem[lid];\n"
                "}\n"
                "work_group_barrier(CLK_GLOBAL_MEM_FENCE);\n"

                "if(lid < q_wg){\n" // [0; q_wg)
                "    if(lid < e_wg) temp[gid] = output[gid + (3 * e_wg)];\n" // [0; e_wg)
                "    else           temp[gid] = output[gid + e_wg];\n" // [e_wg; q_wg)
                "}\n"
                "else if(lid < h_wg) {\n" // [q_wg; h_wg)
                "    if(lid < (3 * e_wg)) temp[gid] = output[gid - e_wg];\n"  // [q_ww; q_wg + e_wg)
                "    else                 temp[gid] = output[gid - (3 * e_wg)];\n"  // [q_wg + e_wg; h_wg)
                "}\n"
                "work_group_barrier(CLK_GLOBAL_MEM_FENCE);\n"

                "output[gid] = temp[gid];\n"
                "}\n";

        #else
            return
                "#define cl_khr_subgroup_named_barrier\n"
                "#include <opencl_memory>\n"
                "#include <opencl_work_item>\n"
                "#include <opencl_synchronization>\n"
                "using namespace cl;\n"

                "void b_function(work_group_named_barrier &b, size_t value, local_ptr<uint[]> lmem)\n"
                "{\n\n"
                "size_t lid = get_local_id(0);\n"
                // Work-items from the 1st subgroup writes to local memory that will be
                // later read byt the 0th subgroup, and the other way around - 0th subgroup
                // writes what 1st subgroup will later read.
                // b.wait(mem_fence::local) should provide sync between those two subgroups.
                "if(get_sub_group_id() < 1) lmem[lid + get_max_sub_group_size()] = value;\n"
                "else                       lmem[lid - get_max_sub_group_size()] = value;\n"
                "b.wait(mem_fence::local);\n\n" // sync writes to lmem for 2 subgroups (ids: 0, 1)
                "}\n"

                "__kernel void " + this->get_kernel_name() + "(global_ptr<uint[]> output, "
                                                              "global_ptr<uint[]> temp, "
                                                              "local_ptr<uint[]> lmem)\n"
                "{\n\n"
                "local<work_group_named_barrier> a(4);\n"
                "local<work_group_named_barrier> b(2);\n"
                "local<work_group_named_barrier> c(2);\n"

                "size_t gid = get_global_id(0);\n"
                "size_t lid = get_local_id(0);\n"
                "if(get_sub_group_id() < 4)"
                "{\n"
                "    lmem[lid] = gid;\n"
                "    a.wait(mem_fence::local);\n" // sync writes to lmem for 4 subgroups (ids: 0, 1, 2, 3)
                     // Now all four subgroups should see changes in lmem.
                "    size_t other_lid = (lid + (2 * get_max_sub_group_size())) % (4 * get_max_sub_group_size());\n"
                "    size_t value = 0;\n"
                "    if(lmem[other_lid] == ((gid - lid) + other_lid)){\n"
                "        value = gid;\n"
                "    }\n"
                "    a.wait(mem_fence::local);\n" // sync reads from lmem for 4 subgroups (ids: 0, 1, 2, 3)

                "    if(get_sub_group_id() < 2)" // ids: 0, 1
                "    {\n"
                "        b_function(b, value, lmem);\n"
                "    }\n"
                "    else" // ids: 2, 3
                "    {\n"
                         // Work-items from the 2nd subgroup writes to local memory that will be
                         // later read byt the 3rd subgroup, and the other way around - 3rd subgroup
                         // writes what 2nd subgroup will later read.
                         // c.wait(mem_fence::local) should provide sync between those two subgroups.
                "        if(get_sub_group_id() < 3) lmem[lid + get_max_sub_group_size()] = value ;\n"
                "        else                       lmem[lid - get_max_sub_group_size()] = value;\n"
                "        c.wait(mem_fence::local);\n" // sync writes to lmem for 2 subgroups (3, 4)
                "    }\n"

                     // Now (0, 1) are in sync (local mem), and (3, 4) are in sync (local mem).
                     // However, subgroups (0, 1) are not in sync with (3, 4).
                "    if(get_sub_group_id() < 4) {\n" // ids: 0, 1, 2, 3
                "        if(get_sub_group_id() < 2) output[gid + (2 * get_max_sub_group_size())] = lmem[lid];\n"
                "        else                       output[gid - (2 * get_max_sub_group_size())] = lmem[lid];\n"
                "        a.wait(mem_fence::global);\n" // sync writes to global memory (output)
                                                       // for 4 subgroups (0, 1, 2, 3)
                "    }\n"
                "}\n"
                "else {\n" // subgroups with id > 4
                "    temp[gid] = gid;\n"
                "}\n"

                // Now (0, 1, 2, 3) are in sync (global mem)
                "if(get_sub_group_id() < 2) {\n"
                "    if(get_sub_group_id() < 1) temp[gid] = output[gid + (3 * get_max_sub_group_size())];\n"
                "    else                       temp[gid] = output[gid + (get_max_sub_group_size())];\n"
                "}\n"
                "else if(get_sub_group_id() < 4) {\n"
                "    if(get_sub_group_id() < 3) temp[gid] = output[gid - (get_max_sub_group_size())];\n"
                "    else                       temp[gid] = output[gid - (3 * get_max_sub_group_size())];\n"
                "}\n"

                // Synchronize the entire work-group (in terms of accesses to global memory)
                "work_group_barrier(mem_fence::global);\n"
                "output[gid] = temp[gid];\n\n"
                "}\n";
        #endif
    }

    size_t get_max_local_size(const cl_kernel kernel,
                              const cl_device_id device,
                              const size_t work_group_size, // default work-group size
                              cl_int& error)
    {
        // Set size of the local memory, we need to to this to correctly calculate
        // max possible work-group size.
        size_t wg_size;
        for(wg_size = work_group_size; wg_size > 1; wg_size /= 2)
        {
            error = clSetKernelArg(kernel, 2, ((wg_size / 2) + 1) * sizeof(cl_uint), NULL);
            RETURN_ON_CL_ERROR(error, "clSetKernelArg")

            size_t max_wg_size;
            error = clGetKernelWorkGroupInfo(
                kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &max_wg_size, NULL
            );
            RETURN_ON_ERROR(error)
            if(max_wg_size >= wg_size) break;
        }

        // -----------------------------------------------------------------------------------
        // ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
        // -----------------------------------------------------------------------------------
        #if defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
            // make sure wg_size is a multiple of 8
            if(wg_size % 8 > 0) wg_size -= (wg_size % 8);
            return wg_size;
        #else
            // make sure that wg_size will produce at least min_num_sub_groups
            // subgroups in each work-group
            size_t local_size[3] = { 1, 1, 1 };
            size_t min_num_sub_groups = 8;
            error = clGetKernelSubGroupInfo(kernel, device, CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT,
                                            sizeof(size_t), &min_num_sub_groups,
                                            sizeof(size_t) * 3, &local_size, NULL);
            RETURN_ON_CL_ERROR(error, "clGetKernelSubGroupInfo")
            if (local_size[0] == 0 || local_size[1] != 1 || local_size[2] != 1)
            {
                if(min_num_sub_groups == 1)
                {
                    RETURN_ON_ERROR_MSG(-1, "Can't produce local size with one subgroup")
                }
                return 0;
            }
            local_size[0] = (std::min)(wg_size, local_size[0]);

            // double-check
            size_t sub_group_count_for_ndrange;
            error = clGetKernelSubGroupInfo(kernel, device, CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE,
                                            sizeof(size_t) * 3, local_size,
                                            sizeof(size_t), &sub_group_count_for_ndrange, NULL);
            RETURN_ON_CL_ERROR(error, "clGetKernelSubGroupInfo")
            if (sub_group_count_for_ndrange < min_num_sub_groups)
            {
                RETURN_ON_ERROR_MSG(-1,
                    "CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE did not return correct value (expected >=%lu, got %lu)",
                    min_num_sub_groups, sub_group_count_for_ndrange
                )
            }

            return local_size[0];
        #endif
    }

    cl_int execute(const cl_kernel kernel,
                   const cl_mem output_buffer,
                   const cl_command_queue queue,
                   const size_t work_size,
                   const size_t work_group_size)
    {
        cl_int err;
        // Get context from queue
        cl_context context;
        err = clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(cl_context), &context, NULL);
        RETURN_ON_CL_ERROR(err, "clGetCommandQueueInfo")

        // create temp buffer
        auto temp_buffer = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE), sizeof(cl_uint) * work_size, NULL, &err);
        RETURN_ON_CL_ERROR(err, "clCreateBuffer")

        err = clSetKernelArg(kernel, 0, sizeof(output_buffer), &output_buffer);
        err |= clSetKernelArg(kernel, 1, sizeof(temp_buffer), &temp_buffer);
        err |= clSetKernelArg(kernel, 2, work_group_size * sizeof(cl_uint), NULL);
        RETURN_ON_CL_ERROR(err, "clSetKernelArg")

        err = clEnqueueNDRangeKernel(
            queue, kernel, 1,
            NULL, &work_size, &work_group_size,
            0, NULL, NULL
        );
        RETURN_ON_CL_ERROR(err, "clEnqueueNDRangeKernel")

        err = clFinish(queue);
        err |= clReleaseMemObject(temp_buffer);

        return err;
    }
};

// ------------------------------------------------------------------------------
// -------------------------- RUN TESTS -----------------------------------------
// ------------------------------------------------------------------------------
AUTO_TEST_CASE(test_work_group_named_barrier_spec_example)
(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
#if !(defined(DEVELOPMENT) && (defined(USE_OPENCLC_KERNELS) || defined(ONLY_SPIRV_COMPILATION)))
    if(!is_extension_available(device, "cl_khr_subgroup_named_barrier"))
    {
        log_info("SKIPPED: Extension `cl_khr_subgroup_named_barrier` is not supported. Skipping tests.\n");
        return CL_SUCCESS;
    }
#endif

    int error = CL_SUCCESS;
    int last_error = CL_SUCCESS;

    RUN_WG_NAMED_BARRIER_TEST_MACRO(spec_example_work_group_named_barrier_test())

    if(error != CL_SUCCESS)
    {
        return -1;
    }
    return error;
}

} // namespace

#endif // TEST_CONFORMANCE_CLCPP_SYNCHRONIZATION_NAMED_BARRIER_TEST_SPEC_EXAMPLE_HPP
