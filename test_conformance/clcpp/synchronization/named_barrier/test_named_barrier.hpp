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
#ifndef TEST_CONFORMANCE_CLCPP_SYNCHRONIZATION_NAMED_BARRIER_TEST_NAMED_BARRIER_HPP
#define TEST_CONFORMANCE_CLCPP_SYNCHRONIZATION_NAMED_BARRIER_TEST_NAMED_BARRIER_HPP

#include "common.hpp"

namespace named_barrier {

struct local_fence_named_barrier_test : public work_group_named_barrier_test_base
{
    std::string str()
    {
        return "local_fence";
    }

    // Return value that is expected to be in output_buffer[i]
    cl_uint operator()(size_t i, size_t work_group_size, size_t max_sub_group_size)
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
                "__kernel void " + this->get_kernel_name() + "(global uint *output, "
                                                              "local uint * lmem)\n"
                "{\n"
                "  size_t gid = get_global_id(0);\n"
                "  output[gid] = gid;\n"
                "}\n";

        #else
            return
                "#define cl_khr_subgroup_named_barrier\n"
                "#include <opencl_memory>\n"
                "#include <opencl_work_item>\n"
                "#include <opencl_synchronization>\n"
                "using namespace cl;\n"
                "__kernel void " + this->get_kernel_name() + "(global_ptr<uint[]> output, "
                                                              "local_ptr<uint[]> lmem)\n"
                "{\n\n"
                "  local<work_group_named_barrier> a(1);\n"
                "  local<work_group_named_barrier> b(2);\n"
                "  size_t gid = get_global_id(0);\n"
                "  size_t lid = get_local_id(0);\n"
                "  size_t value;\n"
                "  if(get_num_sub_groups() == 1)\n"
                "  {\n"
                "    size_t other_lid = (lid + 1) % get_enqueued_local_size(0);\n"
                "    size_t other_gid = (gid - lid) + other_lid;\n"
                "    lmem[other_lid] = other_gid;\n"
                "    a.wait(mem_fence::local);\n"
                "    value = lmem[lid];" // lmem[lid] shoule be equal to gid
                "  }\n"
                "  else if(get_num_sub_groups() == 2)\n"
                "  {\n"
                "    size_t other_lid = (lid + get_max_sub_group_size()) % get_enqueued_local_size(0);\n"
                "    size_t other_gid = (gid - lid) + other_lid;\n"
                "    lmem[other_lid] = other_gid;\n"
                "    b.wait(mem_fence::local);\n"
                "    value = lmem[lid];" // lmem[lid] shoule be equal to gid
                "  }\n"
                "  else if(get_num_sub_groups() > 2)\n"
                "  {\n"
                "    if(get_sub_group_id() < 2)\n"
                "    {\n"
                "      const size_t two_first_subgroups = 2 * get_max_sub_group_size();"
                       // local and global id of some work-item outside of work-item subgroup,
                       // but within subgroups 0 and 1.
                "      size_t other_lid = (lid + get_max_sub_group_size()) % two_first_subgroups;\n"
                "      size_t other_gid = (gid - lid) + other_lid;\n"
                "      lmem[other_lid] = other_gid;\n"
                "      b.wait(mem_fence::local);\n" // subgroup 0 and 1 are sync (local)
                "      value = lmem[lid];" // lmem[lid] shoule be equal to gid
                "    }\n"
                "    else\n"
                "    {\n"
                "      value = gid;\n"
                "    }\n"
                "  }\n"
                "  output[gid] = value;\n"
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
            error = clSetKernelArg(kernel, 1, wg_size * sizeof(cl_uint), NULL);
            RETURN_ON_CL_ERROR(error, "clSetKernelArg")

            size_t max_wg_size;
            error = clGetKernelWorkGroupInfo(
                kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &max_wg_size, NULL
            );
            RETURN_ON_CL_ERROR(error, "clGetKernelWorkGroupInfo")
            if(max_wg_size >= wg_size) break;
        }
        return wg_size;
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

        err = clSetKernelArg(kernel, 0, sizeof(output_buffer), &output_buffer);
        err |= clSetKernelArg(kernel, 1, work_group_size * sizeof(cl_uint), NULL);
        RETURN_ON_CL_ERROR(err, "clSetKernelArg")

        err = clEnqueueNDRangeKernel(
            queue, kernel, 1,
            NULL, &work_size, &work_group_size,
            0, NULL, NULL
        );
        RETURN_ON_CL_ERROR(err, "clEnqueueNDRangeKernel")

        err = clFinish(queue);
        return err;
    }
};

struct global_fence_named_barrier_test : public work_group_named_barrier_test_base
{
    std::string str()
    {
        return "global_fence";
    }

    // Return value that is expected to be in output_buffer[i]
    cl_uint operator()(size_t i, size_t work_group_size, size_t max_sub_group_size)
    {
        return static_cast<cl_uint>(i % work_group_size);
    }

    // At the end every work-item writes its local id to ouput[work-item-global-id].
    std::string generate_program()
    {
        // -----------------------------------------------------------------------------------
        // ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
        // -----------------------------------------------------------------------------------
        #if defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
            return
                "__kernel void " + this->get_kernel_name() + "(global uint * output, "
                                                              "global uint * temp)\n"
                "{\n"
                "size_t gid = get_global_id(0);\n"
                "output[gid] = get_local_id(0);\n"
                "}\n";

        #else
            return
                "#define cl_khr_subgroup_named_barrier\n"
                "#include <opencl_memory>\n"
                "#include <opencl_work_item>\n"
                "#include <opencl_synchronization>\n"
                "using namespace cl;\n"
                "__kernel void " + this->get_kernel_name() + "(global_ptr<uint[]> output, "
                                                              "global_ptr<uint[]> temp)\n"
                "{\n\n"
                "  local<work_group_named_barrier> a(1);\n"
                "  local<work_group_named_barrier> b(2);\n"
                "  size_t gid = get_global_id(0);\n"
                "  size_t lid = get_local_id(0);\n"
                "  size_t value;\n"
                "  if(get_num_sub_groups() == 1)\n"
                "  {\n"
                "    size_t other_lid = (lid + 1) % get_enqueued_local_size(0);\n"
                "    size_t other_gid = (gid - lid) + other_lid;\n"
                "    temp[other_gid] = other_lid + 1;\n"
                "    a.wait(mem_fence::global);\n"
                "    size_t other_lid_same_subgroup = (lid + 2) % get_sub_group_size();\n"
                "    size_t other_gid_same_subgroup = (gid - lid) + other_lid_same_subgroup;\n"
                "    temp[other_gid_same_subgroup] = temp[other_gid_same_subgroup] - 1;\n"
                "    a.wait(mem_fence::global, memory_scope_sub_group);\n"
                "    value = temp[gid];" // temp[gid] shoule be equal to lid
                "  }\n"
                "  else if(get_num_sub_groups() == 2)\n"
                "  {\n"
                "    size_t other_lid = (lid + get_max_sub_group_size()) % get_enqueued_local_size(0);\n"
                "    size_t other_gid = (gid - lid) + other_lid;\n"
                "    temp[other_gid] = other_lid + 1;\n"
                "    b.wait(mem_fence::global);\n" // both subgroups wait, both are sync
                "    size_t other_lid_same_subgroup = "
                       "((lid + 1) % get_sub_group_size()) + (get_sub_group_id() * get_sub_group_size());\n"
                "    size_t other_gid_same_subgroup = (gid - lid) + other_lid_same_subgroup;\n"
                "    temp[other_gid_same_subgroup] = temp[other_gid_same_subgroup] - 1;\n"
                "    b.wait(mem_fence::global, memory_scope_sub_group);\n"  // both subgroups wait, sync only within subgroup
                "    value = temp[gid];" // temp[gid] shoule be equal to lid
                "  }\n"
                "  else if(get_num_sub_groups() > 2)\n"
                "  {\n"
                "    if(get_sub_group_id() < 2)\n"
                "    {\n"
                "      const size_t two_first_subgroups = 2 * get_max_sub_group_size();"
                       // local and global id of some work-item outside of work-item subgroup,
                       // but within subgroups 0 and 1.
                "      size_t other_lid = (lid + get_max_sub_group_size()) % two_first_subgroups;\n"
                "      size_t other_gid = (gid - lid) + other_lid;\n"
                "      temp[other_gid] = other_lid + 1;\n"
                "      b.wait(mem_fence::global);\n" // both subgroups wait, both are sync
                       // local and global id of some other work-item within work-item subgroup
                "      size_t other_lid_same_subgroup = "
                         "((lid + 1) % get_sub_group_size()) + (get_sub_group_id() * get_sub_group_size());\n"
                "      size_t other_gid_same_subgroup = (gid - lid) + other_lid_same_subgroup;\n"
                "      temp[other_gid_same_subgroup] = temp[other_gid_same_subgroup] - 1;\n"
                "      b.wait(mem_fence::global, memory_scope_sub_group);\n" // both subgroups wait, sync only within subgroup
                "      value = temp[gid];" // temp[gid] shoule be equal to lid
                "    }\n"
                "    else\n"
                "    {\n"
                "      value = lid;\n"
                "    }\n"
                "  }\n"
                "  output[gid] = value;\n"
                "}\n";
        #endif
    }

    size_t get_max_local_size(const cl_kernel kernel,
                              const cl_device_id device,
                              const size_t work_group_size, // default work-group size
                              cl_int& error)
    {
        size_t max_wg_size;
        error = clGetKernelWorkGroupInfo(
            kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &max_wg_size, NULL
        );
        RETURN_ON_CL_ERROR(error, "clGetKernelWorkGroupInfo")
        return (std::min)(max_wg_size, work_group_size);
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

struct global_local_fence_named_barrier_test : public work_group_named_barrier_test_base
{
    std::string str()
    {
        return "global_local_fence";
    }

    // Return value that is expected to be in output_buffer[i]
    cl_uint operator()(size_t i, size_t work_group_size, size_t max_sub_group_size)
    {
        return static_cast<cl_uint>(i % work_group_size);
    }

    // At the end every work-item writes its local id to ouput[work-item-global-id].
    std::string generate_program()
    {
        // -----------------------------------------------------------------------------------
        // ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
        // -----------------------------------------------------------------------------------
        #if defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
            return
                "__kernel void " + this->get_kernel_name() + "(global uint * output, "
                                                              "global uint * temp,"
                                                              "local uint * lmem)\n"
                "{\n"
                "size_t gid = get_global_id(0);\n"
                "output[gid] = get_local_id(0);\n"
                "}\n";

        #else
            return
                "#define cl_khr_subgroup_named_barrier\n"
                "#include <opencl_memory>\n"
                "#include <opencl_work_item>\n"
                "#include <opencl_synchronization>\n"
                "using namespace cl;\n"
                "__kernel void " + this->get_kernel_name() + "(global_ptr<uint[]> output, "
                                                              "global_ptr<uint[]> temp,"
                                                              "local_ptr<uint[]> lmem)\n"
                "{\n\n"
                "  local<work_group_named_barrier> a(1);\n"
                "  local<work_group_named_barrier> b(2);\n"
                "  size_t gid = get_global_id(0);\n"
                "  size_t lid = get_local_id(0);\n"
                "  size_t value = 0;\n"
                "  if(get_num_sub_groups() == 1)\n"
                "  {\n"
                "    size_t other_lid = (lid + 1) % get_enqueued_local_size(0);\n"
                "    size_t other_gid = (gid - lid) + other_lid;\n"
                "    lmem[other_lid] = other_gid;\n"
                "    temp[other_gid] = other_lid;\n"
                "    a.wait(mem_fence::local | mem_fence::global);\n"
                "    if(lmem[lid] == gid) value = temp[gid];\n"
                "  }\n"
                "  else if(get_num_sub_groups() == 2)\n"
                "  {\n"
                "    size_t other_lid = (lid + get_max_sub_group_size()) % get_enqueued_local_size(0);\n"
                "    size_t other_gid = (gid - lid) + other_lid;\n"
                "    lmem[other_lid] = other_gid;\n"
                "    temp[other_gid] = other_lid;\n"
                "    b.wait(mem_fence::local | mem_fence::global);\n"
                "    if(lmem[lid] == gid) value = temp[gid];\n"
                "  }\n"
                "  else if(get_num_sub_groups() > 2)\n"
                "  {\n"
                "    if(get_sub_group_id() < 2)\n"
                "    {\n"
                "      const size_t two_first_subgroups = 2 * get_max_sub_group_size();"
                       // local and global id of some work-item outside of work-item subgroup,
                       // but within subgroups 0 and 1.
                "      size_t other_lid = (lid + get_max_sub_group_size()) % two_first_subgroups;\n"
                "      size_t other_gid = (gid - lid) + other_lid;\n"
                "      lmem[other_lid] = other_gid;\n"
                "      temp[other_gid] = other_lid;\n"
                "      b.wait(mem_fence::local | mem_fence::global);\n"
                "      if(lmem[lid] == gid) value = temp[gid];\n"
                "    }\n"
                "    else\n"
                "    {\n"
                "      value = lid;\n"
                "    }\n"
                "  }\n"
                "  output[gid] = value;\n"
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
            error = clSetKernelArg(kernel, 2, wg_size * sizeof(cl_uint), NULL);
            RETURN_ON_CL_ERROR(error, "clSetKernelArg")

            size_t max_wg_size;
            error = clGetKernelWorkGroupInfo(
                kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &max_wg_size, NULL
            );
            RETURN_ON_CL_ERROR(error, "clGetKernelWorkGroupInfo")
            if(max_wg_size >= wg_size) break;
        }
        return wg_size;
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
        auto temp_buffer = clCreateBuffer(
            context, (cl_mem_flags)(CL_MEM_READ_WRITE),
            sizeof(cl_uint) * work_size, NULL, &err
        );
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
AUTO_TEST_CASE(test_work_group_named_barrier)
(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    int error = CL_SUCCESS;
    int last_error = CL_SUCCESS;

#if !(defined(DEVELOPMENT) && (defined(USE_OPENCLC_KERNELS) || defined(ONLY_SPIRV_COMPILATION)))
    if(!is_extension_available(device, "cl_khr_subgroup_named_barrier"))
    {
        log_info("SKIPPED: Extension `cl_khr_subgroup_named_barrier` is not supported. Skipping tests.\n");
        return CL_SUCCESS;
    }

    // An implementation shall support at least 8 named barriers per work-group. The exact
    // maximum number can be queried using clGetDeviceInfo with CL_DEVICE_MAX_NAMED_BARRIER_COUNT_KHR
    // from the OpenCL 2.2 Extension Specification.
    cl_uint named_barrier_count;
    error = clGetDeviceInfo(device, CL_DEVICE_MAX_NAMED_BARRIER_COUNT_KHR, sizeof(cl_uint), &named_barrier_count, NULL);
    RETURN_ON_CL_ERROR(error, "clGetDeviceInfo")

    if(named_barrier_count < 8)
    {
        RETURN_ON_ERROR_MSG(-1, "Maximum number of named barriers must be at least 8.");
    }
#endif

    RUN_WG_NAMED_BARRIER_TEST_MACRO(local_fence_named_barrier_test())
    RUN_WG_NAMED_BARRIER_TEST_MACRO(global_fence_named_barrier_test())
    RUN_WG_NAMED_BARRIER_TEST_MACRO(global_local_fence_named_barrier_test())

    if(error != CL_SUCCESS)
    {
        return -1;
    }
    return error;
}

} // namespace

#endif // TEST_CONFORMANCE_CLCPP_SYNCHRONIZATION_NAMED_BARRIER_TEST_NAMED_BARRIER_HPP
