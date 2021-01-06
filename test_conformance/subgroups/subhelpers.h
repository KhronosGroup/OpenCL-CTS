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
#ifndef SUBHELPERS_H
#define SUBHELPERS_H

#include "testHarness.h"
#include "kernelHelpers.h"
#include "typeWrappers.h"

#include <limits>
#include <vector>

class subgroupsAPI {
public:
    subgroupsAPI(cl_platform_id platform, bool useCoreSubgroups)
    {
        static_assert(CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE
                          == CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE_KHR,
                      "Enums have to be the same");
        static_assert(CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE
                          == CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE_KHR,
                      "Enums have to be the same");
        if (useCoreSubgroups)
        {
            _clGetKernelSubGroupInfo_ptr = &clGetKernelSubGroupInfo;
            clGetKernelSubGroupInfo_name = "clGetKernelSubGroupInfo";
        }
        else
        {
            _clGetKernelSubGroupInfo_ptr = (clGetKernelSubGroupInfoKHR_fn)
                clGetExtensionFunctionAddressForPlatform(
                    platform, "clGetKernelSubGroupInfoKHR");
            clGetKernelSubGroupInfo_name = "clGetKernelSubGroupInfoKHR";
        }
    }
    clGetKernelSubGroupInfoKHR_fn clGetKernelSubGroupInfo_ptr()
    {
        return _clGetKernelSubGroupInfo_ptr;
    }
    const char *clGetKernelSubGroupInfo_name;

private:
    clGetKernelSubGroupInfoKHR_fn _clGetKernelSubGroupInfo_ptr;
};

// Some template helpers
template <typename Ty> struct TypeName;
template <> struct TypeName<cl_half>
{
    static const char *val() { return "half"; }
};
template <> struct TypeName<cl_uint>
{
    static const char *val() { return "uint"; }
};
template <> struct TypeName<cl_int>
{
    static const char *val() { return "int"; }
};
template <> struct TypeName<cl_ulong>
{
    static const char *val() { return "ulong"; }
};
template <> struct TypeName<cl_long>
{
    static const char *val() { return "long"; }
};
template <> struct TypeName<float>
{
    static const char *val() { return "float"; }
};
template <> struct TypeName<double>
{
    static const char *val() { return "double"; }
};

template <typename Ty> struct TypeDef;
template <> struct TypeDef<cl_half>
{
    static const char *val() { return "typedef half Type;\n"; }
};
template <> struct TypeDef<cl_uint>
{
    static const char *val() { return "typedef uint Type;\n"; }
};
template <> struct TypeDef<cl_int>
{
    static const char *val() { return "typedef int Type;\n"; }
};
template <> struct TypeDef<cl_ulong>
{
    static const char *val() { return "typedef ulong Type;\n"; }
};
template <> struct TypeDef<cl_long>
{
    static const char *val() { return "typedef long Type;\n"; }
};
template <> struct TypeDef<float>
{
    static const char *val() { return "typedef float Type;\n"; }
};
template <> struct TypeDef<double>
{
    static const char *val() { return "typedef double Type;\n"; }
};

template <typename Ty, int Which> struct TypeIdentity;
// template <> struct TypeIdentity<cl_half,0> { static cl_half val() { return
// (cl_half)0.0; } }; template <> struct TypeIdentity<cl_half,0> { static
// cl_half val() { return -(cl_half)65536.0; } }; template <> struct
// TypeIdentity<cl_half,0> { static cl_half val() { return (cl_half)65536.0; }
// };

template <> struct TypeIdentity<cl_uint, 0>
{
    static cl_uint val() { return (cl_uint)0; }
};
template <> struct TypeIdentity<cl_uint, 1>
{
    static cl_uint val() { return (cl_uint)0; }
};
template <> struct TypeIdentity<cl_uint, 2>
{
    static cl_uint val() { return (cl_uint)0xffffffff; }
};

template <> struct TypeIdentity<cl_int, 0>
{
    static cl_int val() { return (cl_int)0; }
};
template <> struct TypeIdentity<cl_int, 1>
{
    static cl_int val() { return (cl_int)0x80000000; }
};
template <> struct TypeIdentity<cl_int, 2>
{
    static cl_int val() { return (cl_int)0x7fffffff; }
};

template <> struct TypeIdentity<cl_ulong, 0>
{
    static cl_ulong val() { return (cl_ulong)0; }
};
template <> struct TypeIdentity<cl_ulong, 1>
{
    static cl_ulong val() { return (cl_ulong)0; }
};
template <> struct TypeIdentity<cl_ulong, 2>
{
    static cl_ulong val() { return (cl_ulong)0xffffffffffffffffULL; }
};

template <> struct TypeIdentity<cl_long, 0>
{
    static cl_long val() { return (cl_long)0; }
};
template <> struct TypeIdentity<cl_long, 1>
{
    static cl_long val() { return (cl_long)0x8000000000000000ULL; }
};
template <> struct TypeIdentity<cl_long, 2>
{
    static cl_long val() { return (cl_long)0x7fffffffffffffffULL; }
};


template <> struct TypeIdentity<float, 0>
{
    static float val() { return 0.F; }
};
template <> struct TypeIdentity<float, 1>
{
    static float val() { return -std::numeric_limits<float>::infinity(); }
};
template <> struct TypeIdentity<float, 2>
{
    static float val() { return std::numeric_limits<float>::infinity(); }
};

template <> struct TypeIdentity<double, 0>
{
    static double val() { return 0.L; }
};

template <> struct TypeIdentity<double, 1>
{
    static double val() { return -std::numeric_limits<double>::infinity(); }
};
template <> struct TypeIdentity<double, 2>
{
    static double val() { return std::numeric_limits<double>::infinity(); }
};

template <typename Ty> struct TypeCheck;
template <> struct TypeCheck<cl_uint>
{
    static bool val(cl_device_id) { return true; }
};
template <> struct TypeCheck<cl_int>
{
    static bool val(cl_device_id) { return true; }
};

static bool int64_ok(cl_device_id device)
{
    char profile[128];
    int error;

    error = clGetDeviceInfo(device, CL_DEVICE_PROFILE, sizeof(profile),
                            (void *)&profile, NULL);
    if (error)
    {
        log_info("clGetDeviceInfo failed with CL_DEVICE_PROFILE\n");
        return false;
    }

    if (strcmp(profile, "EMBEDDED_PROFILE") == 0)
        return is_extension_available(device, "cles_khr_int64");

    return true;
}

template <> struct TypeCheck<cl_ulong>
{
    static bool val(cl_device_id device) { return int64_ok(device); }
};
template <> struct TypeCheck<cl_long>
{
    static bool val(cl_device_id device) { return int64_ok(device); }
};
template <> struct TypeCheck<cl_float>
{
    static bool val(cl_device_id) { return true; }
};
template <> struct TypeCheck<cl_half>
{
    static bool val(cl_device_id device)
    {
        return is_extension_available(device, "cl_khr_fp16");
    }
};
template <> struct TypeCheck<double>
{
    static bool val(cl_device_id device)
    {
        int error;
        cl_device_fp_config c;
        error = clGetDeviceInfo(device, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(c),
                                (void *)&c, NULL);
        if (error)
        {
            log_info(
                "clGetDeviceInfo failed with CL_DEVICE_DOUBLE_FP_CONFIG\n");
            return false;
        }
        return c != 0;
    }
};


// Run a test kernel to compute the result of a built-in on an input
static int run_kernel(cl_context context, cl_command_queue queue,
                      cl_kernel kernel, size_t global, size_t local,
                      void *idata, size_t isize, void *mdata, size_t msize,
                      void *odata, size_t osize, size_t tsize = 0)
{
    clMemWrapper in;
    clMemWrapper xy;
    clMemWrapper out;
    clMemWrapper tmp;
    int error;

    in = clCreateBuffer(context, CL_MEM_READ_ONLY, isize, NULL, &error);
    test_error(error, "clCreateBuffer failed");

    xy = clCreateBuffer(context, CL_MEM_WRITE_ONLY, msize, NULL, &error);
    test_error(error, "clCreateBuffer failed");

    out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, osize, NULL, &error);
    test_error(error, "clCreateBuffer failed");

    if (tsize)
    {
        tmp = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
                             tsize, NULL, &error);
        test_error(error, "clCreateBuffer failed");
    }

    error = clSetKernelArg(kernel, 0, sizeof(in), (void *)&in);
    test_error(error, "clSetKernelArg failed");

    error = clSetKernelArg(kernel, 1, sizeof(xy), (void *)&xy);
    test_error(error, "clSetKernelArg failed");

    error = clSetKernelArg(kernel, 2, sizeof(out), (void *)&out);
    test_error(error, "clSetKernelArg failed");

    if (tsize)
    {
        error = clSetKernelArg(kernel, 3, sizeof(tmp), (void *)&tmp);
        test_error(error, "clSetKernelArg failed");
    }

    error = clEnqueueWriteBuffer(queue, in, CL_FALSE, 0, isize, idata, 0, NULL,
                                 NULL);
    test_error(error, "clEnqueueWriteBuffer failed");

    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0,
                                   NULL, NULL);
    test_error(error, "clEnqueueNDRangeKernel failed");

    error = clEnqueueReadBuffer(queue, xy, CL_FALSE, 0, msize, mdata, 0, NULL,
                                NULL);
    test_error(error, "clEnqueueReadBuffer failed");

    error = clEnqueueReadBuffer(queue, out, CL_FALSE, 0, osize, odata, 0, NULL,
                                NULL);
    test_error(error, "clEnqueueReadBuffer failed");

    error = clFinish(queue);
    test_error(error, "clFinish failed");

    return error;
}

// Driver for testing a single built in function
template <typename Ty, typename Fns, size_t GSIZE, size_t LSIZE,
          size_t TSIZE = 0>
struct test
{
    static int run(cl_device_id device, cl_context context,
                   cl_command_queue queue, int num_elements, const char *kname,
                   const char *src, int dynscl, bool useCoreSubgroups)
    {
        size_t tmp;
        int error;
        int subgroup_size, num_subgroups;
        size_t realSize;
        size_t global;
        size_t local;
        clProgramWrapper program;
        clKernelWrapper kernel;
        cl_platform_id platform;
        cl_int sgmap[2 * GSIZE];
        Ty mapin[LSIZE];
        Ty mapout[LSIZE];

        // Make sure a test of type Ty is supported by the device
        if (!TypeCheck<Ty>::val(device)) return 0;

        error = clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(platform),
                                (void *)&platform, NULL);
        test_error(error, "clGetDeviceInfo failed for CL_DEVICE_PLATFORM");
        std::stringstream kernel_sstr;
        if (useCoreSubgroups)
        {
            kernel_sstr
                << "#pragma OPENCL EXTENSION cl_khr_subgroups : enable\n";
        }
        kernel_sstr << "#define XY(M,I) M[I].x = get_sub_group_local_id(); "
                       "M[I].y = get_sub_group_id();\n";
        kernel_sstr << TypeDef<Ty>::val();
        kernel_sstr << src;
        const std::string &kernel_str = kernel_sstr.str();
        const char *kernel_src = kernel_str.c_str();

        error = create_single_kernel_helper_with_build_options(
            context, &program, &kernel, 1, &kernel_src, kname, "-cl-std=CL2.0");
        if (error != 0) return error;

        // Determine some local dimensions to use for the test.
        global = GSIZE;
        error = get_max_common_work_group_size(context, kernel, GSIZE, &local);
        test_error(error, "get_max_common_work_group_size failed");

        // Limit it a bit so we have muliple work groups
        // Ideally this will still be large enough to give us multiple subgroups
        if (local > LSIZE) local = LSIZE;

        // Get the sub group info
        subgroupsAPI subgroupsApiSet(platform, useCoreSubgroups);
        clGetKernelSubGroupInfoKHR_fn clGetKernelSubGroupInfo_ptr =
            subgroupsApiSet.clGetKernelSubGroupInfo_ptr();
        if (clGetKernelSubGroupInfo_ptr == NULL)
        {
            log_error("ERROR: %s function not available",
                      subgroupsApiSet.clGetKernelSubGroupInfo_name);
            return TEST_FAIL;
        }
        error = clGetKernelSubGroupInfo_ptr(
            kernel, device, CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE,
            sizeof(local), (void *)&local, sizeof(tmp), (void *)&tmp, NULL);
        if (error != CL_SUCCESS)
        {
            log_error("ERROR: %s function error for "
                      "CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE",
                      subgroupsApiSet.clGetKernelSubGroupInfo_name);
            return TEST_FAIL;
        }

        subgroup_size = (int)tmp;

        error = clGetKernelSubGroupInfo_ptr(
            kernel, device, CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE,
            sizeof(local), (void *)&local, sizeof(tmp), (void *)&tmp, NULL);
        if (error != CL_SUCCESS)
        {
            log_error("ERROR: %s function error for "
                      "CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE",
                      subgroupsApiSet.clGetKernelSubGroupInfo_name);
            return TEST_FAIL;
        }

        num_subgroups = (int)tmp;
        // Make sure the number of sub groups is what we expect
        if (num_subgroups != (local + subgroup_size - 1) / subgroup_size)
        {
            log_error("ERROR: unexpected number of subgroups (%d) returned\n",
                      num_subgroups);
            return TEST_FAIL;
        }

        std::vector<Ty> idata;
        std::vector<Ty> odata;
        size_t input_array_size = GSIZE;
        size_t output_array_size = GSIZE;

        if (dynscl != 0)
        {
            input_array_size =
                (int)global / (int)local * num_subgroups * dynscl;
            output_array_size = (int)global / (int)local * dynscl;
        }

        idata.resize(input_array_size);
        odata.resize(output_array_size);

        // Run the kernel once on zeroes to get the map
        memset(&idata[0], 0, input_array_size * sizeof(Ty));
        error = run_kernel(context, queue, kernel, global, local, &idata[0],
                           input_array_size * sizeof(Ty), sgmap,
                           global * sizeof(cl_int) * 2, &odata[0],
                           output_array_size * sizeof(Ty), TSIZE * sizeof(Ty));
        if (error) return error;

        // Generate the desired input for the kernel
        Fns::gen(&idata[0], mapin, sgmap, subgroup_size, (int)local,
                 (int)global / (int)local);

        error = run_kernel(context, queue, kernel, global, local, &idata[0],
                           input_array_size * sizeof(Ty), sgmap,
                           global * sizeof(cl_int) * 2, &odata[0],
                           output_array_size * sizeof(Ty), TSIZE * sizeof(Ty));
        if (error) return error;


        // Check the result
        return Fns::chk(&idata[0], &odata[0], mapin, mapout, sgmap,
                        subgroup_size, (int)local, (int)global / (int)local);
    }
};

#endif
