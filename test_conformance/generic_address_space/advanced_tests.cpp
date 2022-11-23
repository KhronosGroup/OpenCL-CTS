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
#include "harness/testHarness.h"
#include "harness/typeWrappers.h"
#include "base.h"

#include <string>
#include <vector>
#include <algorithm>
#include <sstream>

typedef enum {
    ARG_TYPE_NONE,

    ARG_TYPE_HOST_PTR,
    ARG_TYPE_HOST_LOCAL,

    ARG_TYPE_COARSE_GRAINED_SVM,
    ARG_TYPE_FINE_GRAINED_BUFFER_SVM,
    ARG_TYPE_FINE_GRAINED_SYSTEM_SVM,
    ARG_TYPE_ATOMICS_SVM
} ExtraKernelArgMemType;

class CSVMWrapper {
public:
    CSVMWrapper() : ptr_(NULL), context_(NULL) { }

    void Attach(cl_context context, void *ptr) {
        context_ = context;
        ptr_ = ptr;
    }

    ~CSVMWrapper() {
        if (ptr_)
            clSVMFree(context_, ptr_);
    }

    operator void *() {
        return ptr_;
    }

private:
    void *ptr_;
    cl_context context_;
};

class CAdvancedTest : public CTest {
public:
    CAdvancedTest(const std::vector<std::string>& kernel) : CTest(), _kernels(kernel), _extraKernelArgMemType(ARG_TYPE_NONE) {

    }

    CAdvancedTest(const std::string& library, const std::vector<std::string>& kernel) : CTest(), _libraryCode(library), _kernels(kernel), _extraKernelArgMemType(ARG_TYPE_NONE) {

    }

    CAdvancedTest(const std::string& kernel, ExtraKernelArgMemType argType = ARG_TYPE_NONE) : CTest(), _kernels(1, kernel), _extraKernelArgMemType(argType) {

    }

    CAdvancedTest(const std::string& library, const std::string& kernel) : CTest(), _libraryCode(library), _kernels(1, kernel), _extraKernelArgMemType(ARG_TYPE_NONE) {

    }

    int PrintCompilationLog(cl_program program, cl_device_id device) {
        cl_int error;
        size_t buildLogSize = 0;

        error = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &buildLogSize);
        test_error(error, "clGetProgramBuildInfo failed");

        std::string log;
        log.resize(buildLogSize);

        error = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, buildLogSize, &log[0], NULL);
        test_error(error, "clGetProgramBuildInfo failed");

        log_error("Build log for device is:\n------------\n");
        log_error("%s\n", log.c_str() );
        log_error( "\n----------\n" );

        return CL_SUCCESS;
    }

    int ExecuteSubcase(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements, const std::string& src) {
        cl_int error;

        clProgramWrapper program, preCompiledLibrary, library, finalProgram;
        clKernelWrapper kernel;

        const char *srcPtr = src.c_str();

        if (!_libraryCode.empty()) {
            program = clCreateProgramWithSource(context, 1, &srcPtr, NULL, &error);
            test_error(error, "clCreateProgramWithSource failed");

            // Use the latest OpenCL-C version supported by the device. This
            // allows calling code to force a particular CL C version if it is
            // required, but also means that callers need not specify a version
            // if they want to assume the most recent CL C.

            auto version = get_max_OpenCL_C_for_context(context);

            const char* cl_std = nullptr;
            if (version >= Version(3, 0))
            {
                cl_std = "-cl-std=CL3.0";
            }
            else if (version >= Version(2, 0) && version < Version(3, 0))
            {
                cl_std = "-cl-std=CL2.0";
            }
            else
            {
                // If the -cl-std build option is not specified, the highest
                // OpenCL C 1.x language version supported by each device is
                // used when compiling the program for each device.
                cl_std = "";
            }

            error = clCompileProgram(program, 1, &deviceID, cl_std, 0, NULL,
                                     NULL, NULL, NULL);

            if (error != CL_SUCCESS)
                PrintCompilationLog(program, deviceID);
            test_error(error, "clCompileProgram failed");

            const char *srcPtrLibrary = _libraryCode.c_str();

            preCompiledLibrary = clCreateProgramWithSource(context, 1, &srcPtrLibrary, NULL, &error);
            test_error(error, "clCreateProgramWithSource failed");

            error = clCompileProgram(preCompiledLibrary, 1, &deviceID, cl_std,
                                     0, NULL, NULL, NULL, NULL);

            if (error != CL_SUCCESS)
                PrintCompilationLog(preCompiledLibrary, deviceID);
            test_error(error, "clCompileProgram failed");

            library = clLinkProgram(context, 1, &deviceID, "-create-library", 1, &preCompiledLibrary, NULL, NULL, &error);
            test_error(error, "clLinkProgram failed");

            cl_program objects[] = { program, library };
            finalProgram = clLinkProgram(context, 1, &deviceID, "", 2, objects, NULL, NULL, &error);
            test_error(error, "clLinkProgram failed");

            kernel = clCreateKernel(finalProgram, "testKernel", &error);
            test_error(error, "clCreateKernel failed");
        }

        else {
            if (create_single_kernel_helper(context, &program, &kernel, 1,
                                            &srcPtr, "testKernel"))
            {
                log_error("create_single_kernel_helper failed\n");
                return -1;
            }
        }

        size_t bufferSize = num_elements * sizeof(cl_uint);
        clMemWrapper buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bufferSize, NULL, &error);
        test_error(error, "clCreateBuffer failed");

        error = clSetKernelArg(kernel, 0, sizeof(buffer), &buffer);
        test_error(error, "clSetKernelArg(0) failed");

        // Warning: the order below is very important as SVM buffer cannot be free'd before corresponding mem_object
        CSVMWrapper svmWrapper;
        clMemWrapper extraArg;
        std::vector<cl_uint> extraArgData(num_elements);
        for (cl_uint i = 0; i < (cl_uint)num_elements; i++)
            extraArgData[i] = i;

        if (_extraKernelArgMemType != ARG_TYPE_NONE) {
            if (_extraKernelArgMemType == ARG_TYPE_HOST_PTR) {
                extraArg = clCreateBuffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, bufferSize, &extraArgData[0], &error);
                test_error(error, "clCreateBuffer failed");
            }

            else {
                void *ptr = NULL;

                switch (_extraKernelArgMemType) {
                case ARG_TYPE_COARSE_GRAINED_SVM:
                    ptr = clSVMAlloc(context, CL_MEM_READ_WRITE, bufferSize, 0);
                    break;
                case ARG_TYPE_FINE_GRAINED_BUFFER_SVM:
                    ptr = clSVMAlloc(context, CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_READ_WRITE, bufferSize, 0);
                    break;
                case ARG_TYPE_FINE_GRAINED_SYSTEM_SVM:
                    ptr = &extraArgData[0];
                    break;
                case ARG_TYPE_ATOMICS_SVM:
                    ptr = clSVMAlloc(context, CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS | CL_MEM_READ_WRITE, bufferSize, 0);
                    break;
                default:
                    break;
                }

                if(_extraKernelArgMemType != ARG_TYPE_HOST_LOCAL) {
                  if (!ptr) {
                    log_error("Allocation failed\n");
                    return -1;
                  }

                  if (_extraKernelArgMemType != ARG_TYPE_FINE_GRAINED_SYSTEM_SVM) {
                  svmWrapper.Attach(context, ptr);
                  }

                  if (_extraKernelArgMemType == ARG_TYPE_COARSE_GRAINED_SVM) {
                    error = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_WRITE, ptr, bufferSize, 0, NULL, NULL);
                    test_error(error, "clEnqueueSVMMap failed");
                  }

                  memcpy(ptr, &extraArgData[0], bufferSize);

                  if (_extraKernelArgMemType == ARG_TYPE_COARSE_GRAINED_SVM) {
                    error = clEnqueueSVMUnmap(queue, ptr, 0, NULL, NULL);
                    test_error(error, "clEnqueueSVMUnmap failed");
                    clFinish(queue);
                  }

                  extraArg = clCreateBuffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, bufferSize, ptr, &error);
                  test_error(error, "clCreateBuffer from SVM buffer failed");
                }
            }

            if(_extraKernelArgMemType == ARG_TYPE_HOST_LOCAL)
              error = clSetKernelArg(kernel, 1, bufferSize, NULL);
            else
              error = clSetKernelArg(kernel, 1, sizeof(extraArg), &extraArg);


            test_error(error, "clSetKernelArg(1) failed");
        }

        size_t globalWorkGroupSize = num_elements;
        size_t localWorkGroupSize = 0;
        error = get_max_common_work_group_size(context, kernel, globalWorkGroupSize, &localWorkGroupSize);
        test_error(error, "Unable to get common work group size");

        error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalWorkGroupSize, &localWorkGroupSize, 0, NULL, NULL);
        test_error(error, "clEnqueueNDRangeKernel failed");

        // verify results
        std::vector<cl_uint> results(num_elements);

        error = clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, bufferSize, &results[0], 0, NULL, NULL);
        test_error(error, "clEnqueueReadBuffer failed");

        size_t passCount = std::count(results.begin(), results.end(), 1);
        if (passCount != results.size()) {
            std::vector<cl_uint>::iterator iter = std::find(results.begin(), results.end(), 0);
            log_error("Verification on device failed at index %ld\n", std::distance(results.begin(), iter));
            log_error("%ld out of %ld failed\n", (results.size()-passCount), results.size());
            return -1;
        }

        return CL_SUCCESS;
    }

    int Execute(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
        cl_int result = CL_SUCCESS;

        for (std::vector<std::string>::const_iterator it = _kernels.begin(); it != _kernels.end(); ++it) {
            log_info("Executing subcase #%ld out of %ld\n", (it - _kernels.begin() + 1), _kernels.size());

            result |= ExecuteSubcase(deviceID, context, queue, num_elements, *it);
        }

        return result;
    }

private:
    const std::string _libraryCode;
    const std::vector<std::string> _kernels;
    const ExtraKernelArgMemType _extraKernelArgMemType;
};

int test_library_function(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    const std::string LIBRARY_FUNCTION = common::CONFORMANCE_VERIFY_FENCE +
        NL
        NL "bool helperFunction(float *floatp, float val) {"
        NL "    if (!isFenceValid(get_fence(floatp)))"
        NL "        return false;"
        NL
        NL "    if (*floatp != val)"
        NL "        return false;"
        NL
        NL "    return true;"
        NL "}"
        NL;

    const std::string KERNEL_FUNCTION = R"OpenCLC(
extern bool helperFunction(float *floatp, float val);

#ifdef __opencl_c_program_scope_global_variables
__global float gfloat = 1.0f;
#endif

__kernel void testKernel(__global uint *results) {
    uint tid = get_global_id(0);

#ifdef __opencl_c_program_scope_global_variables
    __global float *gfloatp = &gfloat;
#endif
    __local float lfloat;
    lfloat = 2.0f;
    __local float *lfloatp = &lfloat;
    float pfloat = 3.0f;
    __private float *pfloatp = &pfloat;

    uint failures = 0;

#ifdef __opencl_c_program_scope_global_variables
    failures += helperFunction(gfloatp, gfloat) ? 0 : 1;
#endif
    failures += helperFunction(lfloatp, lfloat) ? 0 : 1;
    failures += helperFunction(pfloatp, pfloat) ? 0 : 1;

    results[tid] = failures == 0;
};
)OpenCLC";

    CAdvancedTest test(LIBRARY_FUNCTION, KERNEL_FUNCTION);

    return test.Execute(deviceID, context, queue, num_elements);
}

int test_generic_variable_volatile(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    std::vector<std::string> KERNEL_FUNCTIONS;

    KERNEL_FUNCTIONS.push_back(common::CONFORMANCE_VERIFY_FENCE +
        NL
        NL "bool helperFunction(float *floatp, float val) {"
        NL "    if (!isFenceValid(get_fence(floatp)))"
        NL "        return false;"
        NL
        NL "    if (*floatp != val)"
        NL "        return false;"
        NL
        NL "    return true;"
        NL "}"
        NL
        NL "__kernel void testKernel(__global uint *results) {"
        NL "    uint tid = get_global_id(0);"
        NL
        NL "    static __global float val;"
        NL "    val = 0.1f;"
        NL "    float * volatile ptr = &val;"
        NL
        NL "    results[tid] = helperFunction(ptr, val);"
        NL "}"
        NL
    );

    KERNEL_FUNCTIONS.push_back(common::CONFORMANCE_VERIFY_FENCE +
        NL
        NL "bool helperFunction(float *floatp, float val) {"
        NL "    if (!isFenceValid(get_fence(floatp)))"
        NL "        return false;"
        NL
        NL "    if (*floatp != val)"
        NL "        return false;"
        NL
        NL "    return true;"
        NL "}"
        NL
        NL "__kernel void testKernel(__global uint *results) {"
        NL "    uint tid = get_global_id(0);"
        NL
        NL "    __local float val;"
        NL "    val = 0.1f;"
        NL "    float * ptr = &val;"
        NL
        NL "    results[tid] = helperFunction(ptr, val);"
        NL "}"
        NL
    );

    KERNEL_FUNCTIONS.push_back(common::CONFORMANCE_VERIFY_FENCE +
        NL
        NL "bool helperFunction(float *floatp, float val) {"
        NL "    if (!isFenceValid(get_fence(floatp)))"
        NL "        return false;"
        NL
        NL "    if (*floatp != val)"
        NL "        return false;"
        NL
        NL "    return true;"
        NL "}"
        NL
        NL "__kernel void testKernel(__global uint *results) {"
        NL "    uint tid = get_global_id(0);"
        NL
        NL "    __private float val;"
        NL "    val = 0.1f;"
        NL "    float * volatile ptr = &val;"
        NL
        NL "    results[tid] = helperFunction(ptr, val);"
        NL "}"
        NL
    );

    CAdvancedTest test(KERNEL_FUNCTIONS);

    return test.Execute(deviceID, context, queue, num_elements);
}

int test_generic_variable_const(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    std::vector<std::string> KERNEL_FUNCTIONS;

    KERNEL_FUNCTIONS.push_back(common::CONFORMANCE_VERIFY_FENCE +
        NL
        NL "bool helperFunction(const float *floatp, float val) {"
        NL "    if (!isFenceValid(get_fence(floatp)))"
        NL "        return false;"
        NL
        NL "    if (*floatp != val)"
        NL "        return false;"
        NL
        NL "    return true;"
        NL "}"
        NL
        NL "__kernel void testKernel(__global uint *results) {"
        NL "    uint tid = get_global_id(0);"
        NL
        NL "    const __private float val = 0.1f;"
        NL "    const float * ptr = &val;"
        NL
        NL "    results[tid] = helperFunction(ptr, val);"
        NL "}"
        NL
    );

    KERNEL_FUNCTIONS.push_back(common::CONFORMANCE_VERIFY_FENCE +
        NL
        NL "bool helperFunction(const float *floatp, float val) {"
        NL "    if (!isFenceValid(get_fence(floatp)))"
        NL "        return false;"
        NL
        NL "    if (*floatp != val)"
        NL "        return false;"
        NL
        NL "    return true;"
        NL "}"
        NL
        NL "__kernel void testKernel(__global uint *results) {"
        NL "    uint tid = get_global_id(0);"
        NL
        NL "    const static __global float val = 0.1f;"
        NL "    const float * ptr = &val;"
        NL
        NL "    results[tid] = helperFunction(ptr, val);"
        NL "}"
        NL
    );

    CAdvancedTest test(KERNEL_FUNCTIONS);

    return test.Execute(deviceID, context, queue, num_elements);
}

int test_generic_variable_gentype(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    const std::string KERNEL_FUNCTION_TEMPLATE = common::CONFORMANCE_VERIFY_FENCE +
        NL
        NL "%s"
        NL
        NL "bool helperFunction(const %s *%sp, %s val) {"
        NL "    if (!isFenceValid(get_fence(%sp)))"
        NL "        return false;"
        NL
        NL "    return %s(*%sp == val);"
        NL "}"
        NL
        NL "__kernel void testKernel(__global uint *results) {"
        NL "    uint tid = get_global_id(0);"
        NL
        NL "    %s %s val = (%s)1;"
        NL "    %s * ptr = &val;"
        NL
        NL "    results[tid] = helperFunction(ptr, val);"
        NL "}"
        NL;
/* Qualcomm fix: 12502  Gen Addr Space - Fix kernel for generic variable gentype (half) test
   const std::string KERNEL_FUNCTION_TEMPLATE_HALF = common::CONFORMANCE_VERIFY_FENCE */
    const std::string vector_sizes[] = { "", "2", "3", "4", "8", "16" };
    const std::string gentype_base[] = { "float", "char", "uchar", "short", "ushort", "int", "uint", "long", "ulong" };
    const std::string gentype_others[] = { "bool", "size_t", "ptrdiff_t", "intptr_t", "uintptr_t" };

    const std::string address_spaces[] = { "static __global", "__private" };

    const std::string vector_cmp = "all";

    std::vector<std::string> KERNEL_FUNCTIONS;

    // Add base types plus theirs vector variants
    for (size_t i = 0; i < sizeof(gentype_base) / sizeof(gentype_base[0]); i++) {
        for (size_t j = 0; j < sizeof(vector_sizes) / sizeof(vector_sizes[0]); j++) {
            for (size_t k = 0; k < sizeof(address_spaces) / sizeof(address_spaces[0]); k++) {
                char temp_kernel[1024];
                const std::string fulltype = gentype_base[i] + vector_sizes[j];
                sprintf(temp_kernel, KERNEL_FUNCTION_TEMPLATE.c_str(),
                    "",
                    fulltype.c_str(), fulltype.c_str(), fulltype.c_str(), fulltype.c_str(),
                    (j > 0 ? vector_cmp.c_str() : ""),
                    fulltype.c_str(), address_spaces[k].c_str(), fulltype.c_str(), fulltype.c_str(),
                    fulltype.c_str());

                KERNEL_FUNCTIONS.push_back(temp_kernel);
            }
        }
    }

    const std::string cl_khr_fp64_pragma = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable";

    // Add double floating types if they are supported
    if (is_extension_available(deviceID, "cl_khr_fp64")) {
        for (size_t j = 0; j < sizeof(vector_sizes) / sizeof(vector_sizes[0]); j++) {
            for (size_t k = 0; k < sizeof(address_spaces) / sizeof(address_spaces[0]); k++) {
                char temp_kernel[1024];
                const std::string fulltype = std::string("double") + vector_sizes[j];
                sprintf(temp_kernel, KERNEL_FUNCTION_TEMPLATE.c_str(),
                    cl_khr_fp64_pragma.c_str(),
                    fulltype.c_str(), fulltype.c_str(), fulltype.c_str(), fulltype.c_str(),
                    (j > 0 ? vector_cmp.c_str() : ""),
                    fulltype.c_str(), address_spaces[k].c_str(), fulltype.c_str(), fulltype.c_str(),
                    fulltype.c_str());

                KERNEL_FUNCTIONS.push_back(temp_kernel);
            }
        }
    }
/* Qualcomm fix: 12502  Gen Addr Space - Fix kernel for generic variable gentype (half) test */
    const std::string cl_khr_fp16_pragma = "#pragma OPENCL EXTENSION cl_khr_fp16 : enable";

    // Add half floating types if they are supported
    if (is_extension_available(deviceID, "cl_khr_fp16")) {
        for (size_t j = 0; j < sizeof(vector_sizes) / sizeof(vector_sizes[0]); j++) {
            for (size_t k = 0; k < sizeof(address_spaces) / sizeof(address_spaces[0]); k++) {
                char temp_kernel[1024];
                const std::string fulltype = std::string("half") + vector_sizes[j];
                sprintf(temp_kernel, KERNEL_FUNCTION_TEMPLATE.c_str(),
                    cl_khr_fp16_pragma.c_str(),
                    fulltype.c_str(), fulltype.c_str(), fulltype.c_str(), fulltype.c_str(),
                    (j > 0 ? vector_cmp.c_str() : ""),
                    fulltype.c_str(), address_spaces[k].c_str(), fulltype.c_str(), fulltype.c_str(),
                    fulltype.c_str());
/* Qualcomm fix: end */
                KERNEL_FUNCTIONS.push_back(temp_kernel);
            }
        }
    }

    // Add other types that do not have vector variants
    for (size_t i = 0; i < sizeof(gentype_others) / sizeof(gentype_others[0]); i++) {
        for (size_t k = 0; k < sizeof(address_spaces) / sizeof(address_spaces[0]); k++) {
            char temp_kernel[1024];
            const std::string fulltype = gentype_others[i];
            sprintf(temp_kernel, KERNEL_FUNCTION_TEMPLATE.c_str(),
                "",
                fulltype.c_str(), fulltype.c_str(), fulltype.c_str(), fulltype.c_str(),
                "",
                fulltype.c_str(), address_spaces[k].c_str(), fulltype.c_str(), fulltype.c_str(),
                fulltype.c_str());

            KERNEL_FUNCTIONS.push_back(temp_kernel);
        }
    }

    CAdvancedTest test(KERNEL_FUNCTIONS);

    return test.Execute(deviceID, context, queue, num_elements);
}

void create_math_kernels(std::vector<std::string>& KERNEL_FUNCTIONS) {
    const std::string KERNEL_FUNCTION_TEMPLATE =
        NL
        NL "__kernel void testKernel(__global uint *results) {"
        NL "    uint tid = get_global_id(0);"
        NL
        NL "    const %s param1 = %s;"
        NL "    %s param2_generic;"
        NL "    %s param2_reference;"
        NL "    %s * ptr = &param2_generic;"
        NL "    %s return_value_generic;"
        NL "    %s return_value_reference;"
        NL
        NL "    return_value_generic = %s(param1, ptr);"
        NL "    return_value_reference = %s(param1, &param2_reference);"
        NL
        NL "    results[tid] = (%s(*ptr == param2_reference) && %s(return_value_generic == return_value_reference));"
        NL "}"
        NL;

    typedef struct {
        std::string bulitin_name;
        std::string base_gentype;
        std::string pointer_gentype;
        std::string first_param_value;
        std::string compare_fn;
    } BuiltinDescriptor;

    BuiltinDescriptor builtins[] = {
        { "fract", "float", "float", "133.55f", "" },
        { "frexp", "float2", "int2", "(float2)(24.12f, 99999.7f)", "all" },
        { "frexp", "float", "int", "1234.5f", "" },
        { "lgamma_r", "float2", "int2", "(float2)(1000.0f, 9999.5f)", "all" },
        { "lgamma_r", "float", "int", "1000.0f", "" },
        { "modf", "float", "float", "1234.56789f", "" },
        { "sincos", "float", "float", "3.141592f", "" }
    };

    for (size_t i = 0; i < sizeof(builtins) / sizeof(builtins[0]); i++) {
        char temp_kernel[1024];
        sprintf(temp_kernel, KERNEL_FUNCTION_TEMPLATE.c_str(), builtins[i].base_gentype.c_str(), builtins[i].first_param_value.c_str(),
            builtins[i].pointer_gentype.c_str(), builtins[i].pointer_gentype.c_str(), builtins[i].pointer_gentype.c_str(), builtins[i].base_gentype.c_str(),
            builtins[i].base_gentype.c_str(), builtins[i].bulitin_name.c_str(), builtins[i].bulitin_name.c_str(),
            builtins[i].compare_fn.c_str(), builtins[i].compare_fn.c_str());

        KERNEL_FUNCTIONS.push_back(temp_kernel);
    }

    // add special case for remquo (3 params)
    KERNEL_FUNCTIONS.push_back(
        NL
        NL "__kernel void testKernel(__global uint *results) {"
        NL "    uint tid = get_global_id(0);"
        NL
        NL "    const float param1 = 1234.56789f;"
        NL "    const float param2 = 123.456789f;"
        NL "    int param3_generic;"
        NL "    int param3_reference;"
        NL "    int * ptr = &param3_generic;"
        NL "    float return_value_generic;"
        NL "    float return_value_reference;"
        NL
        NL "    return_value_generic = remquo(param1, param2, ptr);"
        NL "    return_value_reference = remquo(param1, param2, &param3_reference);"
        NL
        NL "    results[tid] = (*ptr == param3_reference && return_value_generic == return_value_reference);"
        NL "}"
        NL
    );
}

std::string get_default_data_for_type(const std::string& type) {
    std::string result;

    if (type == "float") {
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                char temp[10];
                sprintf(temp, "%d.%df, ", i, j);
                result += std::string(temp);
            }
        }
    }

    else if (type == "double") {
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                char temp[10];
                sprintf(temp, "%d.%d, ", i, j);
                result += std::string(temp);
            }
        }
    }

    else {
        for (int i = 0; i < 100; i++) {
            char temp[10];
            sprintf(temp, "%d, ", i);
            result += std::string(temp);
        }
    }

    return result;
}

void create_vload_kernels(std::vector<std::string>& KERNEL_FUNCTIONS, cl_device_id deviceID) {
    const std::string KERNEL_FUNCTION_TEMPLATE_GLOBAL =
        NL
        NL "%s"
        NL "__global %s data[] = { %s };"
        NL
        NL "__kernel void testKernel(__global uint *results) {"
        NL "    uint tid = get_global_id(0);"
        NL
        NL "    // Testing: %s"
        NL "    const %s * ptr = data;"
        NL "    %s%s result_generic = vload%s(2, ptr);"
        NL "    %s%s result_reference = vload%s(2, data);"
        NL
        NL "    results[tid] = all(result_generic == result_reference);"
        NL "}"
        NL;

    const std::string KERNEL_FUNCTION_TEMPLATE_LOCAL =
        NL
        NL "%s"
        NL "__constant %s to_copy_from[] = { %s };"
        NL
        NL "__kernel void testKernel(__global uint *results) {"
        NL "    uint tid = get_global_id(0);"
        NL
        NL "    __local %s data[100];"
        NL "    for (int i = 0; i < sizeof(to_copy_from) / sizeof(to_copy_from[0]); i++)"
        NL "        data[i] = to_copy_from[i];"
        NL
        NL "    const %s * ptr = data;"
        NL "    %s%s result_generic = vload%s(2, ptr);"
        NL "    %s%s result_reference = vload%s(2, data);"
        NL
        NL "    results[tid] = all(result_generic == result_reference);"
        NL "}"
        NL;

    const std::string KERNEL_FUNCTION_TEMPLATE_PRIVATE =
        NL
        NL "%s"
        NL "__kernel void testKernel(__global uint *results) {"
        NL "    uint tid = get_global_id(0);"
        NL
        NL "    %s data[] = { %s };"
        NL "    // Testing: %s"
        NL "    const %s * ptr = data;"
        NL "    %s%s result_generic = vload%s(2, ptr);"
        NL "    %s%s result_reference = vload%s(2, data);"
        NL
        NL "    results[tid] = all(result_generic == result_reference);"
        NL "}"
        NL;

    const std::string vector_sizes[] = { "2", "3", "4", "8", "16" };
    const std::string gentype_base[] = { "double", "float", "char", "uchar", "short", "ushort", "int", "uint", "long", "ulong" };
    const std::string kernel_variants[] = { KERNEL_FUNCTION_TEMPLATE_GLOBAL, KERNEL_FUNCTION_TEMPLATE_LOCAL, KERNEL_FUNCTION_TEMPLATE_PRIVATE };

    const std::string cl_khr_fp64_pragma = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable";

    for (size_t i = 0; i < sizeof(gentype_base) / sizeof(gentype_base[0]); i++) {
        const char *pragma_str = "";

        if (i == 0) {
            if (!is_extension_available(deviceID, "cl_khr_fp64"))
                continue;
            else
                pragma_str = cl_khr_fp64_pragma.c_str();
        }

        for (size_t j = 0; j < sizeof(vector_sizes) / sizeof(vector_sizes[0]); j++) {
            for (size_t k = 0; k < sizeof(kernel_variants) / sizeof(kernel_variants[0]); k++) {
                char temp_kernel[4098];
                sprintf(temp_kernel, kernel_variants[k].c_str(),
                    pragma_str,
                    gentype_base[i].c_str(),
                    get_default_data_for_type(gentype_base[i]).c_str(),
                    gentype_base[i].c_str(),
                    gentype_base[i].c_str(),
                    gentype_base[i].c_str(), vector_sizes[j].c_str(), vector_sizes[j].c_str(),
                    gentype_base[i].c_str(), vector_sizes[j].c_str(), vector_sizes[j].c_str()
                );

                KERNEL_FUNCTIONS.push_back(temp_kernel);
            }
        }
    }
}

void create_vstore_kernels(std::vector<std::string>& KERNEL_FUNCTIONS, cl_device_id deviceID) {
    const std::string KERNEL_FUNCTION_TEMPLATE_GLOBAL =
        NL
        NL "%s"
        NL "__global %s data_generic[] = { %s };"
        NL "__global %s data_reference[] = { %s };"
        NL
        NL "__kernel void testKernel(__global uint *results) {"
        NL "    uint tid = get_global_id(0);"
        NL
        NL "    %s%s input = (%s%s)(1);"
        NL "    %s * ptr = data_generic;"
        NL
        NL "    vstore%s(input, 2, ptr);"
        NL "    vstore%s(input, 2, data_reference);"
        NL
        NL "    bool result = true;"
        NL "    for (int i = 0; i < sizeof(data_generic) / sizeof(data_generic[0]); i++)"
        NL "        if (data_generic[i] != data_reference[i])"
        NL "            result = false;"
        NL
        NL "    results[tid] = result;"
        NL "}"
        NL;

    const std::string KERNEL_FUNCTION_TEMPLATE_LOCAL =
        NL
        NL "%s"
        NL "__constant %s to_copy_from[] = { %s };"
        NL
        NL "__kernel void testKernel(__global uint *results) {"
        NL "    uint tid = get_global_id(0);"
        NL
        NL "    __local %s data_generic[100];"
        NL "    for (int i = 0; i < sizeof(to_copy_from) / sizeof(to_copy_from[0]); i++)"
        NL "        data_generic[i] = to_copy_from[i];"
        NL
        NL "    __local %s data_reference[100];"
        NL "    for (int i = 0; i < sizeof(to_copy_from) / sizeof(to_copy_from[0]); i++)"
        NL "        data_reference[i] = to_copy_from[i];"
        NL
        NL "    %s%s input = (%s%s)(1);"
        NL "    %s * ptr = data_generic;"
        NL
        NL "    vstore%s(input, 2, ptr);"
        NL "    vstore%s(input, 2, data_reference);"
        NL
        NL "    work_group_barrier(CLK_LOCAL_MEM_FENCE);"
        NL
        NL "    bool result = true;"
        NL "    for (int i = 0; i < sizeof(data_generic) / sizeof(data_generic[0]); i++)"
        NL "        if (data_generic[i] != data_reference[i])"
        NL "            result = false;"
        NL
        NL "    results[tid] = result;"
        NL "}"
        NL;

    const std::string KERNEL_FUNCTION_TEMPLATE_PRIVATE =
        NL
        NL "%s"
        NL "__kernel void testKernel(__global uint *results) {"
        NL "    uint tid = get_global_id(0);"
        NL
        NL "    __private %s data_generic[] = { %s };"
        NL "    __private %s data_reference[] = { %s };"
        NL
        NL "    %s%s input = (%s%s)(1);"
        NL "    %s * ptr = data_generic;"
        NL
        NL "    vstore%s(input, 2, ptr);"
        NL "    vstore%s(input, 2, data_reference);"
        NL
        NL "    bool result = true;"
        NL "    for (int i = 0; i < sizeof(data_generic) / sizeof(data_generic[0]); i++)"
        NL "        if (data_generic[i] != data_reference[i])"
        NL "            result = false;"
        NL
        NL "    results[tid] = result;"
        NL "}"
        NL;

    const std::string vector_sizes[] = { "2", "3", "4", "8", "16" };
    const std::string gentype_base[] = { "double", "float", "char", "uchar", "short", "ushort", "int", "uint", "long", "ulong" };
    const std::string kernel_variants[] = { KERNEL_FUNCTION_TEMPLATE_GLOBAL, KERNEL_FUNCTION_TEMPLATE_LOCAL, KERNEL_FUNCTION_TEMPLATE_PRIVATE };

    const std::string cl_khr_fp64_pragma = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable";

    for (size_t i = 0; i < sizeof(gentype_base) / sizeof(gentype_base[0]); i++) {
        const char *pragma_str = "";
        if (i == 0) {
            if (!is_extension_available(deviceID, "cl_khr_fp64"))
                continue;
            else
                pragma_str = cl_khr_fp64_pragma.c_str();
        }


        for (size_t j = 0; j < sizeof(vector_sizes) / sizeof(vector_sizes[0]); j++) {
            for (size_t k = 0; k < sizeof(kernel_variants) / sizeof(kernel_variants[0]); k++) {
                char temp_kernel[4098];

                switch (k) {
                    case 0: // global template
                    case 2: // private template
                        sprintf(temp_kernel, kernel_variants[k].c_str(),
                            pragma_str,
                            gentype_base[i].c_str(), get_default_data_for_type(gentype_base[i]).c_str(),
                            gentype_base[i].c_str(), get_default_data_for_type(gentype_base[i]).c_str(),
                            gentype_base[i].c_str(), vector_sizes[j].c_str(), gentype_base[i].c_str(), vector_sizes[j].c_str(),
                            gentype_base[i].c_str(),
                            vector_sizes[j].c_str(),
                            vector_sizes[j].c_str()
                        );
                        break;

                    case 1: // local template
                        sprintf(temp_kernel, kernel_variants[k].c_str(),
                            pragma_str,
                            gentype_base[i].c_str(), get_default_data_for_type(gentype_base[i]).c_str(),
                            gentype_base[i].c_str(),
                            gentype_base[i].c_str(),
                            gentype_base[i].c_str(), vector_sizes[j].c_str(), gentype_base[i].c_str(), vector_sizes[j].c_str(),
                            gentype_base[i].c_str(),
                            vector_sizes[j].c_str(),
                            vector_sizes[j].c_str()
                        );
                        break;
                }

                KERNEL_FUNCTIONS.push_back(temp_kernel);
            }
        }
    }
}

int test_builtin_functions(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    std::vector<std::string> KERNEL_FUNCTIONS;

    create_math_kernels(KERNEL_FUNCTIONS);
    create_vload_kernels(KERNEL_FUNCTIONS, deviceID);
    create_vstore_kernels(KERNEL_FUNCTIONS, deviceID);

    CAdvancedTest test(KERNEL_FUNCTIONS);

    return test.Execute(deviceID, context, queue, num_elements);
}

int test_generic_advanced_casting(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    std::vector<std::string> KERNEL_FUNCTIONS;

    KERNEL_FUNCTIONS.push_back(
        NL
        NL "__global char arr[16] = { 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3 };"
        NL
        NL "__kernel void testKernel(__global uint *results) {"
        NL "    uint tid = get_global_id(0);"
        NL
        NL "    const int * volatile ptr = (const int *)arr;"
        NL
        NL "    results[tid] = (ptr[0] == 0x00000000) && (ptr[1] == 0x01010101) && (ptr[2] == 0x02020202) && (ptr[3] == 0x03030303);"
        NL "}"
        NL
    );

    KERNEL_FUNCTIONS.push_back(
        NL
        NL "__kernel void testKernel(__global uint *results) {"
        NL "    uint tid = get_global_id(0);"
        NL
        NL "    __local int i;"
        NL "    i = 0x11112222;"
        NL "    short *ptr = (short *)&i;"
        NL "    local int *lptr = (local int *)ptr;"
        NL
        NL "    results[tid] = (lptr == &i) && (*lptr == i);"
        NL "}"
        NL
    );

    KERNEL_FUNCTIONS.push_back(
        NL
        NL "__kernel void testKernel(__global uint *results) {"
        NL "    uint tid = get_global_id(0);"
        NL
        NL "    int i = 0x11112222;"
        NL
        NL "    void *ptr = &i;"
        NL "    int copy = *((int *)ptr);"
        NL
        NL "    results[tid] = (copy == i);"
        NL "}"
        NL
    );

    CAdvancedTest test(KERNEL_FUNCTIONS);

    return test.Execute(deviceID, context, queue, num_elements);
}

int test_generic_ptr_to_host_mem_svm(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    cl_int result = CL_SUCCESS;

    /* Test SVM capabilities and select matching tests */
    cl_device_svm_capabilities caps;
    auto version = get_device_cl_version(deviceID);
    auto expected_min_version = Version(2, 0);

    cl_int error = clGetDeviceInfo(deviceID, CL_DEVICE_SVM_CAPABILITIES, sizeof(caps), &caps, NULL);
    test_error(error, "clGetDeviceInfo(CL_DEVICE_SVM_CAPABILITIES) failed");

    if ((version < expected_min_version)
        || (version >= Version(3, 0) && caps == 0))
        return TEST_SKIPPED_ITSELF;

    if (caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER) {
        CAdvancedTest test_global_svm_ptr(common::GLOBAL_KERNEL_FUNCTION, ARG_TYPE_COARSE_GRAINED_SVM);
        result |= test_global_svm_ptr.Execute(deviceID, context, queue, num_elements);
    }

    if (caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) {
        CAdvancedTest test_global_svm_ptr(common::GLOBAL_KERNEL_FUNCTION, ARG_TYPE_FINE_GRAINED_BUFFER_SVM);
        result |= test_global_svm_ptr.Execute(deviceID, context, queue, num_elements);
    }

    if (caps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM) {
        CAdvancedTest test_global_svm_ptr(common::GLOBAL_KERNEL_FUNCTION, ARG_TYPE_FINE_GRAINED_SYSTEM_SVM);
        result |= test_global_svm_ptr.Execute(deviceID, context, queue, num_elements);
    }

    if (caps & CL_DEVICE_SVM_ATOMICS) {
        CAdvancedTest test_global_svm_ptr(common::GLOBAL_KERNEL_FUNCTION, ARG_TYPE_ATOMICS_SVM);
        result |= test_global_svm_ptr.Execute(deviceID, context, queue, num_elements);
    }

    return result;
}

int test_generic_ptr_to_host_mem(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    cl_int result = CL_SUCCESS;

    CAdvancedTest test_global_ptr(common::GLOBAL_KERNEL_FUNCTION, ARG_TYPE_HOST_PTR);
    result |= test_global_ptr.Execute(deviceID, context, queue, num_elements);

    CAdvancedTest test_local_ptr(common::LOCAL_KERNEL_FUNCTION, ARG_TYPE_HOST_LOCAL);
    result |= test_local_ptr.Execute(deviceID, context, queue, num_elements / 64);

    return result;
}
