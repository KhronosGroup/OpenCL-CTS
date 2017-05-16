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
#ifndef TEST_CONFORMANCE_CLCPP_API_TEST_CTORS_DTORS_HPP
#define TEST_CONFORMANCE_CLCPP_API_TEST_CTORS_DTORS_HPP

#include <vector>
#include <limits>
#include <algorithm>

#include "../common.hpp"

// Verify queries clGetProgramInfo correctly return the presence of constructors and/or destructors
// in the program (using option CL_PROGRAM_SCOPE_GLOBAL_CTORS_PRESENT/CL_PROGRAM_SCOPE_GLOBAL_DTORS_PRESENT)
// (both are present, either one is present, none is present).

std::string generate_ctor_dtor_program(const bool ctor, const bool dtor)
{
    std::string program;
    if(ctor)
    {
        program +=
            "struct ctor_test_class {\n"
            // non-trivial ctor
            "   ctor_test_class(int y) { x = y;};\n"
            "   int x;\n"
            "};\n"
            "ctor_test_class ctor = ctor_test_class(1024);\n"
        ;
    }
    if(dtor)
    {
        program +=
            "struct dtor_test_class {\n"
            // non-trivial dtor
            "   ~dtor_test_class() { x = -1024; };\n"
            "   int x;\n"
            "};\n"
            "dtor_test_class dtor;\n"
        ;
    }
    program += "__kernel void test_ctor_dtor()\n {\n }\n";
    return program;
}

int test_get_program_info_global_ctors_dtors_present(cl_device_id device,
                                                     cl_context context,
                                                     cl_command_queue queue,
                                                     const bool ctor,
                                                     const bool dtor)
{
    int error = CL_SUCCESS;
    cl_program program;

    // program source and options
    std::string options = "";
    std::string source = generate_ctor_dtor_program(ctor, dtor);
    const char * source_ptr = source.c_str();

// -----------------------------------------------------------------------------------
// ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
// -----------------------------------------------------------------------------------
// Only OpenCL C++ to SPIR-V compilation
#if defined(DEVELOPMENT) && defined(ONLY_SPIRV_COMPILATION)
    // Create program
    error = create_openclcpp_program(context, &program, 1, &source_ptr, options.c_str());
    RETURN_ON_ERROR(error)
    return error;
// Use OpenCL C kernels instead of OpenCL C++ kernels (test C++ host code)
#elif defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
    return CL_SUCCESS;
// Normal run
#else
    // Create program
    error = create_openclcpp_program(context, &program, 1, &source_ptr, options.c_str());
    RETURN_ON_ERROR(error)
#endif

    // CL_PROGRAM_SCOPE_GLOBAL_CTORS_PRESENT cl_bool
    // This indicates that the program object contains non-trivial constructor(s) that will be
    // executed by runtime before any kernel from the program is executed.

    // CL_PROGRAM_SCOPE_GLOBAL_DTORS_PRESENT cl_bool
    // This indicates that the program object contains non-trivial destructor(s) that will be
    // executed by runtime when program is destroyed.

    // CL_PROGRAM_SCOPE_GLOBAL_CTORS_PRESENT
    cl_bool ctors_present;
    size_t cl_bool_size;
    error = clGetProgramInfo(
        program,
        CL_PROGRAM_SCOPE_GLOBAL_CTORS_PRESENT,
        sizeof(cl_bool),
        static_cast<void*>(&ctors_present),
        &cl_bool_size
    );
    RETURN_ON_CL_ERROR(error, "clGetProgramInfo")
    if(cl_bool_size != sizeof(cl_bool))
    {
        error = -1;
        CHECK_ERROR_MSG(-1, "Test failed, param_value_size_ret != sizeof(cl_bool) (%lu != %lu).", cl_bool_size, sizeof(cl_bool));
    }
    if(ctor && ctors_present != CL_TRUE)
    {
        error = -1;
        CHECK_ERROR_MSG(-1, "Test failed, CL_PROGRAM_SCOPE_GLOBAL_CTORS_PRESENT: 0, should be: 1.");
    }
    else if(!ctor && ctors_present == CL_TRUE)
    {
        error = -1;
        CHECK_ERROR_MSG(-1, "Test failed, CL_PROGRAM_SCOPE_GLOBAL_CTORS_PRESENT: 1, should be: 0.");
    }

    // CL_PROGRAM_SCOPE_GLOBAL_DTORS_PRESENT
    cl_bool dtors_present = 0;
    error = clGetProgramInfo(
        program,
        CL_PROGRAM_SCOPE_GLOBAL_DTORS_PRESENT,
        sizeof(cl_bool),
        static_cast<void*>(&ctors_present),
        &cl_bool_size
    );
    RETURN_ON_CL_ERROR(error, "clGetProgramInfo")
    if(cl_bool_size != sizeof(cl_bool))
    {
        error = -1;
        CHECK_ERROR_MSG(-1, "Test failed, param_value_size_ret != sizeof(cl_bool) (%lu != %lu).", cl_bool_size, sizeof(cl_bool));
    }
    if(dtor && dtors_present != CL_TRUE)
    {
        error = -1;
        CHECK_ERROR_MSG(-1, "Test failed, CL_PROGRAM_SCOPE_GLOBAL_DTORS_PRESENT: 0, should be: 1.");
    }
    else if(!dtor && dtors_present == CL_TRUE)
    {
        error = -1;
        CHECK_ERROR_MSG(-1, "Test failed, CL_PROGRAM_SCOPE_GLOBAL_DTORS_PRESENT: 1, should be: 0.");
    }

    clReleaseProgram(program);
    return error;
}

AUTO_TEST_CASE(test_global_scope_ctors_dtors_present)
(cl_device_id device, cl_context context, cl_command_queue queue, int count)
{
    int error = CL_SUCCESS;
    int last_error = CL_SUCCESS;
    // both present
    last_error = test_get_program_info_global_ctors_dtors_present(device, context, queue, true, true);
    CHECK_ERROR(last_error);
    error |= last_error;
    // dtor
    last_error = test_get_program_info_global_ctors_dtors_present(device, context, queue, false, true);
    CHECK_ERROR(last_error);
    error |= last_error;
    // ctor
    last_error = test_get_program_info_global_ctors_dtors_present(device, context, queue, true, false);
    CHECK_ERROR(last_error);
    error |= last_error;
    // none present
    last_error = test_get_program_info_global_ctors_dtors_present(device, context, queue, false, false);
    CHECK_ERROR(last_error);
    error |= last_error;

    if(error != CL_SUCCESS)
    {
        return -1;
    }
    return error;
}

#endif // TEST_CONFORMANCE_CLCPP_API_TEST_CTORS_DTORS_HPP
