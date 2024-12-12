//
// Copyright (c) 2020-2023 The Khronos Group Inc.
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

#include "testBase.h"

const char *sample_kernel_code_single_line[] = {
    "__kernel void sample_test(__global float *src, __global int *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = (int)src[tid];\n"
    "\n"
    "}\n"
};

REGISTER_TEST(get_program_il)
{
    clProgramWrapper source_program;
    size_t il_size = -1;
    int error;

    /* If a program has been created with clCreateProgramWithIL, CL_PROGRAM_IL
     * should return the program IL it was created with and it's size */
    if (gCoreILProgram || is_extension_available(device, "cl_khr_il_program"))
    {
        clProgramWrapper il_program;
        std::string spvStr = "op_function_none";
        const char *spvName = spvStr.c_str();

        std::vector<unsigned char> spirv_binary = readSPIRV(spvName);

        size_t file_bytes = spirv_binary.size();
        if (file_bytes == 0)
        {
            test_fail("ERROR: SPIRV file %s not found!\n", spvName);
        }

        /* Create program with IL */
        unsigned char *spirv_buffer = &spirv_binary[0];

        error = get_program_with_il(il_program, device, context, spvName);

        SPIRV_CHECK_ERROR(error, "Unable to create program with IL.");
        if (il_program == NULL)
        {
            test_fail("ERROR: Unable to create test program!\n");
        }

        /* Check program IL is the same as the source IL */
        unsigned char *buffer = new unsigned char[file_bytes];
        error = clGetProgramInfo(il_program, CL_PROGRAM_IL, file_bytes, buffer,
                                 &il_size);
        SPIRV_CHECK_ERROR(error, "Unable to get program info.");

        if (il_size != file_bytes)
        {
            test_fail("ERROR: Returned IL size is not the same as source IL "
                      "size (%zu "
                      "!= %zu)!\n",
                      il_size, file_bytes);
        }

        if (memcmp(buffer, spirv_buffer, file_bytes) != 0)
        {
            test_fail("ERROR: Returned IL is not the same as source IL!\n");
        }

        delete[] buffer;
    }

    /* CL_PROGRAM_IL shouldn't return IL value unless program is created with
     * clCreateProgramWithIL */
    error = create_single_kernel_helper_create_program(
        context, &source_program, 1, sample_kernel_code_single_line);
    if (source_program == NULL)
    {
        test_fail("ERROR: Unable to create test program!\n");
    }

    if (gCompilationMode != kSpir_v)
    {
        error =
            clGetProgramInfo(source_program, CL_PROGRAM_IL, 0, NULL, &il_size);
        SPIRV_CHECK_ERROR(error, "Unable to get program il length");
        if (il_size != 0)
        {
            test_fail(
                "ERROR: Returned length of non-IL program IL is non-zero!\n");
        }
    }

    return 0;
}
