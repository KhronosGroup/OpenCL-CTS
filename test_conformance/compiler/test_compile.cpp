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
#include "testBase.h"
#if defined(_WIN32)
#include <time.h>
#elif defined(__linux__) || defined(__APPLE__)
#include <sys/time.h>
#include <unistd.h>
#endif
#include "harness/conversions.h"

#define MAX_LINE_SIZE_IN_PROGRAM 1024
#define MAX_LOG_SIZE_IN_PROGRAM 2048

const char *sample_kernel_start =
    "__kernel void sample_test(__global float *src, __global int *dst)\n"
    "{\n"
    "    float temp = 0.0f;\n"
    "    int  tid = get_global_id(0);\n";

const char *sample_kernel_end = "}\n";

const char *sample_kernel_lines[] = { "dst[tid] = src[tid];\n",
                                      "dst[tid] = src[tid] * 3.f;\n",
                                      "temp = src[tid] / 4.f;\n",
                                      "dst[tid] = dot(temp,src[tid]);\n",
                                      "dst[tid] = dst[tid] + temp;\n" };

/* I compile and link therefore I am. Robert Ioffe */
/* The following kernels are used in testing Improved Compilation and Linking
 * feature */

const char *simple_kernel = "__kernel void\n"
                            "CopyBuffer(\n"
                            "    __global float* src,\n"
                            "    __global float* dst )\n"
                            "{\n"
                            "    int id = (int)get_global_id(0);\n"
                            "    dst[id] = src[id];\n"
                            "}\n";

const char *simple_kernel_with_defines =
    "__kernel void\n"
    "CopyBuffer(\n"
    "    __global float* src,\n"
    "    __global float* dst )\n"
    "{\n"
    "    int id = (int)get_global_id(0);\n"
    "    float temp = src[id] - 42;\n"
    "    dst[id] = FIRST + temp + SECOND;\n"
    "}\n";

const char *simple_kernel_template = "__kernel void\n"
                                     "CopyBuffer%d(\n"
                                     "    __global float* src,\n"
                                     "    __global float* dst )\n"
                                     "{\n"
                                     "    int id = (int)get_global_id(0);\n"
                                     "    dst[id] = src[id];\n"
                                     "}\n";

const char *composite_kernel_start = "__kernel void\n"
                                     "CompositeKernel(\n"
                                     "    __global float* src,\n"
                                     "    __global float* dst )\n"
                                     "{\n";

const char *composite_kernel_end = "}\n";

const char *composite_kernel_template = "    CopyBuffer%d(src, dst);\n";

const char *composite_kernel_extern_template = "extern __kernel void\n"
                                               "CopyBuffer%d(\n"
                                               "    __global float* src,\n"
                                               "    __global float* dst );\n";

const char *another_simple_kernel = "extern __kernel void\n"
                                    "CopyBuffer(\n"
                                    "    __global float* src,\n"
                                    "    __global float* dst );\n"
                                    "__kernel void\n"
                                    "AnotherCopyBuffer(\n"
                                    "    __global float* src,\n"
                                    "    __global float* dst )\n"
                                    "{\n"
                                    "    CopyBuffer(src, dst);\n"
                                    "}\n";

const char *simple_header = "extern __kernel void\n"
                            "CopyBuffer(\n"
                            "    __global float* src,\n"
                            "    __global float* dst );\n";

const char *simple_header_name = "simple_header.h";

const char *another_simple_kernel_with_header = "#include \"simple_header.h\"\n"
                                                "__kernel void\n"
                                                "AnotherCopyBuffer(\n"
                                                "    __global float* src,\n"
                                                "    __global float* dst )\n"
                                                "{\n"
                                                "    CopyBuffer(src, dst);\n"
                                                "}\n";

const char *header_name_templates[4] = { "simple_header%d.h",
                                         "foo/simple_header%d.h",
                                         "foo/bar/simple_header%d.h",
                                         "foo/bar/baz/simple_header%d.h" };

const char *include_header_name_templates[4] = {
    "#include \"simple_header%d.h\"\n", "#include \"foo/simple_header%d.h\"\n",
    "#include \"foo/bar/simple_header%d.h\"\n",
    "#include \"foo/bar/baz/simple_header%d.h\"\n"
};

const char *compile_extern_var = "extern constant float foo;\n";
const char *compile_extern_struct = "extern constant struct bar bart;\n";
const char *compile_extern_function = "extern int baz(int, int);\n";

const char *compile_static_var = "static constant float foo = 2.78;\n";
const char *compile_static_struct = "static constant struct bar {float x, y, "
                                    "z, r; int color; } foo = {3.14159};\n";
const char *compile_static_function =
    "static int foo(int x, int y) { return x*x + y*y; }\n";

const char *compile_regular_var = "constant float foo = 4.0f;\n";
const char *compile_regular_struct =
    "constant struct bar {float x, y, z, r; int color; } foo = {0.f, 0.f, 0.f, "
    "0.f, 0};\n";
const char *compile_regular_function =
    "int foo(int x, int y) { return x*x + y*y; }\n";

const char *link_static_var_access = // use with compile_static_var
    "extern constant float foo;\n"
    "float access_foo() { return foo; }\n";

const char *link_static_struct_access = // use with compile_static_struct
    "extern constant struct bar{float x, y, z, r; int color; } foo;\n"
    "struct bar access_foo() {return foo; }\n";

const char *link_static_function_access = // use with compile_static_function
    "extern int foo(int, int);\n"
    "int access_foo() { int blah = foo(3, 4); return blah + 5; }\n";

int test_large_single_compile(cl_context context, cl_device_id deviceID,
                              unsigned int numLines)
{
    int error;
    cl_program program;
    const char **lines;
    unsigned int numChoices, i;
    MTdata d;

    /* First, allocate the array for our line pointers */
    lines = (const char **)malloc(numLines * sizeof(const char *));
    if (lines == NULL)
    {
        log_error(
            "ERROR: Unable to allocate lines array with %d lines! (in %s:%d)\n",
            numLines, __FILE__, __LINE__);
        return -1;
    }

    /* First and last lines are easy */
    lines[0] = sample_kernel_start;
    lines[numLines - 1] = sample_kernel_end;

    numChoices = sizeof(sample_kernel_lines) / sizeof(sample_kernel_lines[0]);

    /* Fill the rest with random lines to hopefully prevent much optimization */
    d = init_genrand(gRandomSeed);
    for (i = 1; i < numLines - 1; i++)
    {
        lines[i] = sample_kernel_lines[genrand_int32(d) % numChoices];
    }
    free_mtdata(d);
    d = NULL;

    /* Try to create a program with these lines */
    error = create_single_kernel_helper_create_program(context, &program,
                                                       numLines, lines);
    if (program == NULL || error != CL_SUCCESS)
    {
        log_error("ERROR: Unable to create long test program with %d lines! "
                  "(%s in %s:%d)",
                  numLines, IGetErrorString(error), __FILE__, __LINE__);
        free(lines);
        if (program != NULL)
        {
            error = clReleaseProgram(program);
            test_error(error, "Unable to release a program object");
        }
        return -1;
    }

    /* Build it */
    error = clBuildProgram(program, 1, &deviceID, NULL, NULL, NULL);
    test_error(error, "Unable to build a long program");

    /* All done! */
    error = clReleaseProgram(program);
    test_error(error, "Unable to release a program object");

    free(lines);

    return 0;
}

int test_large_compile(cl_device_id deviceID, cl_context context,
                       cl_command_queue queue, int num_elements)
{
    unsigned int toTest[] = {
        64, 128, 256, 512, 1024, 2048, 4096, 0
    }; // 8192, 16384, 32768, 0 };
    unsigned int i;

    log_info("Testing large compiles...this might take awhile...\n");

    for (i = 0; toTest[i] != 0; i++)
    {
        log_info("   %d...\n", toTest[i]);

#if defined(_WIN32)
        clock_t start = clock();
#elif defined(__linux__) || defined(__APPLE__)
        timeval time1, time2;
        gettimeofday(&time1, NULL);
#endif

        if (test_large_single_compile(context, deviceID, toTest[i]) != 0)
        {
            log_error(
                "ERROR: long program test failed for %d lines! (in %s:%d)\n",
                toTest[i], __FILE__, __LINE__);
            return -1;
        }

#if defined(_WIN32)
        clock_t end = clock();
        log_perf((float)(end - start) / (float)CLOCKS_PER_SEC, false,
                 "clock() time in secs", "%d lines", toTest[i]);
#elif defined(__linux__) || defined(__APPLE__)
        gettimeofday(&time2, NULL);
        log_perf((float)(float)(time2.tv_sec - time1.tv_sec)
                     + 1.0e-6 * (time2.tv_usec - time1.tv_usec),
                 false, "wall time in secs", "%d lines", toTest[i]);
#endif
    }

    return 0;
}

static int verifyCopyBuffer(cl_context context, cl_command_queue queue,
                            cl_kernel kernel);

#if defined(__APPLE__) || defined(__linux)
#define _strdup strdup
#endif

int test_large_multi_file_library(cl_context context, cl_device_id deviceID,
                                  cl_command_queue queue, unsigned int numLines)
{
    int error;
    cl_program program;
    cl_program *simple_kernels;
    const char **lines;
    unsigned int i;
    char buffer[MAX_LINE_SIZE_IN_PROGRAM];

    simple_kernels = (cl_program *)malloc(numLines * sizeof(cl_program));
    if (simple_kernels == NULL)
    {
        log_error("ERROR: Unable to allocate kernels array with %d kernels! "
                  "(in %s:%d)\n",
                  numLines, __FILE__, __LINE__);
        return -1;
    }
    /* First, allocate the array for our line pointers */
    lines = (const char **)malloc((2 * numLines + 2) * sizeof(const char *));
    if (lines == NULL)
    {
        free(simple_kernels);
        log_error(
            "ERROR: Unable to allocate lines array with %d lines! (in %s:%d)\n",
            (2 * numLines + 2), __FILE__, __LINE__);
        return -1;
    }

    for (i = 0; i < numLines; i++)
    {
        sprintf(buffer, composite_kernel_extern_template, i);
        lines[i] = _strdup(buffer);
    }
    /* First and last lines are easy */
    lines[numLines] = composite_kernel_start;
    lines[2 * numLines + 1] = composite_kernel_end;

    /* Fill the rest with templated kernels */
    for (i = numLines + 1; i < 2 * numLines + 1; i++)
    {
        sprintf(buffer, composite_kernel_template, i - numLines - 1);
        lines[i] = _strdup(buffer);
    }

    /* Try to create a program with these lines */
    error = create_single_kernel_helper_create_program(context, &program,
                                                       2 * numLines + 2, lines);
    if (program == NULL || error != CL_SUCCESS)
    {
        log_error("ERROR: Unable to create long test program with %d lines! "
                  "(%s) (in %s:%d)\n",
                  numLines, IGetErrorString(error), __FILE__, __LINE__);
        free(simple_kernels);
        for (i = 0; i < numLines; i++)
        {
            free((void *)lines[i]);
            free((void *)lines[i + numLines + 1]);
        }
        free(lines);
        if (program != NULL)
        {
            error = clReleaseProgram(program);
            test_error(error, "Unable to release program object");
        }

        return -1;
    }

    /* Compile it */
    error = clCompileProgram(program, 1, &deviceID, NULL, 0, NULL, NULL, NULL,
                             NULL);
    test_error(error, "Unable to compile a simple program");

    /* Create and compile templated kernels */
    for (i = 0; i < numLines; i++)
    {
        sprintf(buffer, simple_kernel_template, i);
        const char *kernel_source = _strdup(buffer);
        simple_kernels[i] =
            clCreateProgramWithSource(context, 1, &kernel_source, NULL, &error);
        if (simple_kernels[i] == NULL || error != CL_SUCCESS)
        {
            log_error("ERROR: Unable to create long test program with %d "
                      "lines! (%s) (in %s:%d)\n",
                      numLines, IGetErrorString(error), __FILE__, __LINE__);
            return -1;
        }

        /* Compile it */
        error = clCompileProgram(simple_kernels[i], 1, &deviceID, NULL, 0, NULL,
                                 NULL, NULL, NULL);
        test_error(error, "Unable to compile a simple program");

        free((void *)kernel_source);
    }

    /* Create library out of compiled templated kernels */
    cl_program my_newly_minted_library =
        clLinkProgram(context, 1, &deviceID, "-create-library", numLines,
                      simple_kernels, NULL, NULL, &error);
    test_error(error, "Unable to create a multi-line library");

    /* Link the program that calls the kernels and the library that contains
     * them */
    cl_program programs[2] = { program, my_newly_minted_library };
    cl_program my_newly_linked_program = clLinkProgram(
        context, 1, &deviceID, NULL, 2, programs, NULL, NULL, &error);
    test_error(error, "Unable to link a program with a library");

    // Create the composite kernel
    cl_kernel kernel =
        clCreateKernel(my_newly_linked_program, "CompositeKernel", &error);
    test_error(error, "Unable to create a composite kernel");

    // Run the composite kernel and verify the results
    error = verifyCopyBuffer(context, queue, kernel);
    if (error != CL_SUCCESS) return error;

    /* All done! */
    error = clReleaseProgram(program);
    test_error(error, "Unable to release program object");

    for (i = 0; i < numLines; i++)
    {
        free((void *)lines[i]);
        free((void *)lines[i + numLines + 1]);
    }
    free(lines);

    for (i = 0; i < numLines; i++)
    {
        error = clReleaseProgram(simple_kernels[i]);
        test_error(error, "Unable to release program object");
    }
    free(simple_kernels);

    error = clReleaseKernel(kernel);
    test_error(error, "Unable to release kernel object");

    error = clReleaseProgram(my_newly_minted_library);
    test_error(error, "Unable to release program object");

    error = clReleaseProgram(my_newly_linked_program);
    test_error(error, "Unable to release program object");

    return 0;
}

int test_multi_file_libraries(cl_device_id deviceID, cl_context context,
                              cl_command_queue queue, int num_elements)
{
    unsigned int toTest[] = {
        2, 4, 8, 16, 32, 64, 128, 256, 0
    }; // 512, 1024, 2048, 4096, 8192, 16384, 32768, 0 };
    unsigned int i;

    log_info("Testing multi-file libraries ...this might take awhile...\n");

    for (i = 0; toTest[i] != 0; i++)
    {
        log_info("   %d...\n", toTest[i]);

#if defined(_WIN32)
        clock_t start = clock();
#elif defined(__linux__) || defined(__APPLE__)
        timeval time1, time2;
        gettimeofday(&time1, NULL);
#endif

        if (test_large_multi_file_library(context, deviceID, queue, toTest[i])
            != 0)
        {
            log_error("ERROR: multi-file library program test failed for %d "
                      "lines! (in %s:%d)\n\n",
                      toTest[i], __FILE__, __LINE__);
            return -1;
        }

#if defined(_WIN32)
        clock_t end = clock();
        log_perf((float)(end - start) / (float)CLOCKS_PER_SEC, false,
                 "clock() time in secs", "%d lines", toTest[i]);
#elif defined(__linux__) || defined(__APPLE__)
        gettimeofday(&time2, NULL);
        log_perf((float)(float)(time2.tv_sec - time1.tv_sec)
                     + 1.0e-6 * (time2.tv_usec - time1.tv_usec),
                 false, "wall time in secs", "%d lines", toTest[i]);
#endif
    }

    return 0;
}

int test_large_multiple_embedded_headers(cl_context context,
                                         cl_device_id deviceID,
                                         cl_command_queue queue,
                                         unsigned int numLines)
{
    int error;
    cl_program program;
    cl_program *simple_kernels;
    cl_program *headers;
    const char **header_names;
    const char **lines;
    unsigned int i;
    char buffer[MAX_LINE_SIZE_IN_PROGRAM];

    simple_kernels = (cl_program *)malloc(numLines * sizeof(cl_program));
    if (simple_kernels == NULL)
    {
        log_error("ERROR: Unable to allocate simple_kernels array with %d "
                  "lines! (in %s:%d)\n",
                  numLines, __FILE__, __LINE__);
        return -1;
    }
    headers = (cl_program *)malloc(numLines * sizeof(cl_program));
    if (headers == NULL)
    {
        log_error("ERROR: Unable to allocate headers array with %d lines! (in "
                  "%s:%d)\n",
                  numLines, __FILE__, __LINE__);
        return -1;
    }
    /* First, allocate the array for our line pointers */
    header_names = (const char **)malloc(numLines * sizeof(const char *));
    if (header_names == NULL)
    {
        log_error("ERROR: Unable to allocate header_names array with %d lines! "
                  "(in %s:%d)\n",
                  numLines, __FILE__, __LINE__);
        return -1;
    }
    lines = (const char **)malloc((2 * numLines + 2) * sizeof(const char *));
    if (lines == NULL)
    {
        log_error(
            "ERROR: Unable to allocate lines array with %d lines! (in %s:%d)\n",
            (2 * numLines + 2), __FILE__, __LINE__);
        return -1;
    }

    for (i = 0; i < numLines; i++)
    {
        sprintf(buffer, include_header_name_templates[i % 4], i);
        lines[i] = _strdup(buffer);
        sprintf(buffer, header_name_templates[i % 4], i);
        header_names[i] = _strdup(buffer);

        sprintf(buffer, composite_kernel_extern_template, i);
        const char *line = buffer;
        error = create_single_kernel_helper_create_program(context, &headers[i],
                                                           1, &line);
        if (headers[i] == NULL || error != CL_SUCCESS)
        {
            log_error("ERROR: Unable to create a simple header program! (%s in "
                      "%s:%d)\n",
                      IGetErrorString(error), __FILE__, __LINE__);
            return -1;
        }
    }
    /* First and last lines are easy */
    lines[numLines] = composite_kernel_start;
    lines[2 * numLines + 1] = composite_kernel_end;

    /* Fill the rest with templated kernels */
    for (i = numLines + 1; i < 2 * numLines + 1; i++)
    {
        sprintf(buffer, composite_kernel_template, i - numLines - 1);
        lines[i] = _strdup(buffer);
    }

    /* Try to create a program with these lines */
    error = create_single_kernel_helper_create_program(context, &program,
                                                       2 * numLines + 2, lines);
    if (program == NULL || error != CL_SUCCESS)
    {
        log_error("ERROR: Unable to create long test program with %d lines! "
                  "(%s) (in %s:%d)\n",
                  numLines, IGetErrorString(error), __FILE__, __LINE__);
        return -1;
    }

    /* Compile it */
    error = clCompileProgram(program, 1, &deviceID, NULL, numLines, headers,
                             header_names, NULL, NULL);
    test_error(error, "Unable to compile a simple program");

    /* Create and compile templated kernels */
    for (i = 0; i < numLines; i++)
    {
        sprintf(buffer, simple_kernel_template, i);
        const char *kernel_source = _strdup(buffer);
        error = create_single_kernel_helper_create_program(
            context, &simple_kernels[i], 1, &kernel_source);
        if (simple_kernels[i] == NULL || error != CL_SUCCESS)
        {
            log_error("ERROR: Unable to create long test program with %d "
                      "lines! (%s) (in %s:%d)\n",
                      numLines, IGetErrorString(error), __FILE__, __LINE__);
            return -1;
        }

        /* Compile it */
        error = clCompileProgram(simple_kernels[i], 1, &deviceID, NULL, 0, NULL,
                                 NULL, NULL, NULL);
        test_error(error, "Unable to compile a simple program");

        free((void *)kernel_source);
    }

    /* Create library out of compiled templated kernels */
    cl_program my_newly_minted_library =
        clLinkProgram(context, 1, &deviceID, "-create-library", numLines,
                      simple_kernels, NULL, NULL, &error);
    test_error(error, "Unable to create a multi-line library");

    /* Link the program that calls the kernels and the library that contains
     * them */
    cl_program programs[2] = { program, my_newly_minted_library };
    cl_program my_newly_linked_program = clLinkProgram(
        context, 1, &deviceID, NULL, 2, programs, NULL, NULL, &error);
    test_error(error, "Unable to link a program with a library");

    // Create the composite kernel
    cl_kernel kernel =
        clCreateKernel(my_newly_linked_program, "CompositeKernel", &error);
    test_error(error, "Unable to create a composite kernel");

    // Run the composite kernel and verify the results
    error = verifyCopyBuffer(context, queue, kernel);
    if (error != CL_SUCCESS) return error;

    /* All done! */
    error = clReleaseProgram(program);
    test_error(error, "Unable to release program object");

    for (i = 0; i < numLines; i++)
    {
        free((void *)lines[i]);
        free((void *)header_names[i]);
    }
    for (i = numLines + 1; i < 2 * numLines + 1; i++)
    {
        free((void *)lines[i]);
    }
    free(lines);
    free(header_names);

    for (i = 0; i < numLines; i++)
    {
        error = clReleaseProgram(simple_kernels[i]);
        test_error(error, "Unable to release program object");
        error = clReleaseProgram(headers[i]);
        test_error(error, "Unable to release header program object");
    }
    free(simple_kernels);
    free(headers);

    error = clReleaseKernel(kernel);
    test_error(error, "Unable to release kernel object");

    error = clReleaseProgram(my_newly_minted_library);
    test_error(error, "Unable to release program object");

    error = clReleaseProgram(my_newly_linked_program);
    test_error(error, "Unable to release program object");

    return 0;
}

int test_multiple_embedded_headers(cl_device_id deviceID, cl_context context,
                                   cl_command_queue queue, int num_elements)
{
    unsigned int toTest[] = {
        2, 4, 8, 16, 32, 64, 128, 256, 0
    }; // 512, 1024, 2048, 4096, 8192, 16384, 32768, 0 };
    unsigned int i;

    log_info(
        "Testing multiple embedded headers ...this might take awhile...\n");

    for (i = 0; toTest[i] != 0; i++)
    {
        log_info("   %d...\n", toTest[i]);

#if defined(_WIN32)
        clock_t start = clock();
#elif defined(__linux__) || defined(__APPLE__)
        timeval time1, time2;
        gettimeofday(&time1, NULL);
#endif

        if (test_large_multiple_embedded_headers(context, deviceID, queue,
                                                 toTest[i])
            != 0)
        {
            log_error("ERROR: multiple embedded headers program test failed "
                      "for %d lines! (in %s:%d)\n",
                      toTest[i], __FILE__, __LINE__);
            return -1;
        }

#if defined(_WIN32)
        clock_t end = clock();
        log_perf((float)(end - start) / (float)CLOCKS_PER_SEC, false,
                 "clock() time in secs", "%d lines", toTest[i]);
#elif defined(__linux__) || defined(__APPLE__)
        gettimeofday(&time2, NULL);
        log_perf((float)(float)(time2.tv_sec - time1.tv_sec)
                     + 1.0e-6 * (time2.tv_usec - time1.tv_usec),
                 false, "wall time in secs", "%d lines", toTest[i]);
#endif
    }

    return 0;
}

double logbase(double a, double base) { return log(a) / log(base); }

int test_large_multiple_libraries(cl_context context, cl_device_id deviceID,
                                  cl_command_queue queue, unsigned int numLines)
{
    int error;
    cl_program *simple_kernels;
    const char **lines;
    unsigned int i;
    char buffer[MAX_LINE_SIZE_IN_PROGRAM];
    /* I want to create (log2(N)+1)/2 libraries */
    unsigned int level = (unsigned int)(logbase(numLines, 2.0) + 1.000001) / 2;
    unsigned int numLibraries = (unsigned int)pow(2.0, level - 1.0);
    unsigned int numFilesInLib = numLines / numLibraries;
    cl_program *my_program_and_libraries =
        (cl_program *)malloc((1 + numLibraries) * sizeof(cl_program));
    if (my_program_and_libraries == NULL)
    {
        log_error("ERROR: Unable to allocate program array with %d programs! "
                  "(in %s:%d)\n",
                  (1 + numLibraries), __FILE__, __LINE__);
        return -1;
    }

    log_info("level - %d, numLibraries - %d, numFilesInLib - %d\n", level,
             numLibraries, numFilesInLib);

    simple_kernels = (cl_program *)malloc(numLines * sizeof(cl_program));
    if (simple_kernels == NULL)
    {
        log_error("ERROR: Unable to allocate kernels array with %d kernels! "
                  "(in %s:%d)\n",
                  numLines, __FILE__, __LINE__);
        return -1;
    }
    /* First, allocate the array for our line pointers */
    lines = (const char **)malloc((2 * numLines + 2) * sizeof(const char *));
    if (lines == NULL)
    {
        log_error(
            "ERROR: Unable to allocate lines array with %d lines! (in %s:%d)\n",
            (2 * numLines + 2), __FILE__, __LINE__);
        return -1;
    }

    for (i = 0; i < numLines; i++)
    {
        sprintf(buffer, composite_kernel_extern_template, i);
        lines[i] = _strdup(buffer);
    }
    /* First and last lines are easy */
    lines[numLines] = composite_kernel_start;
    lines[2 * numLines + 1] = composite_kernel_end;

    /* Fill the rest with templated kernels */
    for (i = numLines + 1; i < 2 * numLines + 1; i++)
    {
        sprintf(buffer, composite_kernel_template, i - numLines - 1);
        lines[i] = _strdup(buffer);
    }

    /* Try to create a program with these lines */
    error = create_single_kernel_helper_create_program(
        context, &my_program_and_libraries[0], 2 * numLines + 2, lines);
    if (my_program_and_libraries[0] == NULL || error != CL_SUCCESS)
    {
        log_error("ERROR: Unable to create long test program with %d lines! "
                  "(%s in %s:%d)\n",
                  numLines, IGetErrorString(error), __FILE__, __LINE__);
        return -1;
    }

    /* Compile it */
    error = clCompileProgram(my_program_and_libraries[0], 1, &deviceID, NULL, 0,
                             NULL, NULL, NULL, NULL);
    test_error(error, "Unable to compile a simple program");

    /* Create and compile templated kernels */
    for (i = 0; i < numLines; i++)
    {
        sprintf(buffer, simple_kernel_template, i);
        const char *kernel_source = _strdup(buffer);
        error = create_single_kernel_helper_create_program(
            context, &simple_kernels[i], 1, &kernel_source);
        if (simple_kernels[i] == NULL || error != CL_SUCCESS)
        {
            log_error("ERROR: Unable to create long test program with %d "
                      "lines! (%s in %s:%d)\n",
                      numLines, IGetErrorString(error), __FILE__, __LINE__);
            return -1;
        }

        /* Compile it */
        error = clCompileProgram(simple_kernels[i], 1, &deviceID, NULL, 0, NULL,
                                 NULL, NULL, NULL);
        test_error(error, "Unable to compile a simple program");

        free((void *)kernel_source);
    }

    /* Create library out of compiled templated kernels */
    for (i = 0; i < numLibraries; i++)
    {
        my_program_and_libraries[i + 1] = clLinkProgram(
            context, 1, &deviceID, "-create-library", numFilesInLib,
            simple_kernels + i * numFilesInLib, NULL, NULL, &error);
        test_error(error, "Unable to create a multi-line library");
    }

    /* Link the program that calls the kernels and the library that contains
     * them */
    cl_program my_newly_linked_program =
        clLinkProgram(context, 1, &deviceID, NULL, numLibraries + 1,
                      my_program_and_libraries, NULL, NULL, &error);
    test_error(error, "Unable to link a program with a library");

    // Create the composite kernel
    cl_kernel kernel =
        clCreateKernel(my_newly_linked_program, "CompositeKernel", &error);
    test_error(error, "Unable to create a composite kernel");

    // Run the composite kernel and verify the results
    error = verifyCopyBuffer(context, queue, kernel);
    if (error != CL_SUCCESS) return error;

    /* All done! */
    for (i = 0; i <= numLibraries; i++)
    {
        error = clReleaseProgram(my_program_and_libraries[i]);
        test_error(error, "Unable to release program object");
    }
    free(my_program_and_libraries);
    for (i = 0; i < numLines; i++)
    {
        free((void *)lines[i]);
    }
    for (i = numLines + 1; i < 2 * numLines + 1; i++)
    {
        free((void *)lines[i]);
    }
    free(lines);

    for (i = 0; i < numLines; i++)
    {
        error = clReleaseProgram(simple_kernels[i]);
        test_error(error, "Unable to release program object");
    }
    free(simple_kernels);

    error = clReleaseKernel(kernel);
    test_error(error, "Unable to release kernel object");

    error = clReleaseProgram(my_newly_linked_program);
    test_error(error, "Unable to release program object");

    return 0;
}

int test_multiple_libraries(cl_device_id deviceID, cl_context context,
                            cl_command_queue queue, int num_elements)
{
    unsigned int toTest[] = {
        2, 8, 32, 128, 256, 0
    }; // 512, 2048, 8192, 32768, 0 };
    unsigned int i;

    log_info("Testing multiple libraries ...this might take awhile...\n");

    for (i = 0; toTest[i] != 0; i++)
    {
        log_info("   %d...\n", toTest[i]);

#if defined(_WIN32)
        clock_t start = clock();
#elif defined(__linux__) || defined(__APPLE__)
        timeval time1, time2;
        gettimeofday(&time1, NULL);
#endif

        if (test_large_multiple_libraries(context, deviceID, queue, toTest[i])
            != 0)
        {
            log_error("ERROR: multiple library program test failed for %d "
                      "lines! (in %s:%d)\n\n",
                      toTest[i], __FILE__, __LINE__);
            return -1;
        }

#if defined(_WIN32)
        clock_t end = clock();
        log_perf((float)(end - start) / (float)CLOCKS_PER_SEC, false,
                 "clock() time in secs", "%d lines", toTest[i]);
#elif defined(__linux__) || defined(__APPLE__)
        gettimeofday(&time2, NULL);
        log_perf((float)(float)(time2.tv_sec - time1.tv_sec)
                     + 1.0e-6 * (time2.tv_usec - time1.tv_usec),
                 false, "wall time in secs", "%d lines", toTest[i]);
#endif
    }

    return 0;
}

int test_large_multiple_files_multiple_libraries(cl_context context,
                                                 cl_device_id deviceID,
                                                 cl_command_queue queue,
                                                 unsigned int numLines)
{
    int error;
    cl_program *simple_kernels;
    const char **lines;
    unsigned int i;
    char buffer[MAX_LINE_SIZE_IN_PROGRAM];
    /* I want to create (log2(N)+1)/4 libraries */
    unsigned int level = (unsigned int)(logbase(numLines, 2.0) + 1.000001) / 2;
    unsigned int numLibraries = (unsigned int)pow(2.0, level - 2.0);
    unsigned int numFilesInLib = numLines / (2 * numLibraries);
    cl_program *my_programs_and_libraries = (cl_program *)malloc(
        (1 + numLibraries + numLibraries * numFilesInLib) * sizeof(cl_program));
    if (my_programs_and_libraries == NULL)
    {
        log_error("ERROR: Unable to allocate program array with %d programs! "
                  "(in %s:%d)\n",
                  (1 + numLibraries + numLibraries * numFilesInLib), __FILE__,
                  __LINE__);
        return -1;
    }
    log_info("level - %d, numLibraries - %d, numFilesInLib - %d\n", level,
             numLibraries, numFilesInLib);

    simple_kernels = (cl_program *)malloc(numLines * sizeof(cl_program));
    if (simple_kernels == NULL)
    {
        log_error("ERROR: Unable to allocate kernels array with %d kernels! "
                  "(in %s:%d)\n",
                  numLines, __FILE__, __LINE__);
        return -1;
    }
    /* First, allocate the array for our line pointers */
    lines = (const char **)malloc((2 * numLines + 2) * sizeof(const char *));
    if (lines == NULL)
    {
        log_error(
            "ERROR: Unable to allocate lines array with %d lines! (in %s:%d)\n",
            (2 * numLines + 2), __FILE__, __LINE__);
        return -1;
    }

    for (i = 0; i < numLines; i++)
    {
        sprintf(buffer, composite_kernel_extern_template, i);
        lines[i] = _strdup(buffer);
    }
    /* First and last lines are easy */
    lines[numLines] = composite_kernel_start;
    lines[2 * numLines + 1] = composite_kernel_end;

    /* Fill the rest with templated kernels */
    for (i = numLines + 1; i < 2 * numLines + 1; i++)
    {
        sprintf(buffer, composite_kernel_template, i - numLines - 1);
        lines[i] = _strdup(buffer);
    }

    /* Try to create a program with these lines */
    error = create_single_kernel_helper_create_program(
        context, &my_programs_and_libraries[0], 2 * numLines + 2, lines);
    if (my_programs_and_libraries[0] == NULL || error != CL_SUCCESS)
    {
        log_error("ERROR: Unable to create long test program with %d lines! "
                  "(%s in %s:%d)\n",
                  numLines, IGetErrorString(error), __FILE__, __LINE__);
        return -1;
    }

    /* Compile it */
    error = clCompileProgram(my_programs_and_libraries[0], 1, &deviceID, NULL,
                             0, NULL, NULL, NULL, NULL);
    test_error(error, "Unable to compile a simple program");

    /* Create and compile templated kernels */
    for (i = 0; i < numLines; i++)
    {
        sprintf(buffer, simple_kernel_template, i);
        const char *kernel_source = _strdup(buffer);
        error = create_single_kernel_helper_create_program(
            context, &simple_kernels[i], 1, &kernel_source);
        if (simple_kernels[i] == NULL || error != CL_SUCCESS)
        {
            log_error("ERROR: Unable to create long test program with %d "
                      "lines! (%s in %s:%d)\n",
                      numLines, IGetErrorString(error), __FILE__, __LINE__);
            return -1;
        }

        /* Compile it */
        error = clCompileProgram(simple_kernels[i], 1, &deviceID, NULL, 0, NULL,
                                 NULL, NULL, NULL);
        test_error(error, "Unable to compile a simple program");

        free((void *)kernel_source);
    }

    /* Copy already compiled kernels */
    for (i = 0; i < numLibraries * numFilesInLib; i++)
    {
        my_programs_and_libraries[i + 1] = simple_kernels[i];
    }

    /* Create library out of compiled templated kernels */
    for (i = 0; i < numLibraries; i++)
    {
        my_programs_and_libraries[i + 1 + numLibraries * numFilesInLib] =
            clLinkProgram(
                context, 1, &deviceID, "-create-library", numFilesInLib,
                simple_kernels
                    + (i * numFilesInLib + numLibraries * numFilesInLib),
                NULL, NULL, &error);
        test_error(error, "Unable to create a multi-line library");
    }

    /* Link the program that calls the kernels and the library that contains
     * them */
    cl_program my_newly_linked_program =
        clLinkProgram(context, 1, &deviceID, NULL,
                      numLibraries + 1 + numLibraries * numFilesInLib,
                      my_programs_and_libraries, NULL, NULL, &error);
    test_error(error, "Unable to link a program with a library");

    // Create the composite kernel
    cl_kernel kernel =
        clCreateKernel(my_newly_linked_program, "CompositeKernel", &error);
    test_error(error, "Unable to create a composite kernel");

    // Run the composite kernel and verify the results
    error = verifyCopyBuffer(context, queue, kernel);
    if (error != CL_SUCCESS) return error;

    /* All done! */
    for (i = 0; i < numLibraries + 1 + numLibraries * numFilesInLib; i++)
    {
        error = clReleaseProgram(my_programs_and_libraries[i]);
        test_error(error, "Unable to release program object");
    }
    free(my_programs_and_libraries);

    for (i = 0; i < numLines; i++)
    {
        free((void *)lines[i]);
    }
    for (i = numLines + 1; i < 2 * numLines + 1; i++)
    {
        free((void *)lines[i]);
    }
    free(lines);

    for (i = numLibraries * numFilesInLib; i < numLines; i++)
    {
        error = clReleaseProgram(simple_kernels[i]);
        test_error(error, "Unable to release program object");
    }
    free(simple_kernels);

    error = clReleaseKernel(kernel);
    test_error(error, "Unable to release kernel object");

    error = clReleaseProgram(my_newly_linked_program);
    test_error(error, "Unable to release program object");

    return 0;
}

int test_multiple_files_multiple_libraries(cl_device_id deviceID,
                                           cl_context context,
                                           cl_command_queue queue,
                                           int num_elements)
{
    unsigned int toTest[] = { 8, 32, 128, 256,
                              0 }; // 512, 2048, 8192, 32768, 0 };
    unsigned int i;

    log_info("Testing multiple files and multiple libraries ...this might take "
             "awhile...\n");

    for (i = 0; toTest[i] != 0; i++)
    {
        log_info("   %d...\n", toTest[i]);

#if defined(_WIN32)
        clock_t start = clock();
#elif defined(__linux__) || defined(__APPLE__)
        timeval time1, time2;
        gettimeofday(&time1, NULL);
#endif

        if (test_large_multiple_files_multiple_libraries(context, deviceID,
                                                         queue, toTest[i])
            != 0)
        {
            log_error("ERROR: multiple files, multiple libraries program test "
                      "failed for %d lines! (in %s:%d)\n\n",
                      toTest[i], __FILE__, __LINE__);
            return -1;
        }

#if defined(_WIN32)
        clock_t end = clock();
        log_perf((float)(end - start) / (float)CLOCKS_PER_SEC, false,
                 "clock() time in secs", "%d lines", toTest[i]);
#elif defined(__linux__) || defined(__APPLE__)
        gettimeofday(&time2, NULL);
        log_perf((float)(float)(time2.tv_sec - time1.tv_sec)
                     + 1.0e-6 * (time2.tv_usec - time1.tv_usec),
                 false, "wall time in secs", "%d lines", toTest[i]);
#endif
    }

    return 0;
}

int test_large_multiple_files(cl_context context, cl_device_id deviceID,
                              cl_command_queue queue, unsigned int numLines)
{
    int error;
    const char **lines;
    unsigned int i;
    char buffer[MAX_LINE_SIZE_IN_PROGRAM];
    cl_program *my_programs =
        (cl_program *)malloc((1 + numLines) * sizeof(cl_program));

    if (my_programs == NULL)
    {
        log_error("ERROR: Unable to allocate my_programs array with %d "
                  "programs! (in %s:%d)\n",
                  (1 + numLines), __FILE__, __LINE__);
        return -1;
    }
    /* First, allocate the array for our line pointers */
    lines = (const char **)malloc((2 * numLines + 2) * sizeof(const char *));
    if (lines == NULL)
    {
        log_error(
            "ERROR: Unable to allocate lines array with %d lines! (in %s:%d)\n",
            (2 * numLines + 2), __FILE__, __LINE__);
        return -1;
    }

    for (i = 0; i < numLines; i++)
    {
        sprintf(buffer, composite_kernel_extern_template, i);
        lines[i] = _strdup(buffer);
    }
    /* First and last lines are easy */
    lines[numLines] = composite_kernel_start;
    lines[2 * numLines + 1] = composite_kernel_end;

    /* Fill the rest with templated kernels */
    for (i = numLines + 1; i < 2 * numLines + 1; i++)
    {
        sprintf(buffer, composite_kernel_template, i - numLines - 1);
        lines[i] = _strdup(buffer);
    }

    /* Try to create a program with these lines */
    error = create_single_kernel_helper_create_program(context, &my_programs[0],
                                                       2 * numLines + 2, lines);
    if (my_programs[0] == NULL || error != CL_SUCCESS)
    {
        log_error("ERROR: Unable to create long test program with %d lines! "
                  "(%s in %s:%d)\n",
                  numLines, IGetErrorString(error), __FILE__, __LINE__);
        return -1;
    }

    /* Compile it */
    error = clCompileProgram(my_programs[0], 1, &deviceID, NULL, 0, NULL, NULL,
                             NULL, NULL);
    test_error(error, "Unable to compile a simple program");

    /* Create and compile templated kernels */
    for (i = 0; i < numLines; i++)
    {
        sprintf(buffer, simple_kernel_template, i);
        const char *kernel_source = _strdup(buffer);
        error = create_single_kernel_helper_create_program(
            context, &my_programs[i + 1], 1, &kernel_source);
        if (my_programs[i + 1] == NULL || error != CL_SUCCESS)
        {
            log_error("ERROR: Unable to create long test program with %d "
                      "lines! (%s in %s:%d)\n",
                      numLines, IGetErrorString(error), __FILE__, __LINE__);
            return -1;
        }

        /* Compile it */
        error = clCompileProgram(my_programs[i + 1], 1, &deviceID, NULL, 0,
                                 NULL, NULL, NULL, NULL);
        test_error(error, "Unable to compile a simple program");

        free((void *)kernel_source);
    }

    /* Link the program that calls the kernels and the library that contains
     * them */
    cl_program my_newly_linked_program =
        clLinkProgram(context, 1, &deviceID, NULL, 1 + numLines, my_programs,
                      NULL, NULL, &error);
    test_error(error, "Unable to link a program with a library");

    // Create the composite kernel
    cl_kernel kernel =
        clCreateKernel(my_newly_linked_program, "CompositeKernel", &error);
    test_error(error, "Unable to create a composite kernel");

    // Run the composite kernel and verify the results
    error = verifyCopyBuffer(context, queue, kernel);
    if (error != CL_SUCCESS) return error;

    /* All done! */
    for (i = 0; i < 1 + numLines; i++)
    {
        error = clReleaseProgram(my_programs[i]);
        test_error(error, "Unable to release program object");
    }
    free(my_programs);
    for (i = 0; i < numLines; i++)
    {
        free((void *)lines[i]);
    }
    for (i = numLines + 1; i < 2 * numLines + 1; i++)
    {
        free((void *)lines[i]);
    }
    free(lines);

    error = clReleaseKernel(kernel);
    test_error(error, "Unable to release kernel object");

    error = clReleaseProgram(my_newly_linked_program);
    test_error(error, "Unable to release program object");

    return 0;
}

int test_multiple_files(cl_device_id deviceID, cl_context context,
                        cl_command_queue queue, int num_elements)
{
    unsigned int toTest[] = { 8, 32, 128, 256,
                              0 }; // 512, 2048, 8192, 32768, 0 };
    unsigned int i;

    log_info("Testing multiple files compilation and linking into a single "
             "executable ...this might take awhile...\n");

    for (i = 0; toTest[i] != 0; i++)
    {
        log_info("   %d...\n", toTest[i]);

#if defined(_WIN32)
        clock_t start = clock();
#elif defined(__linux__) || defined(__APPLE__)
        timeval time1, time2;
        gettimeofday(&time1, NULL);
#endif

        if (test_large_multiple_files(context, deviceID, queue, toTest[i]) != 0)
        {
            log_error("ERROR: multiple files program test failed for %d lines! "
                      "(in %s:%d)\n\n",
                      toTest[i], __FILE__, __LINE__);
            return -1;
        }

#if defined(_WIN32)
        clock_t end = clock();
        log_perf((float)(end - start) / (float)CLOCKS_PER_SEC, false,
                 "clock() time in secs", "%d lines", toTest[i]);
#elif defined(__linux__) || defined(__APPLE__)
        gettimeofday(&time2, NULL);
        log_perf((float)(float)(time2.tv_sec - time1.tv_sec)
                     + 1.0e-6 * (time2.tv_usec - time1.tv_usec),
                 false, "wall time in secs", "%d lines", toTest[i]);
#endif
    }

    return 0;
}

int test_simple_compile_only(cl_device_id deviceID, cl_context context,
                             cl_command_queue queue, int num_elements)
{
    int error;
    cl_program program;

    log_info("Testing a simple compilation only...\n");
    error = create_single_kernel_helper_create_program(context, &program, 1,
                                                       &simple_kernel);
    if (program == NULL || error != CL_SUCCESS)
    {
        log_error(
            "ERROR: Unable to create a simple test program! (%s in %s:%d)\n",
            IGetErrorString(error), __FILE__, __LINE__);
        return -1;
    }

    error = clCompileProgram(program, 1, &deviceID, NULL, 0, NULL, NULL, NULL,
                             NULL);
    test_error(error, "Unable to compile a simple program");

    /* All done! */
    error = clReleaseProgram(program);
    test_error(error, "Unable to release program object");

    return 0;
}

int test_simple_static_compile_only(cl_device_id deviceID, cl_context context,
                                    cl_command_queue queue, int num_elements)
{
    int error;
    cl_program program;

    log_info("Testing a simple static compilations only...\n");

    error = create_single_kernel_helper_create_program(context, &program, 1,
                                                       &compile_static_var);
    if (program == NULL || error != CL_SUCCESS)
    {
        log_error("ERROR: Unable to create a simple static variable test "
                  "program! (%s in %s:%d)\n",
                  IGetErrorString(error), __FILE__, __LINE__);
        return -1;
    }

    log_info("Compiling a static variable...\n");
    error = clCompileProgram(program, 1, &deviceID, NULL, 0, NULL, NULL, NULL,
                             NULL);
    test_error(error, "Unable to compile a simple static variable program");

    /* All done! */
    error = clReleaseProgram(program);
    test_error(error, "Unable to release program object");

    error = create_single_kernel_helper_create_program(context, &program, 1,
                                                       &compile_static_struct);
    if (program == NULL || error != CL_SUCCESS)
    {
        log_error("ERROR: Unable to create a simple static struct test "
                  "program! (%s in %s:%d)\n",
                  IGetErrorString(error), __FILE__, __LINE__);
        return -1;
    }

    log_info("Compiling a static struct...\n");
    error = clCompileProgram(program, 1, &deviceID, NULL, 0, NULL, NULL, NULL,
                             NULL);
    test_error(error, "Unable to compile a simple static variable program");

    /* All done! */
    error = clReleaseProgram(program);
    test_error(error, "Unable to release program object");

    error = create_single_kernel_helper_create_program(
        context, &program, 1, &compile_static_function);
    if (program == NULL || error != CL_SUCCESS)
    {
        log_error("ERROR: Unable to create a simple static function test "
                  "program! (%s in %s:%d)\n",
                  IGetErrorString(error), __FILE__, __LINE__);
        return -1;
    }

    log_info("Compiling a static function...\n");
    error = clCompileProgram(program, 1, &deviceID, NULL, 0, NULL, NULL, NULL,
                             NULL);
    test_error(error, "Unable to compile a simple static function program");

    /* All done! */
    error = clReleaseProgram(program);
    test_error(error, "Unable to release program object");

    return 0;
}

int test_simple_extern_compile_only(cl_device_id deviceID, cl_context context,
                                    cl_command_queue queue, int num_elements)
{
    int error;
    cl_program program;

    log_info("Testing a simple extern compilations only...\n");
    error = create_single_kernel_helper_create_program(context, &program, 1,
                                                       &simple_header);
    if (program == NULL || error != CL_SUCCESS)
    {
        log_error("ERROR: Unable to create a simple extern kernel test "
                  "program! (%s in %s:%d)\n",
                  IGetErrorString(error), __FILE__, __LINE__);
        return -1;
    }

    log_info("Compiling an extern kernel...\n");
    error = clCompileProgram(program, 1, &deviceID, NULL, 0, NULL, NULL, NULL,
                             NULL);
    test_error(error, "Unable to compile a simple extern kernel program");

    /* All done! */
    error = clReleaseProgram(program);
    test_error(error, "Unable to release program object");

    error = create_single_kernel_helper_create_program(context, &program, 1,
                                                       &compile_extern_var);
    if (program == NULL || error != CL_SUCCESS)
    {
        log_error("ERROR: Unable to create a simple extern variable test "
                  "program! (%s in %s:%d)\n",
                  IGetErrorString(error), __FILE__, __LINE__);
        return -1;
    }

    log_info("Compiling an extern variable...\n");
    error = clCompileProgram(program, 1, &deviceID, NULL, 0, NULL, NULL, NULL,
                             NULL);
    test_error(error, "Unable to compile a simple extern variable program");

    /* All done! */
    error = clReleaseProgram(program);
    test_error(error, "Unable to release program object");

    error = create_single_kernel_helper_create_program(context, &program, 1,
                                                       &compile_extern_struct);
    if (program == NULL || error != CL_SUCCESS)
    {
        log_error("ERROR: Unable to create a simple extern struct test "
                  "program! (%s in %s:%d)\n",
                  IGetErrorString(error), __FILE__, __LINE__);
        return -1;
    }

    log_info("Compiling an extern struct...\n");
    error = clCompileProgram(program, 1, &deviceID, NULL, 0, NULL, NULL, NULL,
                             NULL);
    test_error(error, "Unable to compile a simple extern variable program");

    /* All done! */
    error = clReleaseProgram(program);
    test_error(error, "Unable to release program object");

    error = create_single_kernel_helper_create_program(
        context, &program, 1, &compile_extern_function);
    if (program == NULL || error != CL_SUCCESS)
    {
        log_error("ERROR: Unable to create a simple extern function test "
                  "program! (%s in %s:%d)\n",
                  IGetErrorString(error), __FILE__, __LINE__);
        return -1;
    }

    log_info("Compiling an extern function...\n");
    error = clCompileProgram(program, 1, &deviceID, NULL, 0, NULL, NULL, NULL,
                             NULL);
    test_error(error, "Unable to compile a simple extern function program");

    /* All done! */
    error = clReleaseProgram(program);
    test_error(error, "Unable to release program object");

    return 0;
}

struct simple_user_data
{
    const char *m_message;
    cl_event m_event;
};

const char *once_upon_a_midnight_dreary = "Once upon a midnight dreary!";

static void CL_CALLBACK simple_compile_callback(cl_program program,
                                                void *user_data)
{
    simple_user_data *simple_compile_user_data = (simple_user_data *)user_data;
    log_info("in the simple_compile_callback: program %p just completed "
             "compiling with '%s'\n",
             program, simple_compile_user_data->m_message);
    if (strcmp(once_upon_a_midnight_dreary, simple_compile_user_data->m_message)
        != 0)
    {
        log_error("ERROR: in the simple_compile_callback: Expected '%s' and "
                  "got %s (in %s:%d)!\n",
                  once_upon_a_midnight_dreary,
                  simple_compile_user_data->m_message, __FILE__, __LINE__);
    }

    int error;
    log_info("in the simple_compile_callback: program %p just completed "
             "compiling with '%p'\n",
             program, simple_compile_user_data->m_event);

    error =
        clSetUserEventStatus(simple_compile_user_data->m_event, CL_COMPLETE);
    if (error != CL_SUCCESS)
    {
        log_error("ERROR: in the simple_compile_callback: Unable to set user "
                  "event status to CL_COMPLETE! (%s in %s:%d)\n",
                  IGetErrorString(error), __FILE__, __LINE__);
        exit(-1);
    }
    log_info("in the simple_compile_callback: Successfully signaled "
             "compile_program_completion_event!\n");
}

int test_simple_compile_with_callback(cl_device_id deviceID, cl_context context,
                                      cl_command_queue queue, int num_elements)
{
    int error;
    cl_program program;
    cl_event compile_program_completion_event;

    log_info("Testing a simple compilation with callback...\n");
    error = create_single_kernel_helper_create_program(context, &program, 1,
                                                       &simple_kernel);
    if (program == NULL || error != CL_SUCCESS)
    {
        log_error(
            "ERROR: Unable to create a simple test program! (%s in %s:%d)\n",
            IGetErrorString(error), __FILE__, __LINE__);
        return -1;
    }

    compile_program_completion_event = clCreateUserEvent(context, &error);
    test_error(error, "Unable to create a user event");

    simple_user_data simple_compile_user_data = {
        once_upon_a_midnight_dreary, compile_program_completion_event
    };

    error = clCompileProgram(program, 1, &deviceID, NULL, 0, NULL, NULL,
                             simple_compile_callback,
                             (void *)&simple_compile_user_data);
    test_error(error, "Unable to compile a simple program with a callback");

    error = clWaitForEvents(1, &compile_program_completion_event);
    test_error(error,
               "clWaitForEvents failed when waiting on "
               "compile_program_completion_event");

    /* All done! */
    error = clReleaseEvent(compile_program_completion_event);
    test_error(error, "Unable to release event object");

    error = clReleaseProgram(program);
    test_error(error, "Unable to release program object");

    return 0;
}

int test_simple_embedded_header_compile(cl_device_id deviceID,
                                        cl_context context,
                                        cl_command_queue queue,
                                        int num_elements)
{
    int error;
    cl_program program, header;

    log_info("Testing a simple embedded header compile only...\n");
    program = clCreateProgramWithSource(
        context, 1, &another_simple_kernel_with_header, NULL, &error);
    if (program == NULL || error != CL_SUCCESS)
    {
        log_error(
            "ERROR: Unable to create a simple test program! (%s in %s:%d)\n",
            IGetErrorString(error), __FILE__, __LINE__);
        return -1;
    }

    header =
        clCreateProgramWithSource(context, 1, &simple_header, NULL, &error);
    if (header == NULL || error != CL_SUCCESS)
    {
        log_error(
            "ERROR: Unable to create a simple header program! (%s in %s:%d)\n",
            IGetErrorString(error), __FILE__, __LINE__);
        return -1;
    }

    error = clCompileProgram(program, 1, &deviceID, NULL, 1, &header,
                             &simple_header_name, NULL, NULL);
    test_error(error,
               "Unable to compile a simple program with embedded header");

    /* All done! */
    error = clReleaseProgram(program);
    test_error(error, "Unable to release program object");

    error = clReleaseProgram(header);
    test_error(error, "Unable to release program object");

    return 0;
}

int test_simple_link_only(cl_device_id deviceID, cl_context context,
                          cl_command_queue queue, int num_elements)
{
    int error;
    cl_program program;

    log_info("Testing a simple linking only...\n");
    error = create_single_kernel_helper_create_program(context, &program, 1,
                                                       &simple_kernel);
    if (program == NULL || error != CL_SUCCESS)
    {
        log_error(
            "ERROR: Unable to create a simple test program! (%s in %s:%d)\n",
            IGetErrorString(error), __FILE__, __LINE__);
        return -1;
    }

    error = clCompileProgram(program, 1, &deviceID, NULL, 0, NULL, NULL, NULL,
                             NULL);
    test_error(error, "Unable to compile a simple program");

    cl_program my_newly_linked_program = clLinkProgram(
        context, 1, &deviceID, NULL, 1, &program, NULL, NULL, &error);
    test_error(error, "Unable to link a simple program");

    /* All done! */
    error = clReleaseProgram(program);
    test_error(error, "Unable to release program object");

    error = clReleaseProgram(my_newly_linked_program);
    test_error(error, "Unable to release program object");

    return 0;
}

int test_two_file_regular_variable_access(cl_device_id deviceID,
                                          cl_context context,
                                          cl_command_queue queue,
                                          int num_elements)
{
    int error;
    cl_program program, second_program, my_newly_linked_program;

    const char *sources[2] = {
        simple_kernel, compile_regular_var
    }; // here we want to avoid linking error due to lack of kernels
    log_info("Compiling and linking two program objects, where one tries to "
             "access regular variable from another...\n");
    error = create_single_kernel_helper_create_program(context, &program, 2,
                                                       sources);
    if (program == NULL || error != CL_SUCCESS)
    {
        log_error("ERROR: Unable to create a test program with regular "
                  "variable! (%s in %s:%d)\n",
                  IGetErrorString(error), __FILE__, __LINE__);
        return -1;
    }

    error = clCompileProgram(program, 1, &deviceID, NULL, 0, NULL, NULL, NULL,
                             NULL);
    test_error(error,
               "Unable to compile a simple program with regular function");

    error = create_single_kernel_helper_create_program(
        context, &second_program, 1, &link_static_var_access);
    if (program == NULL || error != CL_SUCCESS)
    {
        log_error("ERROR: Unable to create a test program that tries to access "
                  "a regular variable! (%s in %s:%d)\n",
                  IGetErrorString(error), __FILE__, __LINE__);
        return -1;
    }

    error = clCompileProgram(second_program, 1, &deviceID, NULL, 0, NULL, NULL,
                             NULL, NULL);
    test_error(
        error,
        "Unable to compile a program that tries to access a regular variable");

    cl_program two_programs[2] = { program, second_program };
    my_newly_linked_program = clLinkProgram(context, 1, &deviceID, NULL, 2,
                                            two_programs, NULL, NULL, &error);
    test_error(error,
               "clLinkProgram: Expected a different error code while linking a "
               "program that tries to access a regular variable");

    /* All done! */
    error = clReleaseProgram(program);
    test_error(error, "Unable to release program object");

    error = clReleaseProgram(second_program);
    test_error(error, "Unable to release program object");

    error = clReleaseProgram(my_newly_linked_program);
    test_error(error, "Unable to release program object");

    return 0;
}

int test_two_file_regular_struct_access(cl_device_id deviceID,
                                        cl_context context,
                                        cl_command_queue queue,
                                        int num_elements)
{
    int error;
    cl_program program, second_program, my_newly_linked_program;

    const char *sources[2] = {
        simple_kernel, compile_regular_struct
    }; // here we want to avoid linking error due to lack of kernels
    log_info("Compiling and linking two program objects, where one tries to "
             "access regular struct from another...\n");
    error = create_single_kernel_helper_create_program(context, &program, 2,
                                                       sources);
    if (program == NULL || error != CL_SUCCESS)
    {
        log_error("ERROR: Unable to create a test program with regular struct! "
                  "(%s in %s:%d)\n",
                  IGetErrorString(error), __FILE__, __LINE__);
        return -1;
    }

    error = clCompileProgram(program, 1, &deviceID, NULL, 0, NULL, NULL, NULL,
                             NULL);
    test_error(error, "Unable to compile a simple program with regular struct");

    error = create_single_kernel_helper_create_program(
        context, &second_program, 1, &link_static_struct_access);
    if (second_program == NULL || error != CL_SUCCESS)
    {
        log_error("ERROR: Unable to create a test program that tries to access "
                  "a regular struct! (%s in %s:%d)\n",
                  IGetErrorString(error), __FILE__, __LINE__);
        return -1;
    }

    error = clCompileProgram(second_program, 1, &deviceID, NULL, 0, NULL, NULL,
                             NULL, NULL);
    test_error(
        error,
        "Unable to compile a program that tries to access a regular struct");

    cl_program two_programs[2] = { program, second_program };
    my_newly_linked_program = clLinkProgram(context, 1, &deviceID, NULL, 2,
                                            two_programs, NULL, NULL, &error);
    test_error(error,
               "clLinkProgram: Expected a different error code while linking a "
               "program that tries to access a regular struct");

    /* All done! */
    error = clReleaseProgram(program);
    test_error(error, "Unable to release program object");

    error = clReleaseProgram(second_program);
    test_error(error, "Unable to release program object");

    error = clReleaseProgram(my_newly_linked_program);
    test_error(error, "Unable to release program object");

    return 0;
}


int test_two_file_regular_function_access(cl_device_id deviceID,
                                          cl_context context,
                                          cl_command_queue queue,
                                          int num_elements)
{
    int error;
    cl_program program, second_program, my_newly_linked_program;

    const char *sources[2] = {
        simple_kernel, compile_regular_function
    }; // here we want to avoid linking error due to lack of kernels
    log_info("Compiling and linking two program objects, where one tries to "
             "access regular function from another...\n");
    error = create_single_kernel_helper_create_program(context, &program, 2,
                                                       sources);
    if (program == NULL || error != CL_SUCCESS)
    {
        log_error("ERROR: Unable to create a test program with regular "
                  "function! (%s in %s:%d)\n",
                  IGetErrorString(error), __FILE__, __LINE__);
        return -1;
    }

    error = clCompileProgram(program, 1, &deviceID, NULL, 0, NULL, NULL, NULL,
                             NULL);
    test_error(error,
               "Unable to compile a simple program with regular function");

    error = create_single_kernel_helper_create_program(
        context, &second_program, 1, &link_static_function_access);
    if (second_program == NULL || error != CL_SUCCESS)
    {
        log_error("ERROR: Unable to create a test program that tries to access "
                  "a regular function! (%s in %s:%d)\n",
                  IGetErrorString(error), __FILE__, __LINE__);
        return -1;
    }

    error = clCompileProgram(second_program, 1, &deviceID, NULL, 0, NULL, NULL,
                             NULL, NULL);
    test_error(
        error,
        "Unable to compile a program that tries to access a regular function");

    cl_program two_programs[2] = { program, second_program };
    my_newly_linked_program = clLinkProgram(context, 1, &deviceID, NULL, 2,
                                            two_programs, NULL, NULL, &error);
    test_error(error,
               "clLinkProgram: Expected a different error code while linking a "
               "program that tries to access a regular function");

    /* All done! */
    error = clReleaseProgram(program);
    test_error(error, "Unable to release program object");

    error = clReleaseProgram(second_program);
    test_error(error, "Unable to release program object");

    error = clReleaseProgram(my_newly_linked_program);
    test_error(error, "Unable to release program object");

    return 0;
}

int test_simple_embedded_header_link(cl_device_id deviceID, cl_context context,
                                     cl_command_queue queue, int num_elements)
{
    int error;
    cl_program program, header, simple_program;

    log_info("Testing a simple embedded header link...\n");
    program = clCreateProgramWithSource(
        context, 1, &another_simple_kernel_with_header, NULL, &error);
    if (program == NULL || error != CL_SUCCESS)
    {
        log_error(
            "ERROR: Unable to create a simple test program! (%s in %s:%d)\n",
            IGetErrorString(error), __FILE__, __LINE__);
        return -1;
    }

    header =
        clCreateProgramWithSource(context, 1, &simple_header, NULL, &error);
    if (header == NULL || error != CL_SUCCESS)
    {
        log_error(
            "ERROR: Unable to create a simple header program! (%s in %s:%d)\n",
            IGetErrorString(error), __FILE__, __LINE__);
        return -1;
    }

    error = clCompileProgram(program, 1, &deviceID, NULL, 1, &header,
                             &simple_header_name, NULL, NULL);
    test_error(error,
               "Unable to compile a simple program with embedded header");

    error = create_single_kernel_helper_create_program(context, &simple_program,
                                                       1, &simple_kernel);
    if (simple_program == NULL || error != CL_SUCCESS)
    {
        log_error(
            "ERROR: Unable to create a simple test program! (%s in %s:%d)\n",
            IGetErrorString(error), __FILE__, __LINE__);
        return -1;
    }

    error = clCompileProgram(simple_program, 1, &deviceID, NULL, 0, NULL, NULL,
                             NULL, NULL);
    test_error(error, "Unable to compile a simple program");

    cl_program two_programs[2] = { program, simple_program };
    cl_program fully_linked_program = clLinkProgram(
        context, 1, &deviceID, "", 2, two_programs, NULL, NULL, &error);
    test_error(error,
               "Unable to create an executable from two binaries, one compiled "
               "with embedded header");

    /* All done! */
    error = clReleaseProgram(program);
    test_error(error, "Unable to release program object");

    error = clReleaseProgram(header);
    test_error(error, "Unable to release program object");

    error = clReleaseProgram(simple_program);
    test_error(error, "Unable to release program object");

    error = clReleaseProgram(fully_linked_program);
    test_error(error, "Unable to release program object");

    return 0;
}

const char *when_i_pondered_weak_and_weary = "When I pondered weak and weary!";

static void CL_CALLBACK simple_link_callback(cl_program program,
                                             void *user_data)
{
    simple_user_data *simple_link_user_data = (simple_user_data *)user_data;
    log_info("in the simple_link_callback: program %p just completed linking "
             "with '%s'\n",
             program, (const char *)simple_link_user_data->m_message);
    if (strcmp(when_i_pondered_weak_and_weary, simple_link_user_data->m_message)
        != 0)
    {
        log_error("ERROR: in the simple_compile_callback: Expected '%s' and "
                  "got %s! (in %s:%d)\n",
                  when_i_pondered_weak_and_weary,
                  simple_link_user_data->m_message, __FILE__, __LINE__);
    }

    int error;
    log_info("in the simple_link_callback: program %p just completed linking "
             "with '%p'\n",
             program, simple_link_user_data->m_event);

    error = clSetUserEventStatus(simple_link_user_data->m_event, CL_COMPLETE);
    if (error != CL_SUCCESS)
    {
        log_error("ERROR: simple_link_callback: Unable to set user event "
                  "status to CL_COMPLETE! (%s in %s:%d)\n",
                  IGetErrorString(error), __FILE__, __LINE__);
        exit(-1);
    }
    log_info("in the simple_link_callback: Successfully signaled "
             "link_program_completion_event event!\n");
}

int test_simple_link_with_callback(cl_device_id deviceID, cl_context context,
                                   cl_command_queue queue, int num_elements)
{
    int error;
    cl_program program;
    cl_event link_program_completion_event;

    log_info("Testing a simple linking with callback...\n");
    error = create_single_kernel_helper_create_program(context, &program, 1,
                                                       &simple_kernel);
    if (program == NULL || error != CL_SUCCESS)
    {
        log_error(
            "ERROR: Unable to create a simple test program! (%s in %s:%d)\n",
            IGetErrorString(error), __FILE__, __LINE__);
        return -1;
    }

    error = clCompileProgram(program, 1, &deviceID, NULL, 0, NULL, NULL, NULL,
                             NULL);
    test_error(error, "Unable to compile a simple program");

    link_program_completion_event = clCreateUserEvent(context, &error);
    test_error(error, "Unable to create a user event");

    simple_user_data simple_link_user_data = { when_i_pondered_weak_and_weary,
                                               link_program_completion_event };

    cl_program my_linked_library = clLinkProgram(
        context, 1, &deviceID, NULL, 1, &program, simple_link_callback,
        (void *)&simple_link_user_data, &error);
    test_error(error, "Unable to link a simple program");

    error = clWaitForEvents(1, &link_program_completion_event);
    test_error(
        error,
        "clWaitForEvents failed when waiting on link_program_completion_event");

    /* All done! */
    error = clReleaseEvent(link_program_completion_event);
    test_error(error, "Unable to release event object");

    error = clReleaseProgram(program);
    test_error(error, "Unable to release program object");

    error = clReleaseProgram(my_linked_library);
    test_error(error, "Unable to release program object");

    return 0;
}

static void initBuffer(float *&srcBuffer, unsigned int cnDimension)
{
    float num = 0.0f;

    for (unsigned int i = 0; i < cnDimension; i++)
    {
        if ((i % 10) == 0)
        {
            num = 0.0f;
        }

        srcBuffer[i] = num;
        num = num + 1.0f;
    }
}

static int verifyCopyBuffer(cl_context context, cl_command_queue queue,
                            cl_kernel kernel)
{
    int error, result = CL_SUCCESS;
    const size_t cnDimension = 32;

    // Allocate source buffer
    float *srcBuffer = (float *)malloc(cnDimension * sizeof(float));
    float *dstBuffer = (float *)malloc(cnDimension * sizeof(float));

    if (srcBuffer == NULL)
    {
        log_error("ERROR: Unable to allocate srcBuffer float array with %lu "
                  "floats! (in %s:%d)\n",
                  cnDimension, __FILE__, __LINE__);
        return -1;
    }
    if (dstBuffer == NULL)
    {
        log_error("ERROR: Unable to allocate dstBuffer float array with %lu "
                  "floats! (in %s:%d)\n",
                  cnDimension, __FILE__, __LINE__);
        return -1;
    }

    if (srcBuffer && dstBuffer)
    {
        // initialize host memory
        initBuffer(srcBuffer, cnDimension);

        // Allocate device memory
        cl_mem deviceMemSrc =
            clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           cnDimension * sizeof(cl_float), srcBuffer, &error);
        test_error(error, "Unable to create a source memory buffer");

        cl_mem deviceMemDst =
            clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                           cnDimension * sizeof(cl_float), 0, &error);
        test_error(error, "Unable to create a destination memory buffer");

        // Set kernel args
        // Set parameter 0 to be the source buffer
        error =
            clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&deviceMemSrc);
        test_error(error, "Unable to set the first kernel argument");

        // Set parameter 1 to be the destination buffer
        error =
            clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&deviceMemDst);
        test_error(error, "Unable to set the second kernel argument");

        // Execute kernel
        error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &cnDimension, 0,
                                       0, NULL, NULL);
        test_error(error, "Unable to enqueue kernel");

        error = clFlush(queue);
        test_error(error, "Unable to flush the queue");

        // copy results from device back to host
        error = clEnqueueReadBuffer(queue, deviceMemDst, CL_TRUE, 0,
                                    cnDimension * sizeof(cl_float), dstBuffer,
                                    0, NULL, NULL);
        test_error(error, "Unable to read the destination buffer");

        error = clFlush(queue);
        test_error(error, "Unable to flush the queue");

        // Compare the source and destination buffers
        const int *pSrc = (int *)srcBuffer;
        const int *pDst = (int *)dstBuffer;
        int mismatch = 0;

        for (size_t i = 0; i < cnDimension; i++)
        {
            if (pSrc[i] != pDst[i])
            {
                if (mismatch < 4)
                {
                    log_info("Offset %08lX:  Expected %08X, Got %08X\n", i * 4,
                             pSrc[i], pDst[i]);
                }
                else
                {
                    log_info(".");
                }
                mismatch++;
            }
        }

        if (mismatch)
        {
            log_info("*** %d mismatches found, TEST FAILS! ***\n", mismatch);
            result = -1;
        }
        else
        {
            log_info("Buffers match, test passes.\n");
        }

        free(srcBuffer);
        srcBuffer = NULL;
        free(dstBuffer);
        dstBuffer = NULL;

        if (deviceMemSrc)
        {
            error = clReleaseMemObject(deviceMemSrc);
            test_error(error, "Unable to release memory object");
        }

        if (deviceMemDst)
        {
            error = clReleaseMemObject(deviceMemDst);
            test_error(error, "Unable to release memory object");
        }
    }
    return result;
}

int test_execute_after_simple_compile_and_link(cl_device_id deviceID,
                                               cl_context context,
                                               cl_command_queue queue,
                                               int num_elements)
{
    int error;
    cl_program program;

    log_info("Testing execution after a simple compile and link...\n");
    error = create_single_kernel_helper_create_program(context, &program, 1,
                                                       &simple_kernel);
    if (program == NULL || error != CL_SUCCESS)
    {
        log_error(
            "ERROR: Unable to create a simple test program! (%s in %s:%d)\n",
            IGetErrorString(error), __FILE__, __LINE__);
        return -1;
    }

    error = clCompileProgram(program, 1, &deviceID, NULL, 0, NULL, NULL, NULL,
                             NULL);
    test_error(error, "Unable to compile a simple program");

    cl_program my_newly_linked_program = clLinkProgram(
        context, 1, &deviceID, NULL, 1, &program, NULL, NULL, &error);
    test_error(error, "Unable to link a simple program");

    cl_kernel kernel =
        clCreateKernel(my_newly_linked_program, "CopyBuffer", &error);
    test_error(error, "Unable to create a simple kernel");

    error = verifyCopyBuffer(context, queue, kernel);
    if (error != CL_SUCCESS) return error;

    /* All done! */
    error = clReleaseKernel(kernel);
    test_error(error, "Unable to release kernel object");

    error = clReleaseProgram(program);
    test_error(error, "Unable to release program object");

    error = clReleaseProgram(my_newly_linked_program);
    test_error(error, "Unable to release program object");

    return 0;
}

int test_execute_after_simple_compile_and_link_no_device_info(
    cl_device_id deviceID, cl_context context, cl_command_queue queue,
    int num_elements)
{
    int error;
    cl_program program;

    log_info("Testing execution after a simple compile and link with no device "
             "information provided...\n");
    error = create_single_kernel_helper_create_program(context, &program, 1,
                                                       &simple_kernel);
    if (program == NULL || error != CL_SUCCESS)
    {
        log_error(
            "ERROR: Unable to create a simple test program! (%s in %s:%d)\n",
            IGetErrorString(error), __FILE__, __LINE__);
        return -1;
    }

    error = clCompileProgram(program, 0, NULL, NULL, 0, NULL, NULL, NULL, NULL);
    test_error(error, "Unable to compile a simple program");

    cl_program my_newly_linked_program =
        clLinkProgram(context, 0, NULL, NULL, 1, &program, NULL, NULL, &error);
    test_error(error, "Unable to link a simple program");

    cl_kernel kernel =
        clCreateKernel(my_newly_linked_program, "CopyBuffer", &error);
    test_error(error, "Unable to create a simple kernel");

    error = verifyCopyBuffer(context, queue, kernel);
    if (error != CL_SUCCESS) return error;

    /* All done! */
    error = clReleaseKernel(kernel);
    test_error(error, "Unable to release kernel object");

    error = clReleaseProgram(program);
    test_error(error, "Unable to release program object");

    error = clReleaseProgram(my_newly_linked_program);
    test_error(error, "Unable to release program object");

    return 0;
}

int test_execute_after_simple_compile_and_link_with_defines(
    cl_device_id deviceID, cl_context context, cl_command_queue queue,
    int num_elements)
{
    int error;
    cl_program program;

    log_info(
        "Testing execution after a simple compile and link with defines...\n");
    error = create_single_kernel_helper_create_program(
        context, &program, 1, &simple_kernel_with_defines,
        "-DFIRST=5 -DSECOND=37");
    if (program == NULL || error != CL_SUCCESS)
    {
        log_error(
            "ERROR: Unable to create a simple test program! (%s in %s:%d)\n",
            IGetErrorString(error), __FILE__, __LINE__);
        return -1;
    }

    error = clCompileProgram(program, 1, &deviceID, "-DFIRST=5 -DSECOND=37", 0,
                             NULL, NULL, NULL, NULL);
    test_error(error, "Unable to compile a simple program");

    cl_program my_newly_linked_program = clLinkProgram(
        context, 1, &deviceID, NULL, 1, &program, NULL, NULL, &error);
    test_error(error, "Unable to link a simple program");

    cl_kernel kernel =
        clCreateKernel(my_newly_linked_program, "CopyBuffer", &error);
    test_error(error, "Unable to create a simple kernel");

    error = verifyCopyBuffer(context, queue, kernel);
    if (error != CL_SUCCESS) return error;

    /* All done! */
    error = clReleaseKernel(kernel);
    test_error(error, "Unable to release kernel object");

    error = clReleaseProgram(program);
    test_error(error, "Unable to release program object");

    error = clReleaseProgram(my_newly_linked_program);
    test_error(error, "Unable to release program object");

    return 0;
}

int test_execute_after_serialize_reload_object(cl_device_id deviceID,
                                               cl_context context,
                                               cl_command_queue queue,
                                               int num_elements)
{
    int error;
    cl_program program;
    size_t binarySize;
    unsigned char *binary;

    log_info("Testing execution after serialization and reloading of the "
             "object...\n");
    error = create_single_kernel_helper_create_program(context, &program, 1,
                                                       &simple_kernel);
    if (program == NULL || error != CL_SUCCESS)
    {
        log_error(
            "ERROR: Unable to create a simple test program! (%s in %s:%d)\n",
            IGetErrorString(error), __FILE__, __LINE__);
        return -1;
    }

    error = clCompileProgram(program, 1, &deviceID, NULL, 0, NULL, NULL, NULL,
                             NULL);
    test_error(error, "Unable to compile a simple program");

    // Get the size of the resulting binary (only one device)
    error = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES,
                             sizeof(binarySize), &binarySize, NULL);
    test_error(error, "Unable to get binary size");

    // Sanity check
    if (binarySize == 0)
    {
        log_error("ERROR: Binary size of program is zero (in %s:%d)\n",
                  __FILE__, __LINE__);
        return -1;
    }

    // Create a buffer and get the actual binary
    binary = (unsigned char *)malloc(sizeof(unsigned char) * binarySize);
    if (binary == NULL)
    {
        log_error("ERROR: Unable to allocate binary character array with %lu "
                  "characters! (in %s:%d)\n",
                  binarySize, __FILE__, __LINE__);
        return -1;
    }

    unsigned char *buffers[1] = { binary };
    cl_int loadErrors[1];

    // Do another sanity check here first
    size_t size;
    error = clGetProgramInfo(program, CL_PROGRAM_BINARIES, 0, NULL, &size);
    test_error(error, "Unable to get expected size of binaries array");
    if (size != sizeof(buffers))
    {
        log_error("ERROR: Expected size of binaries array in clGetProgramInfo "
                  "is incorrect (should be %d, got %d) (in %s:%d)\n",
                  (int)sizeof(buffers), (int)size, __FILE__, __LINE__);
        free(binary);
        return -1;
    }

    error = clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(buffers),
                             &buffers, NULL);
    test_error(error, "Unable to get program binary");

    // use clCreateProgramWithBinary
    cl_program program_with_binary = clCreateProgramWithBinary(
        context, 1, &deviceID, &binarySize, (const unsigned char **)buffers,
        loadErrors, &error);
    test_error(error, "Unable to create program with binary");

    cl_program my_newly_linked_program =
        clLinkProgram(context, 1, &deviceID, NULL, 1, &program_with_binary,
                      NULL, NULL, &error);
    test_error(error, "Unable to link a simple program");

    cl_kernel kernel =
        clCreateKernel(my_newly_linked_program, "CopyBuffer", &error);
    test_error(error, "Unable to create a simple kernel");

    error = verifyCopyBuffer(context, queue, kernel);
    if (error != CL_SUCCESS) return error;

    /* All done! */
    error = clReleaseKernel(kernel);
    test_error(error, "Unable to release kernel object");

    error = clReleaseProgram(program);
    test_error(error, "Unable to release program object");

    error = clReleaseProgram(my_newly_linked_program);
    test_error(error, "Unable to release program object");

    error = clReleaseProgram(program_with_binary);
    test_error(error, "Unable to release program object");

    free(binary);

    return 0;
}

int test_execute_after_serialize_reload_library(cl_device_id deviceID,
                                                cl_context context,
                                                cl_command_queue queue,
                                                int num_elements)
{
    int error;
    cl_program program, another_program;
    size_t binarySize;
    unsigned char *binary;

    log_info(
        "Testing execution after linking a binary with a simple library...\n");
    // we will test creation of a simple library from one file
    error = create_single_kernel_helper_create_program(context, &program, 1,
                                                       &simple_kernel);
    if (program == NULL || error != CL_SUCCESS)
    {
        log_error(
            "ERROR: Unable to create a simple test program! (%s in %s:%d)\n",
            IGetErrorString(error), __FILE__, __LINE__);
        return -1;
    }

    error = clCompileProgram(program, 1, &deviceID, NULL, 0, NULL, NULL, NULL,
                             NULL);
    test_error(error, "Unable to compile a simple program");

    cl_program my_newly_minted_library =
        clLinkProgram(context, 1, &deviceID, "-create-library", 1, &program,
                      NULL, NULL, &error);
    test_error(error, "Unable to create a simple library");


    // Get the size of the resulting library (only one device)
    error = clGetProgramInfo(my_newly_minted_library, CL_PROGRAM_BINARY_SIZES,
                             sizeof(binarySize), &binarySize, NULL);
    test_error(error, "Unable to get binary size");

    // Sanity check
    if (binarySize == 0)
    {
        log_error("ERROR: Binary size of program is zero (in %s:%d)\n",
                  __FILE__, __LINE__);
        return -1;
    }

    // Create a buffer and get the actual binary
    binary = (unsigned char *)malloc(sizeof(unsigned char) * binarySize);
    if (binary == NULL)
    {
        log_error("ERROR: Unable to allocate binary character array with %lu "
                  "characters (in %s:%d)!",
                  binarySize, __FILE__, __LINE__);
        return -1;
    }
    unsigned char *buffers[1] = { binary };
    cl_int loadErrors[1];

    // Do another sanity check here first
    size_t size;
    error = clGetProgramInfo(my_newly_minted_library, CL_PROGRAM_BINARIES, 0,
                             NULL, &size);
    test_error(error, "Unable to get expected size of binaries array");
    if (size != sizeof(buffers))
    {
        log_error("ERROR: Expected size of binaries array in clGetProgramInfo "
                  "is incorrect (should be %d, got %d) (in %s:%d)\n",
                  (int)sizeof(buffers), (int)size, __FILE__, __LINE__);
        free(binary);
        return -1;
    }

    error = clGetProgramInfo(my_newly_minted_library, CL_PROGRAM_BINARIES,
                             sizeof(buffers), &buffers, NULL);
    test_error(error, "Unable to get program binary");

    // use clCreateProgramWithBinary
    cl_program library_with_binary = clCreateProgramWithBinary(
        context, 1, &deviceID, &binarySize, (const unsigned char **)buffers,
        loadErrors, &error);
    test_error(error, "Unable to create program with binary");

    error = create_single_kernel_helper_create_program(
        context, &another_program, 1, &another_simple_kernel);
    if (another_program == NULL || error != CL_SUCCESS)
    {
        log_error(
            "ERROR: Unable to create a simple test program! (%s in %s:%d)\n",
            IGetErrorString(error), __FILE__, __LINE__);
        return -1;
    }

    error = clCompileProgram(another_program, 1, &deviceID, NULL, 0, NULL, NULL,
                             NULL, NULL);
    test_error(error, "Unable to compile a simple program");

    cl_program program_and_archive[2] = { another_program,
                                          library_with_binary };
    cl_program fully_linked_program = clLinkProgram(
        context, 1, &deviceID, "", 2, program_and_archive, NULL, NULL, &error);
    test_error(error,
               "Unable to create an executable from a binary and a library");

    cl_kernel kernel =
        clCreateKernel(fully_linked_program, "CopyBuffer", &error);
    test_error(error, "Unable to create a simple kernel");

    error = verifyCopyBuffer(context, queue, kernel);
    if (error != CL_SUCCESS) return error;

    cl_kernel another_kernel =
        clCreateKernel(fully_linked_program, "AnotherCopyBuffer", &error);
    test_error(error, "Unable to create another simple kernel");

    error = verifyCopyBuffer(context, queue, another_kernel);
    if (error != CL_SUCCESS) return error;

    /* All done! */
    error = clReleaseKernel(kernel);
    test_error(error, "Unable to release kernel object");

    error = clReleaseKernel(another_kernel);
    test_error(error, "Unable to release another kernel object");

    error = clReleaseProgram(program);
    test_error(error, "Unable to release program object");

    error = clReleaseProgram(another_program);
    test_error(error, "Unable to release program object");

    error = clReleaseProgram(my_newly_minted_library);
    test_error(error, "Unable to release program object");

    error = clReleaseProgram(library_with_binary);
    test_error(error, "Unable to release program object");

    error = clReleaseProgram(fully_linked_program);
    test_error(error, "Unable to release program object");

    free(binary);

    return 0;
}

static void CL_CALLBACK program_compile_completion_callback(cl_program program,
                                                            void *user_data)
{
    int error;
    cl_event compile_program_completion_event = (cl_event)user_data;
    log_info("in the program_compile_completion_callback: program %p just "
             "completed compiling with '%p'\n",
             program, compile_program_completion_event);

    error = clSetUserEventStatus(compile_program_completion_event, CL_COMPLETE);
    if (error != CL_SUCCESS)
    {
        log_error("ERROR: in the program_compile_completion_callback: Unable "
                  "to set user event status to CL_COMPLETE! (%s in %s:%d)\n",
                  IGetErrorString(error), __FILE__, __LINE__);
        exit(-1);
    }
    log_info("in the program_compile_completion_callback: Successfully "
             "signaled compile_program_completion_event event!\n");
}

static void CL_CALLBACK program_link_completion_callback(cl_program program,
                                                         void *user_data)
{
    int error;
    cl_event link_program_completion_event = (cl_event)user_data;
    log_info("in the program_link_completion_callback: program %p just "
             "completed linking with '%p'\n",
             program, link_program_completion_event);

    error = clSetUserEventStatus(link_program_completion_event, CL_COMPLETE);
    if (error != CL_SUCCESS)
    {
        log_error("ERROR: in the program_link_completion_callback: Unable to "
                  "set user event status to CL_COMPLETE! (%s in %s:%d)\n",
                  IGetErrorString(error), __FILE__, __LINE__);
        exit(-1);
    }
    log_info("in the program_link_completion_callback: Successfully signaled "
             "link_program_completion_event event!\n");
}

int test_execute_after_simple_compile_and_link_with_callbacks(
    cl_device_id deviceID, cl_context context, cl_command_queue queue,
    int num_elements)
{
    int error;
    cl_program program;
    cl_event compile_program_completion_event, link_program_completion_event;

    log_info("Testing execution after a simple compile and link with "
             "callbacks...\n");
    error = create_single_kernel_helper_create_program(context, &program, 1,
                                                       &simple_kernel);
    if (program == NULL || error != CL_SUCCESS)
    {
        log_error(
            "ERROR: Unable to create a simple test program! (%s in %s:%d)\n",
            IGetErrorString(error), __FILE__, __LINE__);
        return -1;
    }

    compile_program_completion_event = clCreateUserEvent(context, &error);
    test_error(error, "Unable to create a user event");

    error = clCompileProgram(program, 1, &deviceID, NULL, 0, NULL, NULL,
                             program_compile_completion_callback,
                             (void *)compile_program_completion_event);
    test_error(error, "Unable to compile a simple program");

    error = clWaitForEvents(1, &compile_program_completion_event);
    test_error(error,
               "clWaitForEvents failed when waiting on "
               "compile_program_completion_event");

    error = clReleaseEvent(compile_program_completion_event);
    test_error(error, "Unable to release event object");

    link_program_completion_event = clCreateUserEvent(context, &error);
    test_error(error, "Unable to create a user event");

    cl_program my_newly_linked_program =
        clLinkProgram(context, 1, &deviceID, NULL, 1, &program,
                      program_link_completion_callback,
                      (void *)link_program_completion_event, &error);
    test_error(error, "Unable to link a simple program");

    error = clWaitForEvents(1, &link_program_completion_event);
    test_error(
        error,
        "clWaitForEvents failed when waiting on link_program_completion_event");

    error = clReleaseEvent(link_program_completion_event);
    test_error(error, "Unable to release event object");

    cl_kernel kernel =
        clCreateKernel(my_newly_linked_program, "CopyBuffer", &error);
    test_error(error, "Unable to create a simple kernel");

    error = verifyCopyBuffer(context, queue, kernel);
    if (error != CL_SUCCESS) return error;

    /* All done! */
    error = clReleaseKernel(kernel);
    test_error(error, "Unable to release kernel object");

    error = clReleaseProgram(program);
    test_error(error, "Unable to release program object");

    error = clReleaseProgram(my_newly_linked_program);
    test_error(error, "Unable to release program object");

    return 0;
}

int test_simple_library_only(cl_device_id deviceID, cl_context context,
                             cl_command_queue queue, int num_elements)
{
    int error;
    cl_program program;

    log_info("Testing creation of a simple library...\n");
    error = create_single_kernel_helper_create_program(context, &program, 1,
                                                       &simple_kernel);
    if (program == NULL || error != CL_SUCCESS)
    {
        log_error(
            "ERROR: Unable to create a simple test program! (%s in %s:%d)\n",
            IGetErrorString(error), __FILE__, __LINE__);
        return -1;
    }

    error = clCompileProgram(program, 1, &deviceID, NULL, 0, NULL, NULL, NULL,
                             NULL);
    test_error(error, "Unable to compile a simple program");

    cl_program my_newly_minted_library =
        clLinkProgram(context, 1, &deviceID, "-create-library", 1, &program,
                      NULL, NULL, &error);
    test_error(error, "Unable to create a simple library");

    /* All done! */
    error = clReleaseProgram(program);
    test_error(error, "Unable to release program object");

    error = clReleaseProgram(my_newly_minted_library);
    test_error(error, "Unable to release program object");

    return 0;
}

int test_simple_library_with_callback(cl_device_id deviceID, cl_context context,
                                      cl_command_queue queue, int num_elements)
{
    int error;
    cl_program program;
    cl_event link_program_completion_event;

    log_info("Testing creation of a simple library with a callback...\n");
    error = create_single_kernel_helper_create_program(context, &program, 1,
                                                       &simple_kernel);
    if (program == NULL || error != CL_SUCCESS)
    {
        log_error(
            "ERROR: Unable to create a simple test program! (%s in %s:%d)\n",
            IGetErrorString(error), __FILE__, __LINE__);
        return -1;
    }

    error = clCompileProgram(program, 1, &deviceID, NULL, 0, NULL, NULL, NULL,
                             NULL);
    test_error(error, "Unable to compile a simple program");

    link_program_completion_event = clCreateUserEvent(context, &error);
    test_error(error, "Unable to create a user event");

    simple_user_data simple_link_user_data = { when_i_pondered_weak_and_weary,
                                               link_program_completion_event };

    cl_program my_newly_minted_library = clLinkProgram(
        context, 1, &deviceID, "-create-library", 1, &program,
        simple_link_callback, (void *)&simple_link_user_data, &error);
    test_error(error, "Unable to create a simple library");

    error = clWaitForEvents(1, &link_program_completion_event);
    test_error(
        error,
        "clWaitForEvents failed when waiting on link_program_completion_event");

    /* All done! */
    error = clReleaseEvent(link_program_completion_event);
    test_error(error, "Unable to release event object");

    error = clReleaseProgram(program);
    test_error(error, "Unable to release program object");

    error = clReleaseProgram(my_newly_minted_library);
    test_error(error, "Unable to release program object");

    return 0;
}

int test_simple_library_with_link(cl_device_id deviceID, cl_context context,
                                  cl_command_queue queue, int num_elements)
{
    int error;
    cl_program program, another_program;

    log_info("Testing creation and linking with a simple library...\n");
    error = create_single_kernel_helper_create_program(context, &program, 1,
                                                       &simple_kernel);
    if (program == NULL || error != CL_SUCCESS)
    {
        log_error(
            "ERROR: Unable to create a simple test program! (%s in %s:%d)\n",
            IGetErrorString(error), __FILE__, __LINE__);
        return -1;
    }

    error = clCompileProgram(program, 1, &deviceID, NULL, 0, NULL, NULL, NULL,
                             NULL);
    test_error(error, "Unable to compile a simple program");

    cl_program my_newly_minted_library =
        clLinkProgram(context, 1, &deviceID, "-create-library", 1, &program,
                      NULL, NULL, &error);
    test_error(error, "Unable to create a simple library");

    error = create_single_kernel_helper_create_program(
        context, &another_program, 1, &another_simple_kernel);
    if (another_program == NULL || error != CL_SUCCESS)
    {
        log_error(
            "ERROR: Unable to create a simple test program! (%s in %s:%d)\n",
            IGetErrorString(error), __FILE__, __LINE__);
        return -1;
    }

    error = clCompileProgram(another_program, 1, &deviceID, NULL, 0, NULL, NULL,
                             NULL, NULL);
    test_error(error, "Unable to compile a simple program");

    cl_program program_and_archive[2] = { another_program,
                                          my_newly_minted_library };
    cl_program fully_linked_program = clLinkProgram(
        context, 1, &deviceID, "", 2, program_and_archive, NULL, NULL, &error);
    test_error(error,
               "Unable to create an executable from a binary and a library");

    /* All done! */
    error = clReleaseProgram(program);
    test_error(error, "Unable to release program object");

    error = clReleaseProgram(another_program);
    test_error(error, "Unable to release program object");

    error = clReleaseProgram(my_newly_minted_library);
    test_error(error, "Unable to release program object");

    error = clReleaseProgram(fully_linked_program);
    test_error(error, "Unable to release program object");

    return 0;
}

int test_execute_after_simple_library_with_link(cl_device_id deviceID,
                                                cl_context context,
                                                cl_command_queue queue,
                                                int num_elements)
{
    int error;
    cl_program program, another_program;

    log_info(
        "Testing execution after linking a binary with a simple library...\n");
    error = create_single_kernel_helper_create_program(context, &program, 1,
                                                       &simple_kernel);
    if (program == NULL || error != CL_SUCCESS)
    {
        log_error(
            "ERROR: Unable to create a simple test program! (%s in %s:%d)\n",
            IGetErrorString(error), __FILE__, __LINE__);
        return -1;
    }

    error = clCompileProgram(program, 1, &deviceID, NULL, 0, NULL, NULL, NULL,
                             NULL);
    test_error(error, "Unable to compile a simple program");

    cl_program my_newly_minted_library =
        clLinkProgram(context, 1, &deviceID, "-create-library", 1, &program,
                      NULL, NULL, &error);
    test_error(error, "Unable to create a simple library");

    error = create_single_kernel_helper_create_program(
        context, &another_program, 1, &another_simple_kernel);
    if (another_program == NULL || error != CL_SUCCESS)
    {
        log_error(
            "ERROR: Unable to create a simple test program! (%s in %s:%d)\n",
            IGetErrorString(error), __FILE__, __LINE__);
        return -1;
    }

    error = clCompileProgram(another_program, 1, &deviceID, NULL, 0, NULL, NULL,
                             NULL, NULL);
    test_error(error, "Unable to compile a simple program");

    cl_program program_and_archive[2] = { another_program,
                                          my_newly_minted_library };
    cl_program fully_linked_program = clLinkProgram(
        context, 1, &deviceID, "", 2, program_and_archive, NULL, NULL, &error);
    test_error(error,
               "Unable to create an executable from a binary and a library");

    cl_kernel kernel =
        clCreateKernel(fully_linked_program, "CopyBuffer", &error);
    test_error(error, "Unable to create a simple kernel");

    error = verifyCopyBuffer(context, queue, kernel);
    if (error != CL_SUCCESS) return error;

    cl_kernel another_kernel =
        clCreateKernel(fully_linked_program, "AnotherCopyBuffer", &error);
    test_error(error, "Unable to create another simple kernel");

    error = verifyCopyBuffer(context, queue, another_kernel);
    if (error != CL_SUCCESS) return error;

    /* All done! */
    error = clReleaseKernel(kernel);
    test_error(error, "Unable to release kernel object");

    error = clReleaseKernel(another_kernel);
    test_error(error, "Unable to release another kernel object");

    error = clReleaseProgram(program);
    test_error(error, "Unable to release program object");

    error = clReleaseProgram(another_program);
    test_error(error, "Unable to release program object");

    error = clReleaseProgram(my_newly_minted_library);
    test_error(error, "Unable to release program object");

    error = clReleaseProgram(fully_linked_program);
    test_error(error, "Unable to release program object");

    return 0;
}

int test_two_file_link(cl_device_id deviceID, cl_context context,
                       cl_command_queue queue, int num_elements)
{
    int error;
    cl_program program, another_program;

    log_info("Testing two file compiling and linking...\n");
    error = create_single_kernel_helper_create_program(context, &program, 1,
                                                       &simple_kernel);
    if (program == NULL || error != CL_SUCCESS)
    {
        log_error(
            "ERROR: Unable to create a simple test program! (%s in %s:%d)\n",
            IGetErrorString(error), __FILE__, __LINE__);
        return -1;
    }

    error = clCompileProgram(program, 1, &deviceID, NULL, 0, NULL, NULL, NULL,
                             NULL);
    test_error(error, "Unable to compile a simple program");


    error = create_single_kernel_helper_create_program(
        context, &another_program, 1, &another_simple_kernel);
    if (another_program == NULL || error != CL_SUCCESS)
    {
        log_error(
            "ERROR: Unable to create a simple test program! (%s in %s:%d)\n",
            IGetErrorString(error), __FILE__, __LINE__);
        return -1;
    }

    error = clCompileProgram(another_program, 1, &deviceID, NULL, 0, NULL, NULL,
                             NULL, NULL);
    test_error(error, "Unable to compile a simple program");

    cl_program two_programs[2] = { program, another_program };
    cl_program fully_linked_program = clLinkProgram(
        context, 1, &deviceID, "", 2, two_programs, NULL, NULL, &error);
    test_error(error, "Unable to create an executable from two binaries");

    /* All done! */
    error = clReleaseProgram(program);
    test_error(error, "Unable to release program object");

    error = clReleaseProgram(another_program);
    test_error(error, "Unable to release program object");

    error = clReleaseProgram(fully_linked_program);
    test_error(error, "Unable to release program object");

    return 0;
}

int test_execute_after_two_file_link(cl_device_id deviceID, cl_context context,
                                     cl_command_queue queue, int num_elements)
{
    int error;
    cl_program program, another_program;

    log_info("Testing two file compiling and linking and execution of two "
             "kernels afterwards ...\n");
    error = create_single_kernel_helper_create_program(context, &program, 1,
                                                       &simple_kernel);
    if (program == NULL || error != CL_SUCCESS)
    {
        log_error(
            "ERROR: Unable to create a simple test program! (%s in %s:%d)\n",
            IGetErrorString(error), __FILE__, __LINE__);
        return -1;
    }

    error = clCompileProgram(program, 1, &deviceID, NULL, 0, NULL, NULL, NULL,
                             NULL);
    test_error(error, "Unable to compile a simple program");

    error = create_single_kernel_helper_create_program(
        context, &another_program, 1, &another_simple_kernel);
    if (another_program == NULL || error != CL_SUCCESS)
    {
        log_error(
            "ERROR: Unable to create a simple test program! (%s in %s:%d)\n",
            IGetErrorString(error), __FILE__, __LINE__);
        return -1;
    }

    error = clCompileProgram(another_program, 1, &deviceID, NULL, 0, NULL, NULL,
                             NULL, NULL);
    test_error(error, "Unable to compile a simple program");

    cl_program two_programs[2] = { program, another_program };
    cl_program fully_linked_program = clLinkProgram(
        context, 1, &deviceID, "", 2, two_programs, NULL, NULL, &error);
    test_error(error, "Unable to create an executable from two binaries");

    cl_kernel kernel =
        clCreateKernel(fully_linked_program, "CopyBuffer", &error);
    test_error(error, "Unable to create a simple kernel");

    error = verifyCopyBuffer(context, queue, kernel);
    if (error != CL_SUCCESS) return error;

    cl_kernel another_kernel =
        clCreateKernel(fully_linked_program, "AnotherCopyBuffer", &error);
    test_error(error, "Unable to create another simple kernel");

    error = verifyCopyBuffer(context, queue, another_kernel);
    if (error != CL_SUCCESS) return error;

    /* All done! */
    error = clReleaseKernel(kernel);
    test_error(error, "Unable to release kernel object");

    error = clReleaseKernel(another_kernel);
    test_error(error, "Unable to release another kernel object");

    error = clReleaseProgram(program);
    test_error(error, "Unable to release program object");

    error = clReleaseProgram(another_program);
    test_error(error, "Unable to release program object");

    error = clReleaseProgram(fully_linked_program);
    test_error(error, "Unable to release program object");

    return 0;
}

int test_execute_after_embedded_header_link(cl_device_id deviceID,
                                            cl_context context,
                                            cl_command_queue queue,
                                            int num_elements)
{
    int error;
    cl_program program, header, simple_program;

    log_info("Testing execution after embedded header link...\n");
    // we will test execution after compiling and linking with embedded headers
    program = clCreateProgramWithSource(
        context, 1, &another_simple_kernel_with_header, NULL, &error);
    if (program == NULL || error != CL_SUCCESS)
    {
        log_error(
            "ERROR: Unable to create a simple test program! (%s in %s:%d)\n",
            IGetErrorString(error), __FILE__, __LINE__);
        return -1;
    }

    header =
        clCreateProgramWithSource(context, 1, &simple_header, NULL, &error);
    if (header == NULL || error != CL_SUCCESS)
    {
        log_error(
            "ERROR: Unable to create a simple header program! (%s in %s:%d)\n",
            IGetErrorString(error), __FILE__, __LINE__);
        return -1;
    }

    error = clCompileProgram(program, 1, &deviceID, NULL, 1, &header,
                             &simple_header_name, NULL, NULL);
    test_error(error,
               "Unable to compile a simple program with embedded header");

    simple_program =
        clCreateProgramWithSource(context, 1, &simple_kernel, NULL, &error);
    if (simple_program == NULL || error != CL_SUCCESS)
    {
        log_error(
            "ERROR: Unable to create a simple test program! (%s in %s:%d)\n",
            IGetErrorString(error), __FILE__, __LINE__);
        return -1;
    }

    error = clCompileProgram(simple_program, 1, &deviceID, NULL, 0, NULL, NULL,
                             NULL, NULL);
    test_error(error, "Unable to compile a simple program");

    cl_program two_programs[2] = { program, simple_program };
    cl_program fully_linked_program = clLinkProgram(
        context, 1, &deviceID, "", 2, two_programs, NULL, NULL, &error);
    test_error(error,
               "Unable to create an executable from two binaries, one compiled "
               "with embedded header");

    cl_kernel kernel =
        clCreateKernel(fully_linked_program, "CopyBuffer", &error);
    test_error(error, "Unable to create a simple kernel");

    error = verifyCopyBuffer(context, queue, kernel);
    if (error != CL_SUCCESS) return error;

    cl_kernel another_kernel =
        clCreateKernel(fully_linked_program, "AnotherCopyBuffer", &error);
    test_error(error, "Unable to create another simple kernel");

    error = verifyCopyBuffer(context, queue, another_kernel);
    if (error != CL_SUCCESS) return error;

    /* All done! */
    error = clReleaseKernel(kernel);
    test_error(error, "Unable to release kernel object");

    error = clReleaseKernel(another_kernel);
    test_error(error, "Unable to release another kernel object");

    error = clReleaseProgram(program);
    test_error(error, "Unable to release program object");

    error = clReleaseProgram(header);
    test_error(error, "Unable to release program object");

    error = clReleaseProgram(simple_program);
    test_error(error, "Unable to release program object");

    error = clReleaseProgram(fully_linked_program);
    test_error(error, "Unable to release program object");

    return 0;
}

#if defined(__APPLE__) || defined(__linux)
#define _mkdir(x) mkdir(x, S_IRWXU)
#define _chdir chdir
#define _rmdir rmdir
#define _unlink unlink
#else
#include <direct.h>
#endif

int test_execute_after_included_header_link(cl_device_id deviceID,
                                            cl_context context,
                                            cl_command_queue queue,
                                            int num_elements)
{
    int error;
    cl_program program, simple_program;

    log_info("Testing execution after included header link...\n");
    // we will test execution after compiling and linking with included headers
    program = clCreateProgramWithSource(
        context, 1, &another_simple_kernel_with_header, NULL, &error);
    if (program == NULL || error != CL_SUCCESS)
    {
        log_error(
            "ERROR: Unable to create a simple test program! (%s in %s:%d)\n",
            IGetErrorString(error), __FILE__, __LINE__);
        return -1;
    }

    /* setup */
#if (defined(__linux__) || defined(__APPLE__)) && (!defined(__ANDROID__))
    /* Some tests systems doesn't allow one to write in the test directory */
    if (_chdir("/tmp") != 0)
    {
        log_error("ERROR: Unable to remove directory foo/bar! (in %s:%d)\n",
                  __FILE__, __LINE__);
        return -1;
    }
#endif
    if (_mkdir("foo") != 0)
    {
        log_error("ERROR: Unable to create directory foo! (in %s:%d)\n",
                  __FILE__, __LINE__);
        return -1;
    }
    if (_mkdir("foo/bar") != 0)
    {
        log_error("ERROR: Unable to create directory foo/bar! (in %s:%d)\n",
                  __FILE__, __LINE__);
        return -1;
    }
    if (_chdir("foo/bar") != 0)
    {
        log_error("ERROR: Unable to change to directory foo/bar! (in %s:%d)\n",
                  __FILE__, __LINE__);
        return -1;
    }
    FILE *simple_header_file = fopen(simple_header_name, "w");
    if (simple_header_file == NULL)
    {
        log_error("ERROR: Unable to create simple header file %s! (in %s:%d)\n",
                  simple_header_name, __FILE__, __LINE__);
        return -1;
    }
    if (fprintf(simple_header_file, "%s", simple_header) < 0)
    {
        log_error(
            "ERROR: Unable to write to simple header file %s! (in %s:%d)\n",
            simple_header_name, __FILE__, __LINE__);
        return -1;
    }
    if (fclose(simple_header_file) != 0)
    {
        log_error("ERROR: Unable to close simple header file %s! (in %s:%d)\n",
                  simple_header_name, __FILE__, __LINE__);
        return -1;
    }
    if (_chdir("../..") != 0)
    {
        log_error("ERROR: Unable to change to original working directory! (in "
                  "%s:%d)\n",
                  __FILE__, __LINE__);
        return -1;
    }
#if (defined(__linux__) || defined(__APPLE__)) && (!defined(__ANDROID__))
    error = clCompileProgram(program, 1, &deviceID, "-I/tmp/foo/bar", 0, NULL,
                             NULL, NULL, NULL);
#else
    error = clCompileProgram(program, 1, &deviceID, "-Ifoo/bar", 0, NULL, NULL,
                             NULL, NULL);
#endif
    test_error(error,
               "Unable to compile a simple program with included header");

    /* cleanup */
    if (_chdir("foo/bar") != 0)
    {
        log_error("ERROR: Unable to change to directory foo/bar! (in %s:%d)\n",
                  __FILE__, __LINE__);
        return -1;
    }
    if (_unlink(simple_header_name) != 0)
    {
        log_error("ERROR: Unable to remove simple header file %s! (in %s:%d)\n",
                  simple_header_name, __FILE__, __LINE__);
        return -1;
    }
    if (_chdir("../..") != 0)
    {
        log_error("ERROR: Unable to change to original working directory! (in "
                  "%s:%d)\n",
                  __FILE__, __LINE__);
        return -1;
    }
    if (_rmdir("foo/bar") != 0)
    {
        log_error("ERROR: Unable to remove directory foo/bar! (in %s:%d)\n",
                  __FILE__, __LINE__);
        return -1;
    }
    if (_rmdir("foo") != 0)
    {
        log_error("ERROR: Unable to remove directory foo! (in %s:%d)\n",
                  __FILE__, __LINE__);
        return -1;
    }

    simple_program =
        clCreateProgramWithSource(context, 1, &simple_kernel, NULL, &error);
    if (simple_program == NULL || error != CL_SUCCESS)
    {
        log_error(
            "ERROR: Unable to create a simple test program! (%s in %s:%d)\n",
            IGetErrorString(error), __FILE__, __LINE__);
        return -1;
    }

    error = clCompileProgram(simple_program, 1, &deviceID, NULL, 0, NULL, NULL,
                             NULL, NULL);
    test_error(error, "Unable to compile a simple program");

    cl_program two_programs[2] = { program, simple_program };
    cl_program fully_linked_program = clLinkProgram(
        context, 1, &deviceID, "", 2, two_programs, NULL, NULL, &error);
    test_error(error,
               "Unable to create an executable from two binaries, one compiled "
               "with embedded header");

    cl_kernel kernel =
        clCreateKernel(fully_linked_program, "CopyBuffer", &error);
    test_error(error, "Unable to create a simple kernel");

    error = verifyCopyBuffer(context, queue, kernel);
    if (error != CL_SUCCESS) return error;

    cl_kernel another_kernel =
        clCreateKernel(fully_linked_program, "AnotherCopyBuffer", &error);
    test_error(error, "Unable to create another simple kernel");

    error = verifyCopyBuffer(context, queue, another_kernel);
    if (error != CL_SUCCESS) return error;

    /* All done! */
    error = clReleaseKernel(kernel);
    test_error(error, "Unable to release kernel object");

    error = clReleaseKernel(another_kernel);
    test_error(error, "Unable to release another kernel object");

    error = clReleaseProgram(program);
    test_error(error, "Unable to release program object");

    error = clReleaseProgram(simple_program);
    test_error(error, "Unable to release program object");

    error = clReleaseProgram(fully_linked_program);
    test_error(error, "Unable to release program object");

    return 0;
}

int test_program_binary_type(cl_device_id deviceID, cl_context context,
                             cl_command_queue queue, int num_elements)
{
    int error;
    cl_program program, another_program, program_with_binary,
        fully_linked_program_with_binary;
    cl_program_binary_type program_type = -1;
    size_t size;
    size_t binarySize;
    unsigned char *binary;

    log_info("Testing querying of program binary type...\n");
    error = create_single_kernel_helper_create_program(context, &program, 1,
                                                       &simple_kernel);
    if (program == NULL || error != CL_SUCCESS)
    {
        log_error(
            "ERROR: Unable to create a simple test program! (%s in %s:%d)\n",
            IGetErrorString(error), __FILE__, __LINE__);
        return -1;
    }

    error = clCompileProgram(program, 1, &deviceID, NULL, 0, NULL, NULL, NULL,
                             NULL);
    test_error(error, "Unable to compile a simple program");

    error = clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BINARY_TYPE,
                                  sizeof(cl_program_binary_type), &program_type,
                                  NULL);
    test_error(error, "Unable to get program binary type");
    if (program_type != CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT)
    {
        log_error("ERROR: Expected program type of a just compiled program to "
                  "be CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT (in %s:%d)\n",
                  __FILE__, __LINE__);
        return -1;
    }
    program_type = -1;

    // Get the size of the resulting binary (only one device)
    error = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES,
                             sizeof(binarySize), &binarySize, NULL);
    test_error(error, "Unable to get binary size");

    // Sanity check
    if (binarySize == 0)
    {
        log_error("ERROR: Binary size of program is zero (in %s:%d)\n",
                  __FILE__, __LINE__);
        return -1;
    }

    // Create a buffer and get the actual binary
    {
        binary = (unsigned char *)malloc(sizeof(unsigned char) * binarySize);
        if (binary == NULL)
        {
            log_error("ERROR: Unable to allocate binary character array with "
                      "%lu characters! (in %s:%d)\n",
                      binarySize, __FILE__, __LINE__);
            return -1;
        }
        unsigned char *buffers[1] = { binary };
        cl_int loadErrors[1];

        // Do another sanity check here first
        size_t size;
        error = clGetProgramInfo(program, CL_PROGRAM_BINARIES, 0, NULL, &size);
        test_error(error, "Unable to get expected size of binaries array");
        if (size != sizeof(buffers))
        {
            log_error(
                "ERROR: Expected size of binaries array in clGetProgramInfo is "
                "incorrect (should be %d, got %d) (in %s:%d)\n",
                (int)sizeof(buffers), (int)size, __FILE__, __LINE__);
            free(binary);
            return -1;
        }

        error = clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(buffers),
                                 &buffers, NULL);
        test_error(error, "Unable to get program binary");

        // use clCreateProgramWithBinary
        program_with_binary = clCreateProgramWithBinary(
            context, 1, &deviceID, &binarySize, (const unsigned char **)buffers,
            loadErrors, &error);
        test_error(error, "Unable to create program with binary");

        error = clGetProgramBuildInfo(
            program_with_binary, deviceID, CL_PROGRAM_BINARY_TYPE,
            sizeof(cl_program_binary_type), &program_type, NULL);
        test_error(error, "Unable to get program binary type");
        if (program_type != CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT)
        {
            log_error("ERROR: Expected program type of a program created from "
                      "compiled object to be "
                      "CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT (in %s:%d)\n",
                      __FILE__, __LINE__);
            return -1;
        }
        program_type = -1;
        free(binary);
    }

    cl_program my_newly_minted_library =
        clLinkProgram(context, 1, &deviceID, "-create-library", 1,
                      &program_with_binary, NULL, NULL, &error);
    test_error(error, "Unable to create a simple library");
    error = clGetProgramBuildInfo(
        my_newly_minted_library, deviceID, CL_PROGRAM_BINARY_TYPE,
        sizeof(cl_program_binary_type), &program_type, NULL);
    test_error(error, "Unable to get program binary type");
    if (program_type != CL_PROGRAM_BINARY_TYPE_LIBRARY)
    {
        log_error("ERROR: Expected program type of a just linked library to be "
                  "CL_PROGRAM_BINARY_TYPE_LIBRARY (in %s:%d)\n",
                  __FILE__, __LINE__);
        return -1;
    }
    program_type = -1;

    // Get the size of the resulting library (only one device)
    error = clGetProgramInfo(my_newly_minted_library, CL_PROGRAM_BINARY_SIZES,
                             sizeof(binarySize), &binarySize, NULL);
    test_error(error, "Unable to get binary size");

    // Sanity check
    if (binarySize == 0)
    {
        log_error("ERROR: Binary size of program is zero (in %s:%d)\n",
                  __FILE__, __LINE__);
        return -1;
    }

    // Create a buffer and get the actual binary
    binary = (unsigned char *)malloc(sizeof(unsigned char) * binarySize);
    if (binary == NULL)
    {
        log_error("ERROR: Unable to allocate binary character array with %lu "
                  "characters! (in %s:%d)\n",
                  binarySize, __FILE__, __LINE__);
        return -1;
    }

    unsigned char *buffers[1] = { binary };
    cl_int loadErrors[1];

    // Do another sanity check here first
    error = clGetProgramInfo(my_newly_minted_library, CL_PROGRAM_BINARIES, 0,
                             NULL, &size);
    test_error(error, "Unable to get expected size of binaries array");
    if (size != sizeof(buffers))
    {
        log_error("ERROR: Expected size of binaries array in clGetProgramInfo "
                  "is incorrect (should be %d, got %d) (in %s:%d)\n",
                  (int)sizeof(buffers), (int)size, __FILE__, __LINE__);
        free(binary);
        return -1;
    }

    error = clGetProgramInfo(my_newly_minted_library, CL_PROGRAM_BINARIES,
                             sizeof(buffers), &buffers, NULL);
    test_error(error, "Unable to get program binary");

    // use clCreateProgramWithBinary
    cl_program library_with_binary = clCreateProgramWithBinary(
        context, 1, &deviceID, &binarySize, (const unsigned char **)buffers,
        loadErrors, &error);
    test_error(error, "Unable to create program with binary");
    error = clGetProgramBuildInfo(
        library_with_binary, deviceID, CL_PROGRAM_BINARY_TYPE,
        sizeof(cl_program_binary_type), &program_type, NULL);
    test_error(error, "Unable to get program binary type");
    if (program_type != CL_PROGRAM_BINARY_TYPE_LIBRARY)
    {
        log_error("ERROR: Expected program type of a library loaded with "
                  "binary to be CL_PROGRAM_BINARY_TYPE_LIBRARY (in %s:%d)\n",
                  __FILE__, __LINE__);
        return -1;
    }
    program_type = -1;
    free(binary);

    error = create_single_kernel_helper_create_program(
        context, &another_program, 1, &another_simple_kernel);
    if (another_program == NULL || error != CL_SUCCESS)
    {
        log_error(
            "ERROR: Unable to create a simple test program! (%s in %s:%d)\n",
            IGetErrorString(error), __FILE__, __LINE__);
        return -1;
    }

    error = clCompileProgram(another_program, 1, &deviceID, NULL, 0, NULL, NULL,
                             NULL, NULL);
    test_error(error, "Unable to compile a simple program");

    cl_program program_and_archive[2] = { another_program,
                                          library_with_binary };
    cl_program fully_linked_program = clLinkProgram(
        context, 1, &deviceID, "", 2, program_and_archive, NULL, NULL, &error);
    test_error(error,
               "Unable to create an executable from a binary and a library");

    error = clGetProgramBuildInfo(
        fully_linked_program, deviceID, CL_PROGRAM_BINARY_TYPE,
        sizeof(cl_program_binary_type), &program_type, NULL);
    test_error(error, "Unable to get program binary type");
    if (program_type != CL_PROGRAM_BINARY_TYPE_EXECUTABLE)
    {
        log_error("ERROR: Expected program type of a newly build executable to "
                  "be CL_PROGRAM_BINARY_TYPE_EXECUTABLE (in %s:%d)\n",
                  __FILE__, __LINE__);
        return -1;
    }
    program_type = -1;

    // Get the size of the resulting binary (only one device)
    error = clGetProgramInfo(fully_linked_program, CL_PROGRAM_BINARY_SIZES,
                             sizeof(binarySize), &binarySize, NULL);
    test_error(error, "Unable to get binary size");

    // Sanity check
    if (binarySize == 0)
    {
        log_error("ERROR: Binary size of program is zero (in %s:%d)\n",
                  __FILE__, __LINE__);
        return -1;
    }

    // Create a buffer and get the actual binary
    {
        binary = (unsigned char *)malloc(sizeof(unsigned char) * binarySize);
        if (binary == NULL)
        {
            log_error("ERROR: Unable to allocate binary character array with "
                      "%lu characters! (in %s:%d)\n",
                      binarySize, __FILE__, __LINE__);
            return -1;
        }
        unsigned char *buffers[1] = { binary };
        cl_int loadErrors[1];

        // Do another sanity check here first
        size_t size;
        error = clGetProgramInfo(fully_linked_program, CL_PROGRAM_BINARIES, 0,
                                 NULL, &size);
        test_error(error, "Unable to get expected size of binaries array");
        if (size != sizeof(buffers))
        {
            log_error(
                "ERROR: Expected size of binaries array in clGetProgramInfo is "
                "incorrect (should be %d, got %d) (in %s:%d)\n",
                (int)sizeof(buffers), (int)size, __FILE__, __LINE__);
            free(binary);
            return -1;
        }

        error = clGetProgramInfo(fully_linked_program, CL_PROGRAM_BINARIES,
                                 sizeof(buffers), &buffers, NULL);
        test_error(error, "Unable to get program binary");

        // use clCreateProgramWithBinary
        fully_linked_program_with_binary = clCreateProgramWithBinary(
            context, 1, &deviceID, &binarySize, (const unsigned char **)buffers,
            loadErrors, &error);
        test_error(error, "Unable to create program with binary");

        error = clGetProgramBuildInfo(
            fully_linked_program_with_binary, deviceID, CL_PROGRAM_BINARY_TYPE,
            sizeof(cl_program_binary_type), &program_type, NULL);
        test_error(error, "Unable to get program binary type");
        if (program_type != CL_PROGRAM_BINARY_TYPE_EXECUTABLE)
        {
            log_error("ERROR: Expected program type of a program created from "
                      "a fully linked executable binary to be "
                      "CL_PROGRAM_BINARY_TYPE_EXECUTABLE (in %s:%d)\n",
                      __FILE__, __LINE__);
            return -1;
        }
        program_type = -1;
        free(binary);
    }

    error = clBuildProgram(fully_linked_program_with_binary, 1, &deviceID, NULL,
                           NULL, NULL);
    test_error(error, "Unable to build a simple program");

    cl_kernel kernel =
        clCreateKernel(fully_linked_program_with_binary, "CopyBuffer", &error);
    test_error(error, "Unable to create a simple kernel");

    error = verifyCopyBuffer(context, queue, kernel);
    if (error != CL_SUCCESS) return error;

    cl_kernel another_kernel = clCreateKernel(fully_linked_program_with_binary,
                                              "AnotherCopyBuffer", &error);
    test_error(error, "Unable to create another simple kernel");

    error = verifyCopyBuffer(context, queue, another_kernel);
    if (error != CL_SUCCESS) return error;

    /* All done! */
    error = clReleaseKernel(kernel);
    test_error(error, "Unable to release kernel object");

    error = clReleaseKernel(another_kernel);
    test_error(error, "Unable to release another kernel object");

    error = clReleaseProgram(program);
    test_error(error, "Unable to release program object");

    /* Oh, one more thing. Steve Jobs and apparently Herb Sutter. The question
     * is "Who is copying whom?" */
    error = create_single_kernel_helper_create_program(context, &program, 1,
                                                       &simple_kernel);
    if (program == NULL || error != CL_SUCCESS)
    {
        log_error(
            "ERROR: Unable to create a simple test program! (%s in %s:%d)\n",
            IGetErrorString(error), __FILE__, __LINE__);
        return -1;
    }

    error = clBuildProgram(program, 1, &deviceID, NULL, NULL, NULL);
    test_error(error, "Unable to build a simple program");
    error = clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BINARY_TYPE,
                                  sizeof(cl_program_binary_type), &program_type,
                                  NULL);
    test_error(error, "Unable to get program binary type");
    if (program_type != CL_PROGRAM_BINARY_TYPE_EXECUTABLE)
    {
        log_error(
            "ERROR: Expected program type of a program created from compiled "
            "object to be CL_PROGRAM_BINARY_TYPE_EXECUTABLE (in %s:%d)\n",
            __FILE__, __LINE__);
        return -1;
    }
    program_type = -1;

    /* All's well that ends well. William Shakespeare */
    error = clReleaseProgram(program);
    test_error(error, "Unable to release program object");

    error = clReleaseProgram(another_program);
    test_error(error, "Unable to release program object");

    error = clReleaseProgram(my_newly_minted_library);
    test_error(error, "Unable to release program object");

    error = clReleaseProgram(library_with_binary);
    test_error(error, "Unable to release program object");

    error = clReleaseProgram(fully_linked_program);
    test_error(error, "Unable to release program object");

    error = clReleaseProgram(fully_linked_program_with_binary);
    test_error(error, "Unable to release program object");

    error = clReleaseProgram(program_with_binary);
    test_error(error, "Unable to release program object");

    return 0;
}

volatile int compileNotificationSent;

void CL_CALLBACK test_notify_compile_complete(cl_program program,
                                              void *userData)
{
    if (userData == NULL || strcmp((char *)userData, "compilation") != 0)
    {
        log_error("ERROR: User data passed in to compile notify function was "
                  "not correct! (in %s:%d)\n",
                  __FILE__, __LINE__);
        compileNotificationSent = -1;
    }
    else
        compileNotificationSent = 1;
    log_info("\n   <-- program successfully compiled\n");
}

volatile int libraryCreationNotificationSent;

void CL_CALLBACK test_notify_create_library_complete(cl_program program,
                                                     void *userData)
{
    if (userData == NULL || strcmp((char *)userData, "create library") != 0)
    {
        log_error("ERROR: User data passed in to library creation notify "
                  "function was not correct! (in %s:%d)\n",
                  __FILE__, __LINE__);
        libraryCreationNotificationSent = -1;
    }
    else
        libraryCreationNotificationSent = 1;
    log_info("\n   <-- library successfully created\n");
}

volatile int linkNotificationSent;

void CL_CALLBACK test_notify_link_complete(cl_program program, void *userData)
{
    if (userData == NULL || strcmp((char *)userData, "linking") != 0)
    {
        log_error("ERROR: User data passed in to link notify function was not "
                  "correct! (in %s:%d)\n",
                  __FILE__, __LINE__);
        linkNotificationSent = -1;
    }
    else
        linkNotificationSent = 1;
    log_info("\n   <-- program successfully linked\n");
}

int test_large_compile_and_link_status_options_log(cl_context context,
                                                   cl_device_id deviceID,
                                                   cl_command_queue queue,
                                                   unsigned int numLines)
{
    int error;
    cl_program program;
    cl_program *simple_kernels;
    const char **lines;
    unsigned int i;
    char buffer[MAX_LINE_SIZE_IN_PROGRAM];
    char *compile_log;
    char *compile_options;
    char *library_log;
    char *library_options;
    char *linking_log;
    char *linking_options;
    cl_build_status status;
    size_t size_ret;

    compileNotificationSent = libraryCreationNotificationSent =
        linkNotificationSent = 0;

    simple_kernels = (cl_program *)malloc(numLines * sizeof(cl_program));
    if (simple_kernels == NULL)
    {
        log_error("ERROR: Unable to allocate kernels array with %d kernels! "
                  "(in %s:%d)\n",
                  numLines, __FILE__, __LINE__);
        return -1;
    }
    /* First, allocate the array for our line pointers */
    lines = (const char **)malloc((2 * numLines + 2) * sizeof(const char *));
    if (lines == NULL)
    {
        log_error(
            "ERROR: Unable to allocate lines array with %d lines! (in %s:%d)\n",
            (2 * numLines + 2), __FILE__, __LINE__);
        return -1;
    }

    for (i = 0; i < numLines; i++)
    {
        sprintf(buffer, composite_kernel_extern_template, i);
        lines[i] = _strdup(buffer);
    }
    /* First and last lines are easy */
    lines[numLines] = composite_kernel_start;
    lines[2 * numLines + 1] = composite_kernel_end;

    /* Fill the rest with templated kernels */
    for (i = numLines + 1; i < 2 * numLines + 1; i++)
    {
        sprintf(buffer, composite_kernel_template, i - numLines - 1);
        lines[i] = _strdup(buffer);
    }

    /* Try to create a program with these lines */
    error = create_single_kernel_helper_create_program(context, &program,
                                                       2 * numLines + 2, lines);
    if (program == NULL || error != CL_SUCCESS)
    {
        log_error("ERROR: Unable to create long test program with %d lines! "
                  "(%s) (in %s:%d)\n",
                  numLines, IGetErrorString(error), __FILE__, __LINE__);
        return -1;
    }

    /* Lets check that the compilation status is CL_BUILD_NONE */
    error = clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_STATUS,
                                  sizeof(status), &status, NULL);
    test_error(error, "Unable to get program compile status");
    if (status != CL_BUILD_NONE)
    {
        log_error("ERROR: Expected compile status to be CL_BUILD_NONE prior to "
                  "the beginning of the compilation! (status: %d in %s:%d)\n",
                  (int)status, __FILE__, __LINE__);
        return -1;
    }

    /* Compile it */
    error =
        clCompileProgram(program, 1, &deviceID, NULL, 0, NULL, NULL,
                         test_notify_compile_complete, (void *)"compilation");
    test_error(error, "Unable to compile a simple program");

    /* Wait for compile to complete (just keep polling, since we're just a test
     */
    error = clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_STATUS,
                                  sizeof(status), &status, NULL);
    test_error(error, "Unable to get program compile status");

    while ((int)status == CL_BUILD_IN_PROGRESS)
    {
        log_info("\n  -- still waiting for compile... (status is %d)", status);
        sleep(1);
        error =
            clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_STATUS,
                                  sizeof(status), &status, NULL);
        test_error(error, "Unable to get program compile status");
    }
    if (status != CL_BUILD_SUCCESS)
    {
        log_error("ERROR: compile failed! (status: %d in %s:%d)\n", (int)status,
                  __FILE__, __LINE__);
        return -1;
    }

    error = clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG, 0,
                                  NULL, &size_ret);
    test_error(error, "Device failed to return compile log size");
    compile_log = (char *)malloc(size_ret);
    error = clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG,
                                  size_ret, compile_log, NULL);
    if (error != CL_SUCCESS)
    {
        log_error("Device failed to return a compile log (in %s:%d)\n",
                  __FILE__, __LINE__);
        test_error(error, "clGetProgramBuildInfo CL_PROGRAM_BUILD_LOG failed");
    }
    log_info("BUILD LOG: %s\n", compile_log);
    free(compile_log);

    error = clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_OPTIONS,
                                  0, NULL, &size_ret);
    test_error(error, "Device failed to return compile options size");
    compile_options = (char *)malloc(size_ret);
    error = clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_OPTIONS,
                                  size_ret, compile_options, NULL);
    test_error(
        error,
        "Device failed to return compile options.\nclGetProgramBuildInfo "
        "CL_PROGRAM_BUILD_OPTIONS failed");

    log_info("BUILD OPTIONS: %s\n", compile_options);
    free(compile_options);

    /* Create and compile templated kernels */
    for (i = 0; i < numLines; i++)
    {
        sprintf(buffer, simple_kernel_template, i);
        const char *kernel_source = _strdup(buffer);
        error = create_single_kernel_helper_create_program(
            context, &simple_kernels[i], 1, &kernel_source);
        if (simple_kernels[i] == NULL || error != CL_SUCCESS)
        {
            log_error("ERROR: Unable to create long test program with %d "
                      "lines! (%s in %s:%d)",
                      numLines, IGetErrorString(error), __FILE__, __LINE__);
            return -1;
        }

        /* Compile it */
        error = clCompileProgram(simple_kernels[i], 1, &deviceID, NULL, 0, NULL,
                                 NULL, NULL, NULL);
        test_error(error, "Unable to compile a simple program");

        free((void *)kernel_source);
    }

    /* Create library out of compiled templated kernels */
    cl_program my_newly_minted_library = clLinkProgram(
        context, 1, &deviceID, "-create-library", numLines, simple_kernels,
        test_notify_create_library_complete, (void *)"create library", &error);
    test_error(error, "Unable to create a multi-line library");

    /* Wait for library creation to complete (just keep polling, since we're
     * just a test */
    error = clGetProgramBuildInfo(my_newly_minted_library, deviceID,
                                  CL_PROGRAM_BUILD_STATUS, sizeof(status),
                                  &status, NULL);
    test_error(error, "Unable to get library creation link status");

    while ((int)status == CL_BUILD_IN_PROGRESS)
    {
        log_info("\n  -- still waiting for library creation... (status is %d)",
                 status);
        sleep(1);
        error = clGetProgramBuildInfo(my_newly_minted_library, deviceID,
                                      CL_PROGRAM_BUILD_STATUS, sizeof(status),
                                      &status, NULL);
        test_error(error, "Unable to get library creation link status");
    }
    if (status != CL_BUILD_SUCCESS)
    {
        log_error("ERROR: library creation failed! (status: %d in %s:%d)\n",
                  (int)status, __FILE__, __LINE__);
        return -1;
    }
    error = clGetProgramBuildInfo(my_newly_minted_library, deviceID,
                                  CL_PROGRAM_BUILD_LOG, 0, NULL, &size_ret);
    test_error(error, "Device failed to return a library creation log size");
    library_log = (char *)malloc(size_ret);
    error = clGetProgramBuildInfo(my_newly_minted_library, deviceID,
                                  CL_PROGRAM_BUILD_LOG, size_ret, library_log,
                                  NULL);
    if (error != CL_SUCCESS)
    {
        log_error("Device failed to return a library creation log (in %s:%d)\n",
                  __FILE__, __LINE__);
        test_error(error, "clGetProgramBuildInfo CL_PROGRAM_BUILD_LOG failed");
    }
    log_info("CREATE LIBRARY LOG: %s\n", library_log);
    free(library_log);

    error = clGetProgramBuildInfo(my_newly_minted_library, deviceID,
                                  CL_PROGRAM_BUILD_OPTIONS, 0, NULL, &size_ret);
    test_error(error, "Device failed to return library creation options size");
    library_options = (char *)malloc(size_ret);
    error = clGetProgramBuildInfo(my_newly_minted_library, deviceID,
                                  CL_PROGRAM_BUILD_OPTIONS, size_ret,
                                  library_options, NULL);
    test_error(
        error,
        "Device failed to return library creation "
        "options.\nclGetProgramBuildInfo CL_PROGRAM_BUILD_OPTIONS failed");

    log_info("CREATE LIBRARY OPTIONS: %s\n", library_options);
    free(library_options);

    /* Link the program that calls the kernels and the library that contains
     * them */
    cl_program programs[2] = { program, my_newly_minted_library };
    cl_program my_newly_linked_program =
        clLinkProgram(context, 1, &deviceID, NULL, 2, programs,
                      test_notify_link_complete, (void *)"linking", &error);
    test_error(error, "Unable to link a program with a library");

    /* Wait for linking to complete (just keep polling, since we're just a test
     */
    error = clGetProgramBuildInfo(my_newly_linked_program, deviceID,
                                  CL_PROGRAM_BUILD_STATUS, sizeof(status),
                                  &status, NULL);
    test_error(error, "Unable to get program link status");

    while ((int)status == CL_BUILD_IN_PROGRESS)
    {
        log_info("\n  -- still waiting for program linking... (status is %d)",
                 status);
        sleep(1);
        error = clGetProgramBuildInfo(my_newly_linked_program, deviceID,
                                      CL_PROGRAM_BUILD_STATUS, sizeof(status),
                                      &status, NULL);
        test_error(error, "Unable to get program link status");
    }
    if (status != CL_BUILD_SUCCESS)
    {
        log_error("ERROR: program linking failed! (status: %d in %s:%d)\n",
                  (int)status, __FILE__, __LINE__);
        return -1;
    }
    error = clGetProgramBuildInfo(my_newly_linked_program, deviceID,
                                  CL_PROGRAM_BUILD_LOG, 0, NULL, &size_ret);
    test_error(error, "Device failed to return a linking log size");
    linking_log = (char *)malloc(size_ret);
    error = clGetProgramBuildInfo(my_newly_linked_program, deviceID,
                                  CL_PROGRAM_BUILD_LOG, size_ret, linking_log,
                                  NULL);
    if (error != CL_SUCCESS)
    {
        log_error("Device failed to return a linking log (in %s:%d).\n",
                  __FILE__, __LINE__);
        test_error(error, "clGetProgramBuildInfo CL_PROGRAM_BUILD_LOG failed");
    }
    log_info("BUILDING LOG: %s\n", linking_log);
    free(linking_log);

    error = clGetProgramBuildInfo(my_newly_linked_program, deviceID,
                                  CL_PROGRAM_BUILD_OPTIONS, 0, NULL, &size_ret);
    test_error(error, "Device failed to return linking options size");
    linking_options = (char *)malloc(size_ret);
    error = clGetProgramBuildInfo(my_newly_linked_program, deviceID,
                                  CL_PROGRAM_BUILD_OPTIONS, size_ret,
                                  linking_options, NULL);
    test_error(
        error,
        "Device failed to return linking options.\nclGetProgramBuildInfo "
        "CL_PROGRAM_BUILD_OPTIONS failed");

    log_info("BUILDING OPTIONS: %s\n", linking_options);
    free(linking_options);

    // Create the composite kernel
    cl_kernel kernel =
        clCreateKernel(my_newly_linked_program, "CompositeKernel", &error);
    test_error(error, "Unable to create a composite kernel");

    // Run the composite kernel and verify the results
    error = verifyCopyBuffer(context, queue, kernel);
    if (error != CL_SUCCESS) return error;

    /* All done! */
    error = clReleaseKernel(kernel);
    test_error(error, "Unable to release kernel object");

    error = clReleaseProgram(program);
    test_error(error, "Unable to release program object");

    for (i = 0; i < numLines; i++)
    {
        free((void *)lines[i]);
        free((void *)lines[i + numLines + 1]);
    }
    free(lines);

    for (i = 0; i < numLines; i++)
    {
        error = clReleaseProgram(simple_kernels[i]);
        test_error(error, "Unable to release program object");
    }
    free(simple_kernels);

    error = clReleaseProgram(my_newly_minted_library);
    test_error(error, "Unable to release program object");

    error = clReleaseProgram(my_newly_linked_program);
    test_error(error, "Unable to release program object");

    return 0;
}

int test_compile_and_link_status_options_log(cl_device_id deviceID,
                                             cl_context context,
                                             cl_command_queue queue,
                                             int num_elements)
{
    unsigned int toTest[] = { 256, 0 }; // 512, 1024, 8192, 16384, 32768, 0 };
    unsigned int i;

    log_info("Testing Compile and Link Status, Options and Logging ...this "
             "might take awhile...\n");

    for (i = 0; toTest[i] != 0; i++)
    {
        log_info("   %d...\n", toTest[i]);

#if defined(_WIN32)
        clock_t start = clock();
#elif defined(__linux__) || defined(__APPLE__)
        timeval time1, time2;
        gettimeofday(&time1, NULL);
#endif

        if (test_large_compile_and_link_status_options_log(context, deviceID,
                                                           queue, toTest[i])
            != 0)
        {
            log_error(
                "ERROR: large program compilation, linking, status, options "
                "and logging test failed for %d lines! (in %s:%d)\n",
                toTest[i], __FILE__, __LINE__);
            return -1;
        }

#if defined(_WIN32)
        clock_t end = clock();
        log_perf((float)(end - start) / (float)CLOCKS_PER_SEC, false,
                 "clock() time in secs", "%d lines", toTest[i]);
#elif defined(__linux__) || defined(__APPLE__)
        gettimeofday(&time2, NULL);
        log_perf((float)(float)(time2.tv_sec - time1.tv_sec)
                     + 1.0e-6 * (time2.tv_usec - time1.tv_usec),
                 false, "wall time in secs", "%d lines", toTest[i]);
#endif
    }

    return 0;
}
