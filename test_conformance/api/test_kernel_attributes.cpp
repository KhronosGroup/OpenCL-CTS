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
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

#include "harness/errorHelpers.h"
#include "harness/typeWrappers.h"
#include "harness/parseParameters.h"

using KernelAttributes = std::vector<std::string>;

static std::string generate_kernel_source(const KernelAttributes& attributes)
{
    std::string kernel;
    for (auto attribute : attributes)
    {
        kernel += "__attribute__((" + attribute + "))\n";
    }
    kernel += "__kernel void test_kernel(){}";
    return kernel;
}


using AttributePermutations = std::vector<KernelAttributes>;

// The following combinations have been chosen as they place each of the
// attribute types in the different orders that they can occur. While distinct
// permutations would provide a complete overview of the API the sheer number of
// combinations increases the runtime of this test by an unreasonable amount
AttributePermutations vect_tests;
AttributePermutations work_tests;
AttributePermutations reqd_tests;

AttributePermutations vect_reqd_tests;
AttributePermutations work_vect_tests;
AttributePermutations reqd_work_tests;

AttributePermutations vect_work_reqd_tests;
AttributePermutations work_reqd_vect_tests;
AttributePermutations reqd_vect_work_tests;


// Generate a vector with vec_type_hint(<data_type>) so that it can be used to
// generate different kernels
static KernelAttributes generate_vec_type_hint_data(cl_device_id device)
{
    KernelAttributes vec_type_hint_data;
    // TODO Test for signed vectors (char/short/int/etc)
    std::vector<std::string> vector_types = { "uchar", "ushort", "uint",
                                              "float" };
    if (gHasLong)
    {
        vector_types.push_back("ulong");
    }
    if (device_supports_half(device))
    {
        vector_types.push_back("half");
    }
    if (device_supports_double(device))
    {
        vector_types.push_back("double");
    }

    const auto vector_sizes = { "2", "3", "4", "8", "16" };
    for (auto type : vector_types)
    {
        for (auto size : vector_sizes)
        {
            vec_type_hint_data.push_back("vec_type_hint(" + type + size + ")");
        }
    }
    return vec_type_hint_data;
}


struct WorkGroupDimensions
{
    int x;
    int y;
    int z;
};

// Generate vectors to store reqd_work_group_size(<dimensions>) and
// work_group_size_hint(<dimensions>) so that they can be used to generate
// different kernels
static KernelAttributes generate_reqd_work_group_size_data(
    const std::vector<WorkGroupDimensions>& work_group_dimensions)
{
    KernelAttributes reqd_work_group_size_data;
    for (auto dimension : work_group_dimensions)
    {
        reqd_work_group_size_data.push_back(
            "reqd_work_group_size(" + std::to_string(dimension.x) + ","
            + std::to_string(dimension.y) + "," + std::to_string(dimension.z)
            + ")");
    }
    return reqd_work_group_size_data;
}

static KernelAttributes generate_work_group_size_data(
    const std::vector<WorkGroupDimensions>& work_group_dimensions)
{
    KernelAttributes work_group_size_hint_data;
    for (auto dimension : work_group_dimensions)
    {
        work_group_size_hint_data.push_back(
            "work_group_size_hint(" + std::to_string(dimension.x) + ","
            + std::to_string(dimension.y) + "," + std::to_string(dimension.z)
            + ")");
    }
    return work_group_size_hint_data;
}

// Populate the Global Vectors which store individual Kernel Attributes
static void populate_single_attribute_tests(
    // Vectors to store the different data that fill the attributes
    const KernelAttributes& vec_type_hint_data,
    const KernelAttributes& work_group_size_hint_data,
    const KernelAttributes& reqd_work_group_size_data)
{
    for (auto vector_test : vec_type_hint_data)
    {
        // Initialise vec_type_hint attribute tests
        vect_tests.push_back({ vector_test });
    }
    for (auto work_group_test : work_group_size_hint_data)
    {

        // Initialise work_group_size_hint attribute test
        work_tests.push_back({ work_group_test });
    }
    for (auto reqd_work_group_test : reqd_work_group_size_data)
    {

        // Initialise reqd_work_group_size attribute tests
        reqd_tests.push_back({ reqd_work_group_test });
    }
}

// Populate the Global Vectors which store the different permutations of 2
// Kernel Attributes
static void populate_double_attribute_tests(
    const KernelAttributes& vec_type_hint_data,
    const KernelAttributes& work_group_size_hint_data,
    const KernelAttributes& reqd_work_group_size_data)
{
    for (auto vector_test : vec_type_hint_data)
    {
        for (auto work_group_test : work_group_size_hint_data)
        {
            // Initialise the tests for the permutation of work_group_size_hint
            // combined with vec_type_hint
            work_vect_tests.push_back({ work_group_test, vector_test });
        }
        for (auto reqd_work_group_test : reqd_work_group_size_data)
        {
            // Initialise the tests for the permutation of vec_type_hint and
            // reqd_work_group_size
            vect_reqd_tests.push_back({ vector_test, reqd_work_group_test });
        }
    }
    for (auto work_group_test : work_group_size_hint_data)
    {

        for (auto reqd_work_group_test : reqd_work_group_size_data)
        {
            // Initialse the tests for the permutation of reqd_work_group_size
            // and  work_group_size_hint
            reqd_work_tests.push_back(
                { reqd_work_group_test, work_group_test });
        }
    }
}

// Populate the Global Vectors which store the different permutations of 3
// Kernel Attributes
static void populate_triple_attribute_tests(
    const KernelAttributes& vec_type_hint_data,
    const KernelAttributes& work_group_size_hint_data,
    const KernelAttributes& reqd_work_group_size_data)
{
    for (auto vector_test : vec_type_hint_data)
    {
        for (auto work_group_test : work_group_size_hint_data)
        {
            for (auto reqd_work_group_test : reqd_work_group_size_data)
            {
                //  Initialise the chosen permutations of 3 attributes
                vect_work_reqd_tests.push_back(
                    { vector_test, work_group_test, reqd_work_group_test });
                work_reqd_vect_tests.push_back(
                    { work_group_test, reqd_work_group_test, vector_test });
                reqd_vect_work_tests.push_back(
                    { reqd_work_group_test, vector_test, work_group_test });
            }
        }
    }
}

static const std::vector<AttributePermutations*>
generate_attribute_tests(const KernelAttributes& vec_type_hint_data,
                         const KernelAttributes& work_group_size_hint_data,
                         const KernelAttributes& reqd_work_group_size_data)
{
    populate_single_attribute_tests(vec_type_hint_data,
                                    work_group_size_hint_data,
                                    reqd_work_group_size_data);
    populate_double_attribute_tests(vec_type_hint_data,
                                    work_group_size_hint_data,
                                    reqd_work_group_size_data);
    populate_triple_attribute_tests(vec_type_hint_data,
                                    work_group_size_hint_data,
                                    reqd_work_group_size_data);

    // Store all of the filled vectors in a single structure
    const std::vector<AttributePermutations*> all_tests = {
        &vect_tests,           &work_tests,           &reqd_tests,

        &work_vect_tests,      &vect_reqd_tests,      &reqd_work_tests,

        &vect_work_reqd_tests, &work_reqd_vect_tests, &reqd_vect_work_tests
    };
    return all_tests;
}

static const std::vector<AttributePermutations*>
initialise_attribute_data(cl_device_id device)
{
    // This vector stores different work group dimensions that can be used by
    // the reqd_work_group_size and work_group_size_hint attributes. It
    // currently only has a single value to minimise time complexity of the
    // overall test but can be easily changed.
    static const std::vector<WorkGroupDimensions> work_group_dimensions = {
        { 1, 1, 1 }
    };
    KernelAttributes vec_type_hint_data = generate_vec_type_hint_data(device);
    KernelAttributes work_group_size_hint_data =
        generate_work_group_size_data(work_group_dimensions);
    KernelAttributes reqd_work_group_size_data =
        generate_reqd_work_group_size_data(work_group_dimensions);

    // Generate all the permutations of attributes to create different test
    // suites
    return generate_attribute_tests(vec_type_hint_data,
                                    work_group_size_hint_data,
                                    reqd_work_group_size_data);
}

static bool run_test(cl_context context, cl_device_id device,
                     const AttributePermutations& permutations)
{
    bool success = true;
    for (auto attribute_permutation : permutations)
    {

        std::string kernel_source_string =
            generate_kernel_source(attribute_permutation);
        const char* kernel_src = kernel_source_string.c_str();
        clProgramWrapper program;
        clKernelWrapper kernel;
        cl_int err = create_single_kernel_helper(context, &program, &kernel, 1,
                                                 &kernel_src, "test_kernel");
        test_error_ret(err, "create_single_kernel_helper", false);

        // Get the size of the kernel attribute string returned
        size_t size = 0;
        err = clGetKernelInfo(kernel, CL_KERNEL_ATTRIBUTES, 0, nullptr, &size);
        test_error_ret(err, "clGetKernelInfo", false);
        std::vector<char> attributes(size);
        err = clGetKernelInfo(kernel, CL_KERNEL_ATTRIBUTES, attributes.size(),
                              attributes.data(), nullptr);
        test_error_ret(err, "clGetKernelInfo", false);
        std::string attribute_string(attributes.data());
        attribute_string.erase(
            std::remove(attribute_string.begin(), attribute_string.end(), ' '),
            attribute_string.end());
        if (gCompilationMode != kOnline)
        {
            if (!attribute_string.empty())
            {
                success = false;
                log_error("Error: Expected an empty string\n");
                log_error("Attribute string reported as: %s\n",
                          attribute_string.c_str());
            }
        }
        else
        {
            bool permutation_success = true;
            for (auto attribute : attribute_permutation)
            {
                if (attribute_string.find(attribute) == std::string::npos)
                {
                    success = false;
                    permutation_success = false;
                    log_error("ERROR: did not find expected attribute: '%s'\n",
                              attribute.c_str());
                }
            }
            if (!permutation_success)
            {
                log_error("Attribute string reported as: %s\n",
                          attribute_string.c_str());
            }
        }
    }
    return success;
}

REGISTER_TEST(kernel_attributes)
{
    bool success = true;

    // Vector to store all of the tests
    const std::vector<AttributePermutations*> all_tests =
        initialise_attribute_data(device);

    for (auto permutations : all_tests)
    {
        success = success && run_test(context, device, *permutations);
    }
    return success ? TEST_PASS : TEST_FAIL;
}
