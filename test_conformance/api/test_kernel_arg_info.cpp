//
// Copyright (c) 2021 The Khronos Group Inc.
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
#include "testBase.h"
#include "harness/errorHelpers.h"
#include "harness/typeWrappers.h"
#include "harness/kernelHelpers.h"

#define MINIMUM_OPENCL_PIPE_VERSION Version(2, 0)

static constexpr size_t CL_VERSION_LENGTH = 128;
static constexpr size_t KERNEL_ARGUMENT_LENGTH = 128;
static constexpr char KERNEL_ARGUMENT_NAME[] = "argument";
static constexpr size_t KERNEL_ARGUMENT_NAME_LENGTH =
    sizeof(KERNEL_ARGUMENT_NAME) + 1;
static constexpr int SINGLE_KERNEL_ARG_NUMBER = 0;
static constexpr int MAX_NUMBER_OF_KERNEL_ARGS = 128;

static const std::vector<cl_kernel_arg_address_qualifier> address_qualifiers = {
    CL_KERNEL_ARG_ADDRESS_GLOBAL, CL_KERNEL_ARG_ADDRESS_LOCAL,
    CL_KERNEL_ARG_ADDRESS_CONSTANT, CL_KERNEL_ARG_ADDRESS_PRIVATE
};

static const std::vector<std::string> image_arguments = {
    "image2d_t", "image3d_t",        "image2d_array_t",
    "image1d_t", "image1d_buffer_t", "image1d_array_t"
};

static const std::vector<cl_kernel_arg_access_qualifier> access_qualifiers = {
    CL_KERNEL_ARG_ACCESS_READ_WRITE, CL_KERNEL_ARG_ACCESS_READ_ONLY,
    CL_KERNEL_ARG_ACCESS_WRITE_ONLY
};

static const std::vector<cl_kernel_arg_type_qualifier> type_qualifiers = {
    CL_KERNEL_ARG_TYPE_NONE,
    CL_KERNEL_ARG_TYPE_CONST,
    CL_KERNEL_ARG_TYPE_VOLATILE,
    CL_KERNEL_ARG_TYPE_RESTRICT,
    CL_KERNEL_ARG_TYPE_CONST | CL_KERNEL_ARG_TYPE_VOLATILE,
    CL_KERNEL_ARG_TYPE_CONST | CL_KERNEL_ARG_TYPE_RESTRICT,
    CL_KERNEL_ARG_TYPE_VOLATILE | CL_KERNEL_ARG_TYPE_RESTRICT,
    CL_KERNEL_ARG_TYPE_CONST | CL_KERNEL_ARG_TYPE_VOLATILE
        | CL_KERNEL_ARG_TYPE_RESTRICT,
};

static const std::vector<cl_kernel_arg_type_qualifier> pipe_qualifiers = {
    CL_KERNEL_ARG_TYPE_PIPE,
    CL_KERNEL_ARG_TYPE_CONST | CL_KERNEL_ARG_TYPE_PIPE,
    CL_KERNEL_ARG_TYPE_VOLATILE | CL_KERNEL_ARG_TYPE_PIPE,
    CL_KERNEL_ARG_TYPE_CONST | CL_KERNEL_ARG_TYPE_VOLATILE
        | CL_KERNEL_ARG_TYPE_PIPE,
};

static std::string
get_address_qualifier(cl_kernel_arg_address_qualifier address_qualifier)
{
    std::string ret;
    if (address_qualifier == CL_KERNEL_ARG_ADDRESS_GLOBAL)
        ret = "global";
    else if (address_qualifier == CL_KERNEL_ARG_ADDRESS_CONSTANT)
        ret = "constant";
    else if (address_qualifier == CL_KERNEL_ARG_ADDRESS_LOCAL)
        ret = "local";
    else if (address_qualifier == CL_KERNEL_ARG_ADDRESS_PRIVATE)
        ret = "private";
    return ret;
}

static std::string
get_access_qualifier(cl_kernel_arg_access_qualifier qualifier)
{
    std::string ret;
    if (qualifier == CL_KERNEL_ARG_ACCESS_READ_ONLY) ret = "read_only";
    if (qualifier == CL_KERNEL_ARG_ACCESS_WRITE_ONLY) ret = "write_only";
    if (qualifier == CL_KERNEL_ARG_ACCESS_READ_WRITE) ret = "read_write";
    return ret;
}

static std::string
get_type_qualifier_prefix(cl_kernel_arg_type_qualifier type_qualifier)
{
    std::string ret;
    if (type_qualifier & CL_KERNEL_ARG_TYPE_CONST) ret += "const ";
    if (type_qualifier & CL_KERNEL_ARG_TYPE_VOLATILE) ret += "volatile ";
    if (type_qualifier & CL_KERNEL_ARG_TYPE_PIPE) ret += "pipe ";
    return ret;
}

static std::string
get_type_qualifier_postfix(cl_kernel_arg_type_qualifier type_qualifier)
{
    std::string ret;
    if (type_qualifier & CL_KERNEL_ARG_TYPE_RESTRICT) ret = "restrict";
    return ret;
}

class KernelArgInfo {
public:
    KernelArgInfo(cl_kernel_arg_address_qualifier input_address_qualifier,
                  cl_kernel_arg_access_qualifier input_access_qualifier,
                  cl_kernel_arg_type_qualifier input_type_qualifier,
                  const std::string& input_arg_type, const int argument_number,
                  const std::string& input_arg_string = "")
        : address_qualifier(input_address_qualifier),
          access_qualifier(input_access_qualifier),
          type_qualifier(input_type_qualifier), arg_string(input_arg_string)
    {
        strcpy(arg_type, input_arg_type.c_str());
        std::string input_arg_name =
            KERNEL_ARGUMENT_NAME + std::to_string(argument_number);
        strcpy(arg_name, input_arg_name.c_str());
    };
    KernelArgInfo() = default;
    cl_kernel_arg_address_qualifier address_qualifier;
    cl_kernel_arg_access_qualifier access_qualifier;
    cl_kernel_arg_type_qualifier type_qualifier;
    char arg_type[KERNEL_ARGUMENT_LENGTH];
    char arg_name[KERNEL_ARGUMENT_LENGTH];
    std::string arg_string;
};

static std::string generate_argument(const KernelArgInfo& kernel_arg)
{
    std::string ret;

    const bool is_image = strstr(kernel_arg.arg_type, "image")
        || strstr(kernel_arg.arg_type, "sampler");
    std::string address_qualifier = "";
    // Image Objects are always allocated from the global address space so the
    // qualifier should not be specified
    if (!is_image)
    {
        address_qualifier = get_address_qualifier(kernel_arg.address_qualifier);
    }

    std::string access_qualifier =
        get_access_qualifier(kernel_arg.access_qualifier);
    std::string type_qualifier_prefix =
        get_type_qualifier_prefix(kernel_arg.type_qualifier);
    std::string type_qualifier_postfix =
        get_type_qualifier_postfix(kernel_arg.type_qualifier);

    ret += address_qualifier + " ";
    ret += access_qualifier + " ";
    ret += type_qualifier_prefix + " ";
    ret += kernel_arg.arg_type;
    ret += " ";
    ret += type_qualifier_postfix + " ";
    ret += kernel_arg.arg_name;
    return ret;
}

/* This function generates a kernel source and allows for multiple arguments to
 * be passed in and subsequently queried. */
static std::string generate_kernel(const std::vector<KernelArgInfo>& all_args,
                                   const bool supports_3d_image_writes = false,
                                   const bool kernel_uses_half_type = false)
{

    std::string ret;
    if (supports_3d_image_writes)
    {
        ret += "#pragma OPENCL EXTENSION cl_khr_3d_image_writes: enable\n";
    }
    if (kernel_uses_half_type)
    {
        ret += "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
    }
    ret += "kernel void get_kernel_arg_info(\n";
    for (int i = 0; i < all_args.size(); ++i)
    {
        const KernelArgInfo& arg = all_args[i];
        ret += generate_argument(all_args[i]);
        if (i == all_args.size() - 1)
        {
            ret += "\n";
        }
        else
        {
            ret += ",\n";
        }
    }
    ret += "){}";
    return ret;
}

static const char* get_kernel_arg_address_qualifier(
    cl_kernel_arg_address_qualifier address_qualifier)
{
    switch (address_qualifier)
    {
        case CL_KERNEL_ARG_ADDRESS_GLOBAL: {
            return "GLOBAL";
        }
        case CL_KERNEL_ARG_ADDRESS_LOCAL: {
            return "LOCAL";
        }
        case CL_KERNEL_ARG_ADDRESS_CONSTANT: {
            return "CONSTANT";
        }
        default: {
            return "PRIVATE";
        }
    }
}

static const char*
get_kernel_arg_access_qualifier(cl_kernel_arg_access_qualifier access_qualifier)
{
    switch (access_qualifier)
    {
        case CL_KERNEL_ARG_ACCESS_READ_ONLY: {
            return "READ_ONLY";
        }
        case CL_KERNEL_ARG_ACCESS_WRITE_ONLY: {
            return "WRITE_ONLY";
        }
        case CL_KERNEL_ARG_ACCESS_READ_WRITE: {
            return "READ_WRITE";
        }
        default: {
            return "NONE";
        }
    }
}

std::string
get_kernel_arg_type_qualifier(cl_kernel_arg_type_qualifier type_qualifier)
{
    std::string ret;

    if (type_qualifier & CL_KERNEL_ARG_TYPE_CONST) ret += "CONST ";
    if (type_qualifier & CL_KERNEL_ARG_TYPE_RESTRICT) ret += "RESTRICT ";
    if (type_qualifier & CL_KERNEL_ARG_TYPE_VOLATILE) ret += "VOLATILE ";
    if (type_qualifier & CL_KERNEL_ARG_TYPE_PIPE) ret += "PIPE";

    return ret;
}

static void output_difference(const KernelArgInfo& expected,
                              const KernelArgInfo& actual)
{
    if (actual.address_qualifier != expected.address_qualifier)
    {
        log_error("Address Qualifier: Expected: %s\t Actual: %s\n",
                  get_kernel_arg_address_qualifier(expected.address_qualifier),
                  get_kernel_arg_address_qualifier(actual.address_qualifier));
    }
    if (actual.access_qualifier != expected.access_qualifier)
    {
        log_error("Access Qualifier: Expected: %s\t Actual: %s\n",
                  get_kernel_arg_access_qualifier(expected.access_qualifier),
                  get_kernel_arg_access_qualifier(actual.access_qualifier));
    }
    if (actual.type_qualifier != expected.type_qualifier)
    {
        log_error(
            "Type Qualifier: Expected: %s\t Actual: %s\n",
            get_kernel_arg_type_qualifier(expected.type_qualifier).c_str(),
            get_kernel_arg_type_qualifier(actual.type_qualifier).c_str());
    }
    if (strcmp(actual.arg_type, expected.arg_type) != 0)
    {
        log_error("Arg Type: Expected: %s\t Actual: %s\n", expected.arg_type,
                  actual.arg_type);
    }
    if (strcmp(actual.arg_name, expected.arg_name) != 0)
    {
        log_error("Arg Name: Expected: %s\t Actual: %s\n", expected.arg_name,
                  actual.arg_name);
    }
    log_error("Argument in Kernel Source Reported as:\n%s\n",
              expected.arg_string.c_str());
}
static int compare_expected_actual(const KernelArgInfo& expected,
                                   const KernelArgInfo& actual)
{
    ++gTestCount;
    int ret = TEST_PASS;
    if ((actual.address_qualifier != expected.address_qualifier)
        || (actual.access_qualifier != expected.access_qualifier)
        || (actual.type_qualifier != expected.type_qualifier)
        || (strcmp(actual.arg_type, expected.arg_type) != 0)
        || (strcmp(actual.arg_name, expected.arg_name) != 0))
    {
        ret = TEST_FAIL;
        output_difference(expected, actual);
        ++gFailCount;
    }
    return ret;
}

static bool device_supports_pipes(cl_device_id deviceID)
{
    auto version = get_device_cl_version(deviceID);
    if (version < MINIMUM_OPENCL_PIPE_VERSION)
    {
        return false;
    }
    cl_uint max_packet_size = 0;
    cl_int err =
        clGetDeviceInfo(deviceID, CL_DEVICE_PIPE_MAX_PACKET_SIZE,
                        sizeof(max_packet_size), &max_packet_size, nullptr);
    test_error_ret(err, "clGetDeviceInfo", false);
    if ((max_packet_size == 0) && (version >= Version(3, 0)))
    {
        return false;
    }
    return true;
}

static std::string get_build_options(cl_device_id deviceID)
{
    std::string ret = "-cl-kernel-arg-info";
    if (get_device_cl_version(deviceID) >= MINIMUM_OPENCL_PIPE_VERSION)
    {
        if (device_supports_pipes(deviceID))
        {
            if (get_device_cl_version(deviceID) >= Version(3, 0))
            {
                ret += " -cl-std=CL3.0";
            }
            else
            {
                ret += " -cl-std=CL2.0";
            }
        }
    }
    return ret;
}

static std::string get_expected_arg_type(const std::string& type_string,
                                         const bool is_pointer)
{
    bool is_unsigned = false;
    std::istringstream type_stream(type_string);
    std::string base_type = "";
    std::string ret = "";
    /* Signed and Unsigned on their own represent an int */
    if (type_string == "signed" || type_string == "signed*")
    {
        base_type = "int";
    }
    else if (type_string == "unsigned" || type_string == "unsigned*")
    {
        base_type = "int";
        is_unsigned = true;
    }
    else
    {
        std::string token;
        /* Iterate through the argument type to determine what the type is and
         * whether or not it is signed */
        while (std::getline(type_stream, token, ' '))
        {
            if (token.find("unsigned") != std::string::npos)
            {
                is_unsigned = true;
            }
            if (token.find("signed") == std::string::npos)
            {
                base_type = token;
            }
        }
    }
    ret = base_type;
    if (is_unsigned)
    {
        ret.insert(0, "u");
    }
    /* Ensure that the data type is a pointer if it is not already when
     * necessary */
    if (is_pointer && ret.back() != '*')
    {
        ret += "*";
    }
    return ret;
}

static KernelArgInfo
create_expected_arg_info(const KernelArgInfo& kernel_argument, bool is_pointer)
{
    KernelArgInfo ret = kernel_argument;
    const std::string arg_string = generate_argument(kernel_argument);
    ret.arg_string = arg_string;

    std::string type_string(kernel_argument.arg_type);
    /* We only need to modify the expected return values for scalar types */
    if ((is_pointer && !isdigit(type_string.back() - 1))
        || !isdigit(type_string.back()))
    {
        std::string expected_arg_type =
            get_expected_arg_type(type_string, is_pointer);

        /* Reset the Contents of expected arg_type char[] and then assign it to
         * the expected value */
        memset(ret.arg_type, 0, sizeof(ret.arg_type));
        strcpy(ret.arg_type, expected_arg_type.c_str());
    }

    /* Any values passed by reference has TYPE_NONE */
    if (!is_pointer)
    {
        ret.type_qualifier = CL_KERNEL_ARG_TYPE_NONE;
    }

    /* If the address qualifier is CONSTANT we expect to see the TYPE_CONST
     * qualifier*/
    if (kernel_argument.address_qualifier == CL_KERNEL_ARG_ADDRESS_CONSTANT)
    {
        ret.type_qualifier |= CL_KERNEL_ARG_TYPE_CONST;
    }

    /* The PIPE qualifier is special. It can only be used in a global scope. It
     * also ignores any other qualifiers */
    if (kernel_argument.type_qualifier & CL_KERNEL_ARG_TYPE_PIPE)
    {
        ret.address_qualifier = CL_KERNEL_ARG_ADDRESS_GLOBAL;
        ret.type_qualifier = CL_KERNEL_ARG_TYPE_PIPE;
    }

    return ret;
}

/* There are too many vector arguments for it to be worth writing down
 * statically and are instead generated here and combined with all of the scalar
 * and unsigned scalar types in a single data structure */
static std::vector<std::string>
generate_all_type_arguments(cl_device_id deviceID)
{
    std::vector<std::string> ret = {
        "char",           "short",        "int",           "float",
        "void",           "uchar",        "unsigned char", "ushort",
        "unsigned short", "uint",         "unsigned int",  "char unsigned",
        "short unsigned", "int unsigned", "signed short",  "signed int",
        "signed long",    "short signed", "int signed",    "signed",
        "unsigned"
    };

    std::vector<std::string> vector_types = { "char",   "uchar", "short",
                                              "ushort", "int",   "uint",
                                              "float" };
    if (gHasLong)
    {
        ret.push_back("long");
        ret.push_back("ulong");
        ret.push_back("unsigned long");
        ret.push_back("long unsigned");
        ret.push_back("long signed");
        vector_types.push_back("long");
        vector_types.push_back("ulong");
    }
    if (device_supports_half(deviceID))
    {
        vector_types.push_back("half");
    }
    if (device_supports_double(deviceID))
    {
        vector_types.push_back("double");
    }
    static const std::vector<std::string> vector_values = { "2", "3", "4", "8",
                                                            "16" };
    for (auto vector_type : vector_types)
    {
        for (auto vector_value : vector_values)
        {
            ret.push_back(vector_type + vector_value);
        }
    }
    return ret;
}

static int
compare_kernel_with_expected(cl_context context, cl_device_id deviceID,
                             const char* kernel_src,
                             const std::vector<KernelArgInfo>& expected_args)
{
    int failed_tests = 0;
    clKernelWrapper kernel;
    clProgramWrapper program;
    cl_int err = create_single_kernel_helper_with_build_options(
        context, &program, &kernel, 1, &kernel_src, "get_kernel_arg_info",
        get_build_options(deviceID).c_str());
    test_error(err, "create_single_kernel_helper_with_build_options");
    for (int i = 0; i < expected_args.size(); ++i)
    {
        KernelArgInfo actual;
        err = clGetKernelArgInfo(kernel, i, CL_KERNEL_ARG_ADDRESS_QUALIFIER,
                                 sizeof(actual.address_qualifier),
                                 &(actual.address_qualifier), nullptr);
        test_error(err, "clGetKernelArgInfo");

        err = clGetKernelArgInfo(kernel, i, CL_KERNEL_ARG_ACCESS_QUALIFIER,
                                 sizeof(actual.access_qualifier),
                                 &(actual.access_qualifier), nullptr);
        test_error(err, "clGetKernelArgInfo");

        err = clGetKernelArgInfo(kernel, i, CL_KERNEL_ARG_TYPE_QUALIFIER,
                                 sizeof(actual.type_qualifier),
                                 &(actual.type_qualifier), nullptr);
        test_error(err, "clGetKernelArgInfo");

        err = clGetKernelArgInfo(kernel, i, CL_KERNEL_ARG_TYPE_NAME,
                                 sizeof(actual.arg_type), &(actual.arg_type),
                                 nullptr);
        test_error(err, "clGetKernelArgInfo");

        err = clGetKernelArgInfo(kernel, i, CL_KERNEL_ARG_NAME,
                                 sizeof(actual.arg_name), &(actual.arg_name),
                                 nullptr);
        test_error(err, "clGetKernelArgInfo");

        failed_tests += compare_expected_actual(expected_args[i], actual);
    }
    return failed_tests;
}

size_t get_param_size(const std::string& arg_type, cl_device_id deviceID,
                      bool is_pipe)
{
    if (is_pipe)
    {
        return (sizeof(int*));
    }
    if (arg_type.find("*") != std::string::npos)
    {
        cl_uint device_address_bits = 0;
        cl_int err = clGetDeviceInfo(deviceID, CL_DEVICE_ADDRESS_BITS,
                                     sizeof(device_address_bits),
                                     &device_address_bits, NULL);
        return (device_address_bits / 8);
    }

    size_t ret(0);
    if (arg_type.find("char") != std::string::npos)
    {
        ret += sizeof(cl_char);
    }
    if (arg_type.find("short") != std::string::npos)
    {
        ret += sizeof(cl_short);
    }
    if (arg_type.find("half") != std::string::npos)
    {
        ret += sizeof(cl_half);
    }
    if (arg_type.find("int") != std::string::npos)
    {
        ret += sizeof(cl_int);
    }
    if (arg_type.find("long") != std::string::npos)
    {
        ret += sizeof(cl_long);
    }
    if (arg_type.find("float") != std::string::npos)
    {
        ret += sizeof(cl_float);
    }
    if (arg_type.find("double") != std::string::npos)
    {
        ret += sizeof(cl_double);
    }
    if (arg_type.back() == '2')
    {
        ret *= 2;
    }
    if (arg_type.back() == '3')
    {
        ret *= 4;
    }
    if (arg_type.back() == '4')
    {
        ret *= 4;
    }
    if (arg_type.back() == '8')
    {
        ret *= 8;
    }
    // If the last character is a 6 it represents a vector of 16
    if (arg_type.back() == '6')
    {
        ret *= 16;
    }
    return ret;
}

static int run_scalar_vector_tests(cl_context context, cl_device_id deviceID)
{
    int failed_tests = 0;

    std::vector<std::string> type_arguments =
        generate_all_type_arguments(deviceID);

    const std::vector<cl_kernel_arg_access_qualifier> access_qualifiers = {
        CL_KERNEL_ARG_ACCESS_NONE, CL_KERNEL_ARG_ACCESS_READ_ONLY,
        CL_KERNEL_ARG_ACCESS_WRITE_ONLY
    };

    std::vector<KernelArgInfo> all_args, expected_args;
    size_t max_param_size = get_max_param_size(deviceID);
    size_t total_param_size(0);
    for (auto address_qualifier : address_qualifiers)
    {
        bool is_private = (address_qualifier == CL_KERNEL_ARG_ADDRESS_PRIVATE);

        /* OpenCL kernels cannot take "private" pointers and only "private"
         * variables can take values */
        bool is_pointer = !is_private;

        for (auto type_qualifier : type_qualifiers)
        {
            bool is_pipe = (type_qualifier & CL_KERNEL_ARG_TYPE_PIPE);
            bool is_restrict = (type_qualifier & CL_KERNEL_ARG_TYPE_RESTRICT);

            for (auto access_qualifier : access_qualifiers)
            {
                bool has_access_qualifier =
                    (access_qualifier != CL_KERNEL_ARG_ACCESS_NONE);

                /*Only images and pipes can have an access qualifier,
                 * otherwise it should be ACCESS_NONE */
                if (!is_pipe && has_access_qualifier)
                {
                    continue;
                }

                /* If the type is a pipe, then either the specified or
                 * default access qualifier is returned and so "NONE" will
                 * never be returned */
                if (is_pipe && !has_access_qualifier)
                {
                    continue;
                }

                /* The "restrict" type qualifier can only apply to
                 * pointers
                 */
                if (is_restrict && !is_pointer)
                {
                    continue;
                }

                /* We cannot have pipe pointers */
                if (is_pipe && is_pointer)
                {
                    continue;
                }


                for (auto arg_type : type_arguments)
                {
                    /* Void Types cannot be private */
                    if (is_private && arg_type == "void")
                    {
                        continue;
                    }

                    if (is_pointer)
                    {
                        arg_type += "*";
                    }
                    size_t param_size =
                        get_param_size(arg_type, deviceID, is_pipe);
                    if (param_size + total_param_size >= max_param_size
                        || all_args.size() == MAX_NUMBER_OF_KERNEL_ARGS)
                    {
                        const std::string kernel_src = generate_kernel(
                            all_args, false, device_supports_half(deviceID));
                        failed_tests += compare_kernel_with_expected(
                            context, deviceID, kernel_src.c_str(),
                            expected_args);
                        all_args.clear();
                        expected_args.clear();
                        total_param_size = 0;
                    }
                    total_param_size += param_size;

                    KernelArgInfo kernel_argument(
                        address_qualifier, access_qualifier, type_qualifier,
                        arg_type, all_args.size());

                    expected_args.push_back(
                        create_expected_arg_info(kernel_argument, is_pointer));

                    all_args.push_back(kernel_argument);
                }
            }
        }
    }
    const std::string kernel_src =
        generate_kernel(all_args, false, device_supports_half(deviceID));
    failed_tests += compare_kernel_with_expected(
        context, deviceID, kernel_src.c_str(), expected_args);
    return failed_tests;
}

static cl_uint get_max_number_of_pipes(cl_device_id deviceID, cl_int& err)
{
    cl_uint ret(0);
    err = clGetDeviceInfo(deviceID, CL_DEVICE_MAX_PIPE_ARGS, sizeof(ret), &ret,
                          nullptr);
    return ret;
}

static int run_pipe_tests(cl_context context, cl_device_id deviceID)
{
    int failed_tests = 0;

    cl_kernel_arg_address_qualifier address_qualifier =
        CL_KERNEL_ARG_ADDRESS_PRIVATE;
    std::vector<std::string> type_arguments =
        generate_all_type_arguments(deviceID);
    const std::vector<cl_kernel_arg_access_qualifier> access_qualifiers = {
        CL_KERNEL_ARG_ACCESS_READ_ONLY, CL_KERNEL_ARG_ACCESS_WRITE_ONLY
    };
    std::vector<KernelArgInfo> all_args, expected_args;
    size_t max_param_size = get_max_param_size(deviceID);
    size_t total_param_size(0);
    cl_int err = CL_SUCCESS;
    cl_uint max_number_of_pipes = get_max_number_of_pipes(deviceID, err);
    test_error_ret(err, "get_max_number_of_pipes", TEST_FAIL);
    cl_uint number_of_pipes(0);

    const bool is_pointer = false;
    const bool is_pipe = true;

    for (auto type_qualifier : pipe_qualifiers)
    {
        for (auto access_qualifier : access_qualifiers)
        {
            for (auto arg_type : type_arguments)
            {
                /* We cannot have void pipes */
                if (arg_type == "void")
                {
                    continue;
                }

                size_t param_size = get_param_size(arg_type, deviceID, is_pipe);
                if (param_size + total_param_size >= max_param_size
                    || number_of_pipes == max_number_of_pipes)
                {
                    const std::string kernel_src = generate_kernel(all_args);
                    failed_tests += compare_kernel_with_expected(
                        context, deviceID, kernel_src.c_str(), expected_args);
                    all_args.clear();
                    expected_args.clear();
                    total_param_size = 0;
                    number_of_pipes = 0;
                }
                total_param_size += param_size;
                number_of_pipes++;

                KernelArgInfo kernel_argument(address_qualifier,
                                              access_qualifier, type_qualifier,
                                              arg_type, all_args.size());

                expected_args.push_back(
                    create_expected_arg_info(kernel_argument, is_pointer));

                all_args.push_back(kernel_argument);
            }
        }
    }
    const std::string kernel_src = generate_kernel(all_args);
    failed_tests += compare_kernel_with_expected(
        context, deviceID, kernel_src.c_str(), expected_args);
    return failed_tests;
}

static int run_sampler_test(cl_context context, cl_device_id deviceID)
{
    cl_kernel_arg_address_qualifier address_qualifier =
        CL_KERNEL_ARG_ADDRESS_PRIVATE;
    cl_kernel_arg_type_qualifier type_qualifier = CL_KERNEL_ARG_TYPE_NONE;
    cl_kernel_arg_access_qualifier access_qualifier = CL_KERNEL_ARG_ACCESS_NONE;
    std::string image_type = "sampler_t";
    bool is_pointer = false;

    KernelArgInfo kernel_argument(address_qualifier, access_qualifier,
                                  type_qualifier, image_type,
                                  SINGLE_KERNEL_ARG_NUMBER);

    KernelArgInfo expected =
        create_expected_arg_info(kernel_argument, is_pointer);

    const std::string kernel_src = generate_kernel({ kernel_argument });

    return compare_kernel_with_expected(context, deviceID, kernel_src.c_str(),
                                        { expected });
}

static int run_image_tests(cl_context context, cl_device_id deviceID)
{
    int failed_tests = 0;
    bool supports_3d_image_writes =
        is_extension_available(deviceID, "cl_khr_3d_image_writes");
    bool is_pointer = false;
    cl_kernel_arg_type_qualifier type_qualifier = CL_KERNEL_ARG_TYPE_NONE;
    cl_kernel_arg_address_qualifier address_qualifier =
        CL_KERNEL_ARG_ADDRESS_GLOBAL;

    for (auto access_qualifier : access_qualifiers)
    {
        bool is_write =
            (access_qualifier == CL_KERNEL_ARG_ACCESS_WRITE_ONLY
             || access_qualifier == CL_KERNEL_ARG_ACCESS_READ_WRITE);
        for (auto image_type : image_arguments)
        {
            bool is_3d_image = image_type == "image3d_t";
            /* We can only test 3d image writes if our device supports it */
            if (is_3d_image && is_write)
            {
                if (!supports_3d_image_writes)
                {
                    continue;
                }
            }
            KernelArgInfo kernel_argument(address_qualifier, access_qualifier,
                                          type_qualifier, image_type,
                                          SINGLE_KERNEL_ARG_NUMBER);
            KernelArgInfo expected =
                create_expected_arg_info(kernel_argument, is_pointer);
            const std::string kernel_src =
                generate_kernel({ kernel_argument }, supports_3d_image_writes);

            failed_tests += compare_kernel_with_expected(
                context, deviceID, kernel_src.c_str(), { expected });
        }
    }
    failed_tests += run_sampler_test(context, deviceID);
    return failed_tests;
}

/* Ensure clGetKernelArgInfo returns successfully when param_value is
 * set to null */
static int test_null_param(cl_context context, cl_device_id deviceID,
                           char const* kernel_src)
{
    clProgramWrapper program;
    clKernelWrapper kernel;
    cl_int err = create_single_kernel_helper_with_build_options(
        context, &program, &kernel, 1, &kernel_src, "get_kernel_arg_info",
        get_build_options(deviceID).c_str());
    test_error_ret(err, "create_single_kernel_helper_with_build_options",
                   TEST_FAIL);

    err = clGetKernelArgInfo(kernel, SINGLE_KERNEL_ARG_NUMBER,
                             CL_KERNEL_ARG_ADDRESS_QUALIFIER, 0, nullptr,
                             nullptr);
    test_error_ret(err, "clGetKernelArgInfo", TEST_FAIL);

    err =
        clGetKernelArgInfo(kernel, SINGLE_KERNEL_ARG_NUMBER,
                           CL_KERNEL_ARG_ACCESS_QUALIFIER, 0, nullptr, nullptr);
    test_error_ret(err, "clGetKernelArgInfo", TEST_FAIL);

    err = clGetKernelArgInfo(kernel, SINGLE_KERNEL_ARG_NUMBER,
                             CL_KERNEL_ARG_TYPE_QUALIFIER, 0, nullptr, nullptr);
    test_error_ret(err, "clGetKernelArgInfo", TEST_FAIL);

    err = clGetKernelArgInfo(kernel, SINGLE_KERNEL_ARG_NUMBER,
                             CL_KERNEL_ARG_TYPE_NAME, 0, nullptr, nullptr);
    test_error_ret(err, "clGetKernelArgInfo", TEST_FAIL);

    err = clGetKernelArgInfo(kernel, SINGLE_KERNEL_ARG_NUMBER,
                             CL_KERNEL_ARG_NAME, 0, nullptr, nullptr);
    test_error_ret(err, "clGetKernelArgInfo", TEST_FAIL);

    return TEST_PASS;
}

/* Ensure clGetKernelArgInfo returns the correct size in bytes for the
 * kernel arg name */
static int test_arg_name_size(cl_context context, cl_device_id deviceID,
                              char const* kernel_src)
{
    size_t size;
    /* We are adding +1 because the argument used in this kernel is argument0
     * which has 1 extra character than just the base argument name */
    char arg_return[sizeof(KERNEL_ARGUMENT_NAME) + 1];
    clProgramWrapper program;
    clKernelWrapper kernel;
    cl_int err = create_single_kernel_helper_with_build_options(
        context, &program, &kernel, 1, &kernel_src, "get_kernel_arg_info",
        get_build_options(deviceID).c_str());

    test_error_ret(err, "create_single_kernel_helper_with_build_options",
                   TEST_FAIL);

    err =
        clGetKernelArgInfo(kernel, SINGLE_KERNEL_ARG_NUMBER, CL_KERNEL_ARG_NAME,
                           sizeof(arg_return), &arg_return, &size);
    test_error_ret(err, "clGetKernelArgInfo", TEST_FAIL);
    if (size == sizeof(KERNEL_ARGUMENT_NAME) + 1)
    {
        return TEST_PASS;
    }
    else
    {
        return TEST_FAIL;
    }
}

static int run_boundary_tests(cl_context context, cl_device_id deviceID)
{
    int failed_tests = 0;

    cl_kernel_arg_address_qualifier address_qualifier =
        CL_KERNEL_ARG_ADDRESS_GLOBAL;
    cl_kernel_arg_access_qualifier access_qualifier = CL_KERNEL_ARG_ACCESS_NONE;
    cl_kernel_arg_type_qualifier type_qualifier = CL_KERNEL_ARG_TYPE_NONE;
    std::string arg_type = "int*";
    KernelArgInfo arg_info(address_qualifier, access_qualifier, type_qualifier,
                           arg_type, SINGLE_KERNEL_ARG_NUMBER);
    const std::string kernel_src = generate_kernel({ arg_info });

    failed_tests += test_arg_name_size(context, deviceID, kernel_src.c_str());

    if (test_null_param(context, deviceID, kernel_src.c_str()) != TEST_PASS)
    {
        failed_tests++;
    }

    return failed_tests;
}

static int run_all_tests(cl_context context, cl_device_id deviceID)
{

    int failed_scalar_tests = run_scalar_vector_tests(context, deviceID);
    if (failed_scalar_tests == 0)
    {
        log_info("All Data Type Tests Passed\n");
    }
    else
    {
        log_error("%d Data Type Test(s) Failed\n", failed_scalar_tests);
    }

    int failed_image_tests = 0;
    if (checkForImageSupport(deviceID) == 0)
    {
        failed_image_tests = run_image_tests(context, deviceID);
        if (failed_image_tests == 0)
        {
            log_info("All Image Tests Passed\n");
        }
        else
        {
            log_error("%d Image Test(s) Failed\n", failed_image_tests);
        }
    }
    int failed_pipe_tests = 0;
    // TODO https://github.com/KhronosGroup/OpenCL-CTS/issues/1244
    if (false)
    {
        failed_pipe_tests = run_pipe_tests(context, deviceID);
        if (failed_pipe_tests == 0)
        {
            log_info("All Pipe Tests Passed\n");
        }
        else
        {
            log_error("%d Pipe Test(s) Failed\n", failed_pipe_tests);
        }
    }

    int failed_boundary_tests = run_boundary_tests(context, deviceID);
    if (failed_boundary_tests == 0)
    {
        log_info("All Edge Case Tests Passed\n");
    }
    else
    {
        log_error("%d Edge Case Test(s) Failed\n", failed_boundary_tests);
    }

    return (failed_scalar_tests + failed_image_tests + failed_pipe_tests
            + failed_boundary_tests);
}

int test_get_kernel_arg_info(cl_device_id deviceID, cl_context context,
                             cl_command_queue queue, int num_elements)
{
    int failed_tests = run_all_tests(context, deviceID);
    if (failed_tests != 0)
    {
        log_error("%d Test(s) Failed\n", failed_tests);
        return TEST_FAIL;
    }
    else
    {
        return TEST_PASS;
    }
}
