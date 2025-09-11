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
#include "testBase.h"
#include "test_unload_platform_compiler_resources.hpp"

#include <cassert>
#include <chrono>
#include <functional>
#include <future>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <fstream>

#if defined(_WIN32)
const std::string slash = "\\";
#else
const std::string slash = "/";
#endif
std::string compilerSpvBinaries =
    "compiler" + slash + "spirv_bin" + slash + "write_kernel.spv";

const std::string spvExt = ".spv";

std::vector<unsigned char> readBinary(const char *file_name)
{
    using namespace std;

    ifstream file(file_name, ios::in | ios::binary | ios::ate);

    std::vector<char> tmpBuffer(0);

    if (file.is_open())
    {
        size_t size = file.tellg();
        tmpBuffer.resize(size);
        file.seekg(0, ios::beg);
        file.read(&tmpBuffer[0], size);
        file.close();
    }
    else
    {
        log_error("File %s not found\n", file_name);
    }

    std::vector<unsigned char> result(tmpBuffer.begin(), tmpBuffer.end());

    return result;
}

namespace {

class unload_test_failure : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;

    explicit unload_test_failure(const std::string &function, cl_int error)
        : std::runtime_error(function + " == " + std::to_string(error))
    {}
};

class build_base {
public:
    build_base(cl_context context, cl_device_id device)
        : m_context{ context }, m_device{ device }
    {}
    virtual ~build_base() { reset(); }
    build_base(const build_base &) = delete;
    build_base &operator=(const build_base &) = delete;

    virtual void create() = 0;

    virtual void compile()
    {
        assert(nullptr != m_program);

        const cl_int err = clCompileProgram(m_program, 1, &m_device, nullptr, 0,
                                            nullptr, nullptr, nullptr, nullptr);
        if (CL_SUCCESS != err)
            throw unload_test_failure("clCompileProgram()", err);
    }

    virtual void link()
    {
        assert(nullptr != m_program);

        cl_int err = CL_INVALID_PLATFORM;
        m_executable = clLinkProgram(m_context, 1, &m_device, nullptr, 1,
                                     &m_program, nullptr, nullptr, &err);
        if (CL_SUCCESS != err)
            throw unload_test_failure("clLinkProgram()", err);
        if (nullptr == m_executable)
            throw unload_test_failure("clLinkProgram returned nullptr");
    }

    virtual void verify()
    {
        assert(nullptr != m_executable);

        cl_int err = CL_INVALID_VALUE;

        const clKernelWrapper kernel =
            clCreateKernel(m_executable, "write_kernel", &err);
        if (CL_SUCCESS != err)
            throw unload_test_failure("clCreateKernel()", err);

        const clCommandQueueWrapper queue =
            clCreateCommandQueue(m_context, m_device, 0, &err);
        if (CL_SUCCESS != err)
            throw unload_test_failure("clCreateCommandQueue()", err);

        const clMemWrapper buffer = clCreateBuffer(
            m_context, CL_MEM_READ_WRITE, sizeof(cl_uint), nullptr, &err);
        if (CL_SUCCESS != err)
            throw unload_test_failure("clCreateBuffer()", err);

        cl_uint value = 0;

        err = clSetKernelArg(kernel, 0, sizeof(buffer), &buffer);
        if (CL_SUCCESS != err)
            throw unload_test_failure("clSetKernelArg()", err);

        static const size_t work_size = 1;
        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &work_size,
                                     nullptr, 0, nullptr, nullptr);
        if (CL_SUCCESS != err)
            throw unload_test_failure("clEnqueueNDRangeKernel()", err);

        err = clEnqueueReadBuffer(queue, buffer, CL_BLOCKING, 0,
                                  sizeof(cl_uint), &value, 0, nullptr, nullptr);
        if (CL_SUCCESS != err)
            throw unload_test_failure("clEnqueueReadBuffer()", err);

        err = clFinish(queue);
        if (CL_SUCCESS != err) throw unload_test_failure("clFinish()", err);

        if (42 != value)
        {
            throw unload_test_failure("Kernel wrote " + std::to_string(value)
                                      + ", expected 42");
        }
    }

    void reset()
    {
        if (m_program)
        {
            clReleaseProgram(m_program);
            m_program = nullptr;
        }
        if (m_executable)
        {
            clReleaseProgram(m_executable);
            m_executable = nullptr;
        }
    }

    void build()
    {
        compile();
        link();
    }

protected:
    const cl_context m_context;
    const cl_device_id m_device;
    cl_program m_program{};
    cl_program m_executable{};
};

/**
 * @brief initializer_list type for constructing loops over build tests.
 */
using build_list = std::initializer_list<std::reference_wrapper<build_base>>;

class build_with_source : public build_base {
public:
    using build_base::build_base;

    void create() final
    {
        assert(nullptr == m_program);

        static const char *sources[] = { write_kernel_source };

        cl_int err = CL_INVALID_PLATFORM;
        m_program =
            clCreateProgramWithSource(m_context, 1, sources, nullptr, &err);
        if (CL_SUCCESS != err)
            throw unload_test_failure("clCreateProgramWithSource()", err);
        if (nullptr == m_program)
            throw unload_test_failure(
                "clCreateProgramWithSource returned nullptr");
    }
};

class build_with_binary : public build_base {
public:
    build_with_binary(const cl_context context, const cl_device_id device,
                      const std::vector<unsigned char> &binary)
        : build_base{ context, device }, m_binary{ binary }
    {}

    build_with_binary(const cl_context context, const cl_device_id device)
        : build_base{ context, device }
    {
        cl_int err = CL_INVALID_VALUE;

        /* Build the program from source */
        static const char *sources[] = { write_kernel_source };
        clProgramWrapper program =
            clCreateProgramWithSource(m_context, 1, sources, nullptr, &err);
        if (CL_SUCCESS != err)
            throw unload_test_failure("clCreateProgramWithSource()", err);

        err = clCompileProgram(program, 1, &m_device, nullptr, 0, nullptr,
                               nullptr, nullptr, nullptr);
        if (CL_SUCCESS != err)
            throw unload_test_failure("clCompileProgram()", err);

        const clProgramWrapper executable =
            clLinkProgram(m_context, 1, &m_device, nullptr, 1, &program,
                          nullptr, nullptr, &err);
        if (CL_SUCCESS != err)
            throw unload_test_failure("clLinkProgram()", err);

        size_t binary_size;
        err = clGetProgramInfo(executable, CL_PROGRAM_BINARY_SIZES,
                               sizeof(binary_size), &binary_size, nullptr);
        if (CL_SUCCESS != err)
            throw unload_test_failure("clGetProgramInfo()", err);

        m_binary.resize(binary_size);

        /* Grab the program binary */
        unsigned char *binaries[] = { m_binary.data() };
        err = clGetProgramInfo(executable, CL_PROGRAM_BINARIES,
                               sizeof(unsigned char *), binaries, nullptr);
        if (CL_SUCCESS != err)
            throw unload_test_failure("clGetProgramInfo()", err);
    }

    void create() final
    {
        assert(nullptr == m_executable);

        const unsigned char *binaries[] = { m_binary.data() };
        const size_t binary_sizes[] = { m_binary.size() };

        cl_int err = CL_INVALID_PLATFORM;
        m_executable = clCreateProgramWithBinary(
            m_context, 1, &m_device, binary_sizes, binaries, nullptr, &err);
        if (CL_SUCCESS != err)
            throw unload_test_failure("clCreateProgramWithBinary()", err);
        if (nullptr == m_executable)
            throw unload_test_failure(
                "clCreateProgramWithBinary returned nullptr");
    }

    void compile() final
    {
        assert(nullptr != m_executable);

        /* Program created from binary, there is nothing to do */
    }

    void link() final
    {
        assert(nullptr != m_executable);

        const cl_int err = clBuildProgram(m_executable, 1, &m_device, nullptr,
                                          nullptr, nullptr);
        if (CL_SUCCESS != err)
            throw unload_test_failure("clBuildProgram()", err);
    }

private:
    std::vector<unsigned char> m_binary;
};

class build_with_il : public build_base {
public:
    build_with_il(const cl_context context, const cl_platform_id platform,
                  const cl_device_id device)
        : build_base{ context, device }
    {
        /* Disable build_with_il if neither core nor extension functionality is
         * available */
        m_enabled = false;

        Version version = get_device_cl_version(device);
        if (version >= Version(2, 1))
        {
            std::string sILVersion = get_device_il_version_string(device);
            if (version < Version(3, 0) || !sILVersion.empty())
            {
                m_enabled = true;
            }

            m_CreateProgramWithIL = clCreateProgramWithIL;
        }
        else if (is_extension_available(device, "cl_khr_il_program"))
        {
            m_CreateProgramWithIL = (decltype(m_CreateProgramWithIL))
                clGetExtensionFunctionAddressForPlatform(
                    platform, "clCreateProgramWithILKHR");
            if (nullptr == m_CreateProgramWithIL)
            {
                throw unload_test_failure("cl_khr_il_program supported, but "
                                          "function address is nullptr");
            }
            m_enabled = true;
        }

        cl_uint address_bits{};
        const cl_int err =
            clGetDeviceInfo(device, CL_DEVICE_ADDRESS_BITS, sizeof(cl_uint),
                            &address_bits, nullptr);
        if (CL_SUCCESS != err)
        {
            throw unload_test_failure("Failure getting device address bits");
        }

        std::vector<unsigned char> kernel_buffer;

        std::string file_name =
            compilerSpvBinaries + std::to_string(address_bits);
        m_spirv_binary = readBinary(file_name.c_str());
        m_spirv_size = m_spirv_binary.size();
    }

    void create() final
    {
        if (!m_enabled) return;

        assert(nullptr == m_program);

        cl_int err = CL_INVALID_PLATFORM;
        m_program = m_CreateProgramWithIL(m_context, &m_spirv_binary[0],
                                          m_spirv_size, &err);
        if (CL_SUCCESS != err)
            throw unload_test_failure("clCreateProgramWithIL()", err);
        if (nullptr == m_program)
            throw unload_test_failure("clCreateProgramWithIL returned nullptr");
    }

    void compile() final
    {
        if (!m_enabled) return;
        build_base::compile();
    }

    void link() final
    {
        if (!m_enabled) return;
        build_base::link();
    }

    void verify() final
    {
        if (!m_enabled) return;
        build_base::verify();
    }

private:
    std::vector<unsigned char> m_spirv_binary;
    size_t m_spirv_size;
    bool m_enabled;

    using CreateProgramWithIL_fn = decltype(&clCreateProgramWithIL);
    CreateProgramWithIL_fn m_CreateProgramWithIL;
};
}

static cl_platform_id device_platform(cl_device_id device)
{
    cl_platform_id platform;
    const cl_int err = clGetDeviceInfo(device, CL_DEVICE_PLATFORM,
                                       sizeof(platform), &platform, nullptr);
    if (CL_SUCCESS != err)
    {
        log_error("Failure getting platform of tested device\n");
        return nullptr;
    }

    return platform;
}

static void unload_platform_compiler(const cl_platform_id platform)
{
    const cl_int err = clUnloadPlatformCompiler(platform);
    if (CL_SUCCESS != err)
        throw unload_test_failure("clUnloadPlatformCompiler()", err);
}

/* Test calling the function with a valid platform */
REGISTER_TEST(unload_valid)
{
    const cl_platform_id platform = device_platform(device);
    const long int err = clUnloadPlatformCompiler(platform);

    if (CL_SUCCESS != err)
    {
        log_error("Test failure: clUnloadPlatformCompiler() == %ld\n", err);
        return 1;
    }

    return 0;
}

/* Test calling the function with invalid platform */
/* Disabling temporarily, see GitHub #977
REGISTER_TEST(unload_invalid)
{
    const long int err = clUnloadPlatformCompiler(nullptr);

    if (CL_INVALID_PLATFORM != err)
    {
        log_error("Test failure: clUnloadPlatformCompiler() == %ld\n", err);
        return 1;
    }

    return 0;
}
*/

/* Test calling the function multiple times in a row */
REGISTER_TEST(unload_repeated)
{
    check_compiler_available(device);

    const cl_platform_id platform = device_platform(device);
    try
    {
        build_with_source source(context, device);
        build_with_binary binary(context, device);
        build_with_il il(context, platform, device);

        for (build_base &test : build_list{ source, binary, il })
        {
            unload_platform_compiler(platform);
            unload_platform_compiler(platform);

            test.create();
            test.build();
            test.verify();
        }
    } catch (const unload_test_failure &e)
    {
        log_error("Test failure: %s\n", e.what());
        return 1;
    }

    return 0;
}

/* Test calling the function between compilation and linking of programs */
REGISTER_TEST(unload_compile_unload_link)
{
    check_compiler_available(device);

    const cl_platform_id platform = device_platform(device);
    try
    {
        build_with_source source(context, device);
        build_with_binary binary(context, device);
        build_with_il il(context, platform, device);

        for (build_base &test : build_list{ source, binary, il })
        {
            unload_platform_compiler(platform);
            test.create();
            test.compile();
            unload_platform_compiler(platform);
            test.link();
            test.verify();
        }
    } catch (const unload_test_failure &e)
    {
        log_error("Test failure: %s\n", e.what());
        return 1;
    }

    return 0;
}

/* Test calling the function between program build and kernel creation */
REGISTER_TEST(unload_build_unload_create_kernel)
{
    check_compiler_available(device);

    const cl_platform_id platform = device_platform(device);
    try
    {
        build_with_source source(context, device);
        build_with_binary binary(context, device);
        build_with_il il(context, platform, device);

        for (build_base &test : build_list{ source, binary, il })
        {
            unload_platform_compiler(platform);
            test.create();
            test.build();
            unload_platform_compiler(platform);
            test.verify();
        }
    } catch (const unload_test_failure &e)
    {
        log_error("Test failure: %s\n", e.what());
        return 1;
    }

    return 0;
}

/* Test linking together two programs that were built with a call to the unload
 * function in between */
REGISTER_TEST(unload_link_different)
{
    check_compiler_available(device);

    const cl_platform_id platform = device_platform(device);

    static const char *sources_1[] = { "unsigned int a() { return 42; }" };
    static const char *sources_2[] = { R"(
		unsigned int a();
		kernel void test(global unsigned int *p)
		{
			*p = a();
		})" };

    cl_int err = CL_INVALID_PLATFORM;

    /* Create and compile program 1 */
    const clProgramWrapper program_1 =
        clCreateProgramWithSource(context, 1, sources_1, nullptr, &err);
    if (CL_SUCCESS != err)
    {
        log_error("Test failure: clCreateProgramWithSource() == %ld\n",
                  static_cast<long int>(err));
        return 1;
    }

    err = clCompileProgram(program_1, 1, &device, nullptr, 0, nullptr, nullptr,
                           nullptr, nullptr);
    if (CL_SUCCESS != err)
    {
        log_error("Test failure: clCompileProgram() == %ld\n",
                  static_cast<long int>(err));
        return 1;
    }

    /* Unload the platform compiler */
    err = clUnloadPlatformCompiler(platform);
    if (CL_SUCCESS != err)
    {
        log_error("Test failure: clUnloadPlatformCompiler() == %ld\n",
                  static_cast<long int>(err));
        return 1;
    }

    /* Create and compile program 2 with the new compiler context */
    const clProgramWrapper program_2 =
        clCreateProgramWithSource(context, 1, sources_2, nullptr, &err);
    if (CL_SUCCESS != err)
    {
        log_error("Test failure: clCreateProgramWithSource() == %ld\n",
                  static_cast<long int>(err));
        return 1;
    }

    err = clCompileProgram(program_2, 1, &device, nullptr, 0, nullptr, nullptr,
                           nullptr, nullptr);
    if (CL_SUCCESS != err)
    {
        log_error("Test failure: clCompileProgram() == %ld\n",
                  static_cast<long int>(err));
        return 1;
    }

    /* Link the two programs into an executable program */
    const cl_program compiled_programs[] = { program_1, program_2 };

    const clProgramWrapper executable =
        clLinkProgram(context, 1, &device, nullptr, 2, compiled_programs,
                      nullptr, nullptr, &err);
    if (CL_SUCCESS != err)
    {
        log_error("Test failure: clLinkProgram() == %ld\n",
                  static_cast<long int>(err));
        return 1;
    }

    /* Verify execution of a kernel from the linked executable */
    const clKernelWrapper kernel = clCreateKernel(executable, "test", &err);
    if (CL_SUCCESS != err)
    {
        log_error("Test failure: clCreateKernel() == %ld\n",
                  static_cast<long int>(err));
        return 1;
    }

    const clCommandQueueWrapper test_queue =
        clCreateCommandQueue(context, device, 0, &err);
    if (CL_SUCCESS != err)
    {
        log_error("Test failure: clCreateCommandQueue() == %ld\n",
                  static_cast<long int>(err));
        return 1;
    }

    const clMemWrapper buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                               sizeof(cl_uint), nullptr, &err);
    if (CL_SUCCESS != err)
    {
        log_error("Test failure: clCreateBuffer() == %ld\n",
                  static_cast<long int>(err));
        return 1;
    }

    cl_uint value = 0;

    err = clSetKernelArg(kernel, 0, sizeof(buffer), &buffer);
    if (CL_SUCCESS != err)
    {
        log_error("Test failure: clSetKernelArg() == %ld\n",
                  static_cast<long int>(err));
        return 1;
    }

    static const size_t work_size = 1;
    err = clEnqueueNDRangeKernel(test_queue, kernel, 1, nullptr, &work_size,
                                 nullptr, 0, nullptr, nullptr);
    if (CL_SUCCESS != err)
    {
        log_error("Test failure: clEnqueueNDRangeKernel() == %ld\n",
                  static_cast<long int>(err));
        return 1;
    }

    err = clEnqueueReadBuffer(test_queue, buffer, CL_BLOCKING, 0,
                              sizeof(cl_uint), &value, 0, nullptr, nullptr);
    if (CL_SUCCESS != err)
    {
        log_error("Test failure: clEnqueueReadBuffer() == %ld\n",
                  static_cast<long int>(err));
        return 1;
    }

    err = clFinish(test_queue);
    if (CL_SUCCESS != err) throw unload_test_failure("clFinish()", err);

    if (42 != value)
    {
        log_error("Test failure: Kernel wrote %lu, expected 42)\n",
                  static_cast<long unsigned>(value));
        return 1;
    }

    return 0;
}

/* Test calling the function in a thread while others threads are building
 * programs */
REGISTER_TEST(unload_build_threaded)
{
    using clock = std::chrono::steady_clock;

    check_compiler_available(device);

    const cl_platform_id platform = device_platform(device);

    const auto end = clock::now() + std::chrono::seconds(5);

    const auto unload_thread = [&end, platform] {
        bool success = true;

        /* Repeatedly unload the compiler */
        try
        {
            while (clock::now() < end)
            {
                unload_platform_compiler(platform);
            }
        } catch (const unload_test_failure &e)
        {
            log_error("Test failure: %s\n", e.what());
            success = false;
        }

        return success;
    };

    const auto build_thread = [&end](build_base *build) {
        bool success = true;

        try
        {
            while (clock::now() < end)
            {
                build->create();
                build->build();
                build->verify();
                build->reset();
            }
        } catch (unload_test_failure &e)
        {
            log_error("Test failure: %s\n", e.what());
            success = false;
        }

        return success;
    };

    build_with_source build_source(context, device);
    build_with_binary build_binary(context, device);
    build_with_il build_il(context, platform, device);

    /* Run all threads in parallel and wait for them to finish */
    std::future<bool> unload_result =
        std::async(std::launch::async, unload_thread);
    std::future<bool> build_source_result =
        std::async(std::launch::async, build_thread, &build_source);
    std::future<bool> build_binary_result =
        std::async(std::launch::async, build_thread, &build_binary);
    std::future<bool> build_il_result =
        std::async(std::launch::async, build_thread, &build_il);

    bool success = true;
    if (!unload_result.get())
    {
        log_error("unload_thread failed\n");
        success = false;
    }
    if (!build_source_result.get())
    {
        log_error("build_with_source failed\n");
        success = false;
    }
    if (!build_binary_result.get())
    {
        log_error("build_with_binary failed\n");
        success = false;
    }
    if (!build_il_result.get())
    {
        log_error("build_with_il failed\n");
        success = false;
    }

    return success ? 0 : 1;
}

/* Test grabbing program build information after calling the unload function */
REGISTER_TEST(unload_build_info)
{
    check_compiler_available(device);

    const cl_platform_id platform = device_platform(device);

    static const char *sources[] = { write_kernel_source };

    cl_int err = CL_INVALID_PLATFORM;
    /* Create and build the initial program from source */
    const clProgramWrapper program =
        clCreateProgramWithSource(context, 1, sources, nullptr, &err);
    if (CL_SUCCESS != err)
    {
        log_error("Test failure: clCreateProgramWithSource() == %ld\n",
                  static_cast<long int>(err));
        return 1;
    }

    static const std::string options("-Dtest");

    err =
        clBuildProgram(program, 1, &device, options.c_str(), nullptr, nullptr);
    if (CL_SUCCESS != err)
    {
        log_error("Test failure: clCompileProgram() == %ld\n",
                  static_cast<long int>(err));
        return 1;
    }

    /* Unload the compiler */
    err = clUnloadPlatformCompiler(platform);
    if (CL_SUCCESS != err)
    {
        log_error("Test failure: clUnloadPlatformCompiler() == %ld\n",
                  static_cast<long int>(err));
        return 1;
    }

    std::vector<cl_program_build_info> infos{ CL_PROGRAM_BUILD_STATUS,
                                              CL_PROGRAM_BUILD_OPTIONS,
                                              CL_PROGRAM_BUILD_LOG,
                                              CL_PROGRAM_BINARY_TYPE };

    if (get_device_cl_version(device) >= Version(2, 0))
    {
        infos.push_back(CL_PROGRAM_BUILD_GLOBAL_VARIABLE_TOTAL_SIZE);
    }

    /* Try grabbing the infos after the compiler unload */
    for (cl_program_build_info info : infos)
    {
        size_t info_size = 0;
        err = clGetProgramBuildInfo(program, device, info, 0, nullptr,
                                    &info_size);
        if (CL_SUCCESS != err)
        {
            log_error("Test failure: clGetProgramBuildInfo() == %ld\n",
                      static_cast<long int>(err));
            return 1;
        }

        std::vector<char> info_value(info_size);

        size_t written_size = 0;
        err = clGetProgramBuildInfo(program, device, info, info_size,
                                    &info_value[0], &written_size);
        if (CL_SUCCESS != err)
        {
            log_error("Test failure: clGetProgramBuildInfo() == %ld\n",
                      static_cast<long int>(err));
            return 1;
        }
        else if (written_size != info_size)
        {
            log_error("Test failure: Written info value size (%zu) was "
                      "different from "
                      "queried size (%zu).\n",
                      written_size, info_size);
            return 1;
        }

        /* Verify the information we know the answer to */
        switch (info)
        {
            case CL_PROGRAM_BUILD_STATUS: {
                constexpr size_t value_size = sizeof(cl_build_status);
                if (value_size != info_size)
                {
                    log_error("Test failure: Expected CL_PROGRAM_BUILD_STATUS "
                              "of size %zu, "
                              "but got %zu\n",
                              value_size, info_size);
                    return 1;
                }
                cl_build_status value;
                memcpy(&value, &info_value[0], value_size);
                if (CL_BUILD_SUCCESS != value)
                {
                    log_error(
                        "Test failure: CL_PROGRAM_BUILD_STATUS did not return "
                        "CL_BUILD_SUCCESS (%ld), but %ld\n",
                        static_cast<long int>(CL_BUILD_SUCCESS),
                        static_cast<long int>(value));
                    return 1;
                }
            }
            break;

            case CL_PROGRAM_BUILD_OPTIONS: {
                const size_t value_size = options.length() + 1;
                if (value_size != info_size)
                {
                    log_error("Test failure: Expected CL_PROGRAM_BUILD_OPTIONS "
                              "of size "
                              "%zu, but got %zu\n",
                              value_size, info_size);
                    return 1;
                }
                else if (options != &info_value[0])
                {
                    log_error("Test failure: CL_PROGRAM_BUILD_OPTIONS returned "
                              "\"%s\" "
                              "instead of \"%s\"\n",
                              &info_value[0], options.c_str());
                    return 1;
                }
            }
            break;

            case CL_PROGRAM_BINARY_TYPE: {
                constexpr size_t value_size = sizeof(cl_program_binary_type);
                if (value_size != info_size)
                {
                    log_error("Test failure: Expected CL_PROGRAM_BINARY_TYPE "
                              "of size %zu, "
                              "but got %zu\n",
                              value_size, info_size);
                    return 1;
                }
                cl_program_binary_type value;
                memcpy(&value, &info_value[0], value_size);
                if (CL_PROGRAM_BINARY_TYPE_EXECUTABLE != value)
                {
                    log_error(
                        "Test failure: CL_PROGRAM_BINARY_TYPE did not return "
                        "CL_PROGRAM_BINARY_TYPE_EXECUTABLE (%ld), but %ld\n",
                        static_cast<long int>(
                            CL_PROGRAM_BINARY_TYPE_EXECUTABLE),
                        static_cast<long int>(value));
                    return 1;
                }
            }
            break;
        }
    }

    return 0;
}

/* Test calling the unload function between program building and fetching the
 * program binaries */
REGISTER_TEST(unload_program_binaries)
{
    check_compiler_available(device);

    const cl_platform_id platform = device_platform(device);

    static const char *sources[] = { write_kernel_source };

    cl_int err = CL_INVALID_PLATFORM;
    /* Create and build the initial program from source */
    const clProgramWrapper program =
        clCreateProgramWithSource(context, 1, sources, nullptr, &err);
    if (CL_SUCCESS != err)
    {
        log_error("Test failure: clCreateProgramWithSource() == %ld\n",
                  static_cast<long int>(err));
        return 1;
    }

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (CL_SUCCESS != err)
    {
        log_error("Test failure: clCompileProgram() == %ld\n",
                  static_cast<long int>(err));
        return 1;
    }

    /* Unload the compiler */
    err = clUnloadPlatformCompiler(platform);
    if (CL_SUCCESS != err)
    {
        log_error("Test failure: clUnloadPlatformCompiler() == %ld\n",
                  static_cast<long int>(err));
        return 1;
    }

    /* Grab the built executable binary after the compiler unload */
    size_t binary_size;
    err = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES,
                           sizeof(binary_size), &binary_size, nullptr);
    if (CL_SUCCESS != err)
    {
        log_error("Test failure: clGetProgramInfo() == %ld\n",
                  static_cast<long int>(err));
        return 1;
    }

    std::vector<unsigned char> binary(binary_size);

    unsigned char *binaries[] = { binary.data() };
    err = clGetProgramInfo(program, CL_PROGRAM_BINARIES,
                           sizeof(unsigned char *), binaries, nullptr);
    if (CL_SUCCESS != err)
    {
        log_error("Test failure: clGetProgramInfo() == %ld\n",
                  static_cast<long int>(err));
        return 1;
    }

    /* Create a new program from the binary and test its execution */
    try
    {
        build_with_binary build_binary(context, device, binary);
        build_binary.create();
        build_binary.build();
        build_binary.verify();
    } catch (unload_test_failure &e)
    {
        log_error("Test failure: %s\n", e.what());
        return 1;
    }

    return 0;
}
