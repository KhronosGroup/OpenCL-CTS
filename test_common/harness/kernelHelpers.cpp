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
#include "crc32.h"
#include "kernelHelpers.h"
#include "deviceInfo.h"
#include "errorHelpers.h"
#include "imageHelpers.h"
#include "typeWrappers.h"
#include "testHarness.h"
#include "parseParameters.h"

#include <cassert>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <mutex>
#include <algorithm>

#if defined(_WIN32)
std::string slash = "\\";
#else
std::string slash = "/";
#endif

static std::mutex gCompilerMutex;

static cl_int get_first_device_id(const cl_context context,
                                  cl_device_id &device);

long get_file_size(const std::string &fileName)
{
    std::ifstream ifs(fileName.c_str(), std::ios::binary);
    if (!ifs.good()) return 0;
    // get length of file:
    ifs.seekg(0, std::ios::end);
    std::ios::pos_type length = ifs.tellg();
    return static_cast<long>(length);
}

static std::string get_kernel_content(unsigned int numKernelLines,
                                      const char *const *kernelProgram)
{
    std::string kernel;
    for (size_t i = 0; i < numKernelLines; ++i)
    {
        std::string chunk(kernelProgram[i], 0, std::string::npos);
        kernel += chunk;
    }

    return kernel;
}

std::string get_kernel_name(const std::string &source)
{
    // Create list of kernel names
    std::string kernelsList;
    size_t kPos = source.find("kernel");
    while (kPos != std::string::npos)
    {
        // check for '__kernel'
        size_t pos = kPos;
        if (pos >= 2 && source[pos - 1] == '_' && source[pos - 2] == '_')
            pos -= 2;

        // check character before 'kernel' (white space expected)
        size_t wsPos = source.find_last_of(" \t\r\n", pos);
        if (wsPos == std::string::npos || wsPos + 1 == pos)
        {
            // check character after 'kernel' (white space expected)
            size_t akPos = kPos + sizeof("kernel") - 1;
            wsPos = source.find_first_of(" \t\r\n", akPos);
            if (!(wsPos == akPos))
            {
                kPos = source.find("kernel", kPos + 1);
                continue;
            }

            bool attributeFound;
            do
            {
                attributeFound = false;
                // find '(' after kernel name name
                size_t pPos = source.find("(", akPos);
                if (!(pPos != std::string::npos)) continue;

                // check for not empty kernel name before '('
                pos = source.find_last_not_of(" \t\r\n", pPos - 1);
                if (!(pos != std::string::npos && pos > akPos)) continue;

                // find character before kernel name
                wsPos = source.find_last_of(" \t\r\n", pos);
                if (!(wsPos != std::string::npos && wsPos >= akPos)) continue;

                std::string name =
                    source.substr(wsPos + 1, pos + 1 - (wsPos + 1));
                // check for kernel attribute
                if (name == "__attribute__")
                {
                    attributeFound = true;
                    int pCount = 1;
                    akPos = pPos + 1;
                    while (pCount > 0 && akPos != std::string::npos)
                    {
                        akPos = source.find_first_of("()", akPos + 1);
                        if (akPos != std::string::npos)
                        {
                            if (source[akPos] == '(')
                                pCount++;
                            else
                                pCount--;
                        }
                    }
                }
                else
                {
                    kernelsList += name + ".";
                }
            } while (attributeFound);
        }
        kPos = source.find("kernel", kPos + 1);
    }
    std::ostringstream oss;
    if (MAX_LEN_FOR_KERNEL_LIST > 0)
    {
        if (kernelsList.size() > MAX_LEN_FOR_KERNEL_LIST + 1)
        {
            kernelsList = kernelsList.substr(0, MAX_LEN_FOR_KERNEL_LIST + 1);
            kernelsList[kernelsList.size() - 1] = '.';
            kernelsList[kernelsList.size() - 1] = '.';
        }
        oss << kernelsList;
    }
    return oss.str();
}

static std::string
get_offline_compilation_file_type_str(const CompilationMode compilationMode)
{
    switch (compilationMode)
    {
        default: assert(0 && "Invalid compilation mode"); abort();
        case kOnline:
            assert(0 && "Invalid compilation mode for offline compilation");
            abort();
        case kBinary: return "binary";
        case kSpir_v: return "SPIR-V";
    }
}

static std::string get_unique_filename_prefix(unsigned int numKernelLines,
                                              const char *const *kernelProgram,
                                              const char *buildOptions)
{
    std::string kernel = get_kernel_content(numKernelLines, kernelProgram);
    std::string kernelName = get_kernel_name(kernel);
    cl_uint kernelCrc = crc32(kernel.data(), kernel.size());
    std::ostringstream oss;
    oss << kernelName << std::hex << std::setfill('0') << std::setw(8)
        << kernelCrc;
    if (buildOptions)
    {
        cl_uint bOptionsCrc = crc32(buildOptions, strlen(buildOptions));
        oss << '.' << std::hex << std::setfill('0') << std::setw(8)
            << bOptionsCrc;
    }
    return oss.str();
}


static std::string
get_cl_build_options_filename_with_path(const std::string &filePath,
                                        const std::string &fileNamePrefix)
{
    return filePath + slash + fileNamePrefix + ".options";
}

static std::string
get_cl_source_filename_with_path(const std::string &filePath,
                                 const std::string &fileNamePrefix)
{
    return filePath + slash + fileNamePrefix + ".cl";
}

static std::string
get_binary_filename_with_path(CompilationMode mode, cl_uint deviceAddrSpaceSize,
                              const std::string &filePath,
                              const std::string &fileNamePrefix)
{
    std::string binaryFilename = filePath + slash + fileNamePrefix;
    if (kSpir_v == mode)
    {
        std::ostringstream extension;
        extension << ".spv" << deviceAddrSpaceSize;
        binaryFilename += extension.str();
    }
    return binaryFilename;
}

static bool file_exist_on_disk(const std::string &filePath,
                               const std::string &fileName)
{
    std::string fileNameWithPath = filePath + slash + fileName;
    bool exist = false;
    std::ifstream ifs;

    ifs.open(fileNameWithPath.c_str(), std::ios::binary);
    if (ifs.good()) exist = true;
    ifs.close();
    return exist;
}

static bool should_save_kernel_source_to_disk(CompilationMode mode,
                                              CompilationCacheMode cacheMode,
                                              const std::string &binaryPath,
                                              const std::string &binaryName)
{
    bool saveToDisk = false;
    if (cacheMode == kCacheModeDumpCl
        || (cacheMode == kCacheModeOverwrite && mode != kOnline))
    {
        saveToDisk = true;
    }
    if (cacheMode == kCacheModeCompileIfAbsent && mode != kOnline)
    {
        saveToDisk = !file_exist_on_disk(binaryPath, binaryName);
    }
    return saveToDisk;
}

static int save_kernel_build_options_to_disk(const std::string &path,
                                             const std::string &prefix,
                                             const char *buildOptions)
{
    std::string filename =
        get_cl_build_options_filename_with_path(path, prefix);
    std::ofstream ofs(filename.c_str(), std::ios::binary);
    if (!ofs.good())
    {
        log_info("Can't save kernel build options: %s\n", filename.c_str());
        return -1;
    }
    ofs.write(buildOptions, strlen(buildOptions));
    ofs.close();
    log_info("Saved kernel build options to file: %s\n", filename.c_str());
    return CL_SUCCESS;
}

static int save_kernel_source_to_disk(const std::string &path,
                                      const std::string &prefix,
                                      const std::string &source)
{
    std::string filename = get_cl_source_filename_with_path(path, prefix);
    std::ofstream ofs(filename.c_str(), std::ios::binary);
    if (!ofs.good())
    {
        log_info("Can't save kernel source: %s\n", filename.c_str());
        return -1;
    }
    ofs.write(source.c_str(), source.size());
    ofs.close();
    log_info("Saved kernel source to file: %s\n", filename.c_str());
    return CL_SUCCESS;
}

static int
save_kernel_source_and_options_to_disk(unsigned int numKernelLines,
                                       const char *const *kernelProgram,
                                       const char *buildOptions)
{
    int error;

    std::string kernel = get_kernel_content(numKernelLines, kernelProgram);
    std::string kernelNamePrefix =
        get_unique_filename_prefix(numKernelLines, kernelProgram, buildOptions);

    // save kernel source to disk
    error = save_kernel_source_to_disk(gCompilationCachePath, kernelNamePrefix,
                                       kernel);

    // save kernel build options to disk if exists
    if (buildOptions != NULL)
        error |= save_kernel_build_options_to_disk(
            gCompilationCachePath, kernelNamePrefix, buildOptions);

    return error;
}

static std::string
get_compilation_mode_str(const CompilationMode compilationMode)
{
    switch (compilationMode)
    {
        default: assert(0 && "Invalid compilation mode"); abort();
        case kOnline: return "online";
        case kBinary: return "binary";
        case kSpir_v: return "spir-v";
    }
}

static cl_int get_cl_device_info_str(const cl_device_id device,
                                     const cl_uint device_address_space_size,
                                     const CompilationMode compilationMode,
                                     std::string &clDeviceInfo)
{
    std::string extensionsString = get_device_extensions_string(device);
    std::string versionString = get_device_version_string(device);

    std::ostringstream clDeviceInfoStream;
    std::string file_type =
        get_offline_compilation_file_type_str(compilationMode);
    clDeviceInfoStream << "# OpenCL device info affecting " << file_type
                       << " offline compilation:" << std::endl
                       << "CL_DEVICE_ADDRESS_BITS=" << device_address_space_size
                       << std::endl
                       << "CL_DEVICE_EXTENSIONS=\"" << extensionsString << "\""
                       << std::endl;
    /* We only need the device's supported IL version(s) when compiling IL
     * that will be loaded with clCreateProgramWithIL() */
    if (compilationMode == kSpir_v)
    {
        std::string ilVersionString = get_device_il_version_string(device);
        clDeviceInfoStream << "CL_DEVICE_IL_VERSION=\"" << ilVersionString
                           << "\"" << std::endl;
    }
    clDeviceInfoStream << "CL_DEVICE_VERSION=\"" << versionString << "\""
                       << std::endl;
    clDeviceInfoStream << "CL_DEVICE_IMAGE_SUPPORT="
                       << (0 == checkForImageSupport(device)) << std::endl;
    clDeviceInfoStream << "CL_DEVICE_NAME=\"" << get_device_name(device).c_str()
                       << "\"" << std::endl;

    clDeviceInfo = clDeviceInfoStream.str();

    return CL_SUCCESS;
}

static int write_cl_device_info(const cl_device_id device,
                                const cl_uint device_address_space_size,
                                const CompilationMode compilationMode,
                                std::string &clDeviceInfoFilename)
{
    std::string clDeviceInfo;
    int error = get_cl_device_info_str(device, device_address_space_size,
                                       compilationMode, clDeviceInfo);
    if (error != CL_SUCCESS)
    {
        return error;
    }

    cl_uint crc = crc32(clDeviceInfo.data(), clDeviceInfo.size());

    /* Get the filename for the clDeviceInfo file.
     * Note: the file includes the hash on its content, so it is usually
     * unnecessary to delete it. */
    std::ostringstream clDeviceInfoFilenameStream;
    clDeviceInfoFilenameStream << gCompilationCachePath << slash
                               << "clDeviceInfo-";
    clDeviceInfoFilenameStream << std::hex << std::setfill('0') << std::setw(8)
                               << crc << ".txt";

    clDeviceInfoFilename = clDeviceInfoFilenameStream.str();

    if ((size_t)get_file_size(clDeviceInfoFilename) == clDeviceInfo.size())
    {
        /* The CL device info file has already been created.
         * Nothing to do. */
        return 0;
    }

    /* The file does not exist or its length is not as expected.
     * Create/overwrite it. */
    std::ofstream ofs(clDeviceInfoFilename);
    if (!ofs.good())
    {
        log_info("OfflineCompiler: can't create CL device info file: %s\n",
                 clDeviceInfoFilename.c_str());
        return -1;
    }
    ofs << clDeviceInfo;
    ofs.close();

    return CL_SUCCESS;
}

static std::string get_offline_compilation_command(
    const cl_uint device_address_space_size,
    const CompilationMode compilationMode, const std::string &bOptions,
    const std::string &sourceFilename, const std::string &outputFilename,
    const std::string &clDeviceInfoFilename)
{
    std::ostringstream wrapperOptions;

    wrapperOptions << gCompilationProgram
                   << " --mode=" << get_compilation_mode_str(compilationMode)
                   << " --source=" << sourceFilename
                   << " --output=" << outputFilename
                   << " --cl-device-info=" << clDeviceInfoFilename;

    if (bOptions != "")
    {
        // Add build options passed to this function
        wrapperOptions << " -- " << bOptions;
    }

    return wrapperOptions.str();
}

static int invoke_offline_compiler(const cl_device_id device,
                                   const cl_uint device_address_space_size,
                                   const CompilationMode compilationMode,
                                   const std::string &bOptions,
                                   const std::string &sourceFilename,
                                   const std::string &outputFilename)
{
    std::string runString;
    std::string clDeviceInfoFilename;

    // See cl_offline_compiler-interface.txt for a description of the
    // format of the CL device information file generated below, and
    // the internal command line interface for invoking the offline
    // compiler.

    cl_int err = write_cl_device_info(device, device_address_space_size,
                                      compilationMode, clDeviceInfoFilename);
    if (err != CL_SUCCESS)
    {
        log_error("Failed writing CL device info file\n");
        return err;
    }

    runString = get_offline_compilation_command(
        device_address_space_size, compilationMode, bOptions, sourceFilename,
        outputFilename, clDeviceInfoFilename);

    // execute script
    log_info("Executing command: %s\n", runString.c_str());
    fflush(stdout);
    int returnCode = system(runString.c_str());
    if (returnCode != 0)
    {
        log_error("ERROR: Command finished with error: 0x%x\n", returnCode);
        return CL_COMPILE_PROGRAM_FAILURE;
    }

    return CL_SUCCESS;
}

static cl_int get_first_device_id(const cl_context context,
                                  cl_device_id &device)
{
    cl_uint numDevices = 0;
    cl_int error = clGetContextInfo(context, CL_CONTEXT_NUM_DEVICES,
                                    sizeof(cl_uint), &numDevices, NULL);
    test_error(error, "clGetContextInfo failed getting CL_CONTEXT_NUM_DEVICES");

    if (numDevices == 0)
    {
        log_error("ERROR: No CL devices found\n");
        return -1;
    }

    std::vector<cl_device_id> devices(numDevices, 0);
    error =
        clGetContextInfo(context, CL_CONTEXT_DEVICES,
                         numDevices * sizeof(cl_device_id), &devices[0], NULL);
    test_error(error, "clGetContextInfo failed getting CL_CONTEXT_DEVICES");

    device = devices[0];
    return CL_SUCCESS;
}

static cl_int get_device_address_bits(const cl_device_id device,
                                      cl_uint &device_address_space_size)
{
    cl_int error =
        clGetDeviceInfo(device, CL_DEVICE_ADDRESS_BITS, sizeof(cl_uint),
                        &device_address_space_size, NULL);
    test_error(error, "Unable to obtain device address bits");

    if (device_address_space_size != 32 && device_address_space_size != 64)
    {
        log_error("ERROR: Unexpected number of device address bits: %u\n",
                  device_address_space_size);
        return -1;
    }

    return CL_SUCCESS;
}

static int get_offline_compiler_output(
    std::ifstream &ifs, const cl_device_id device, cl_uint deviceAddrSpaceSize,
    const CompilationMode compilationMode, const std::string &bOptions,
    const std::string &kernelPath, const std::string &kernelNamePrefix)
{
    std::string sourceFilename =
        get_cl_source_filename_with_path(kernelPath, kernelNamePrefix);
    std::string outputFilename = get_binary_filename_with_path(
        compilationMode, deviceAddrSpaceSize, kernelPath, kernelNamePrefix);

    ifs.open(outputFilename.c_str(), std::ios::binary);
    if (!ifs.good())
    {
        std::string file_type =
            get_offline_compilation_file_type_str(compilationMode);
        if (gCompilationCacheMode == kCacheModeForceRead)
        {
            log_info("OfflineCompiler: can't open cached %s file: %s\n",
                     file_type.c_str(), outputFilename.c_str());
            return -1;
        }
        else
        {
            int error = invoke_offline_compiler(device, deviceAddrSpaceSize,
                                                compilationMode, bOptions,
                                                sourceFilename, outputFilename);
            if (error != CL_SUCCESS) return error;

            // open output file for reading
            ifs.open(outputFilename.c_str(), std::ios::binary);
            if (!ifs.good())
            {
                log_info("OfflineCompiler: can't read generated %s file: %s\n",
                         file_type.c_str(), outputFilename.c_str());
                return -1;
            }
        }
    }

    if (compilationMode == kSpir_v && !gDisableSPIRVValidation)
    {
        std::string runString = gSPIRVValidator + " " + outputFilename;

        int returnCode = system(runString.c_str());
        if (returnCode == -1)
        {
            log_error("Error: failed to invoke SPIR-V validator\n");
            return CL_COMPILE_PROGRAM_FAILURE;
        }
        else if (returnCode != 0)
        {
            log_error(
                "Failed to validate SPIR-V file %s: system() returned 0x%x\n",
                outputFilename.c_str(), returnCode);
            return CL_COMPILE_PROGRAM_FAILURE;
        }
    }

    return CL_SUCCESS;
}

static int create_single_kernel_helper_create_program_offline(
    cl_context context, cl_device_id device, cl_program *outProgram,
    unsigned int numKernelLines, const char *const *kernelProgram,
    const char *buildOptions, CompilationMode compilationMode)
{
    if (kCacheModeDumpCl == gCompilationCacheMode)
    {
        return -1;
    }

    // Get device CL_DEVICE_ADDRESS_BITS
    int error;
    cl_uint device_address_space_size = 0;
    if (device == NULL)
    {
        error = get_first_device_id(context, device);
        test_error(error, "Failed to get device ID for first device");
    }
    error = get_device_address_bits(device, device_address_space_size);
    if (error != CL_SUCCESS) return error;

    // set build options
    std::string bOptions;
    bOptions += buildOptions ? std::string(buildOptions) : "";

    std::string kernelName =
        get_unique_filename_prefix(numKernelLines, kernelProgram, buildOptions);


    std::ifstream ifs;
    error = get_offline_compiler_output(ifs, device, device_address_space_size,
                                        compilationMode, bOptions,
                                        gCompilationCachePath, kernelName);
    if (error != CL_SUCCESS) return error;

    ifs.seekg(0, ifs.end);
    size_t length = static_cast<size_t>(ifs.tellg());
    ifs.seekg(0, ifs.beg);

    // treat modifiedProgram as input for clCreateProgramWithBinary
    if (compilationMode == kBinary)
    {
        // read binary from file:
        std::vector<unsigned char> modifiedKernelBuf(length);

        ifs.read((char *)&modifiedKernelBuf[0], length);
        ifs.close();

        size_t lengths = modifiedKernelBuf.size();
        const unsigned char *binaries = { &modifiedKernelBuf[0] };
        log_info("offlineCompiler: clCreateProgramWithSource replaced with "
                 "clCreateProgramWithBinary\n");
        *outProgram = clCreateProgramWithBinary(context, 1, &device, &lengths,
                                                &binaries, NULL, &error);
        if (*outProgram == NULL || error != CL_SUCCESS)
        {
            print_error(error, "clCreateProgramWithBinary failed");
            return error;
        }
    }
    // treat modifiedProgram as input for clCreateProgramWithIL
    else if (compilationMode == kSpir_v)
    {
        // read spir-v from file:
        std::vector<unsigned char> modifiedKernelBuf(length);

        ifs.read((char *)&modifiedKernelBuf[0], length);
        ifs.close();

        size_t length = modifiedKernelBuf.size();
        log_info("offlineCompiler: clCreateProgramWithSource replaced with "
                 "clCreateProgramWithIL\n");
        if (gCoreILProgram)
        {
            *outProgram = clCreateProgramWithIL(context, &modifiedKernelBuf[0],
                                                length, &error);
        }
        else
        {
            cl_platform_id platform;
            error = clGetDeviceInfo(device, CL_DEVICE_PLATFORM,
                                    sizeof(cl_platform_id), &platform, NULL);
            test_error(error, "clGetDeviceInfo for CL_DEVICE_PLATFORM failed");

            clCreateProgramWithILKHR_fn clCreateProgramWithILKHR = NULL;
            clCreateProgramWithILKHR = (clCreateProgramWithILKHR_fn)
                clGetExtensionFunctionAddressForPlatform(
                    platform, "clCreateProgramWithILKHR");
            if (clCreateProgramWithILKHR == NULL)
            {
                log_error(
                    "ERROR: clGetExtensionFunctionAddressForPlatform failed\n");
                return -1;
            }
            *outProgram = clCreateProgramWithILKHR(
                context, &modifiedKernelBuf[0], length, &error);
        }

        if (*outProgram == NULL || error != CL_SUCCESS)
        {
            if (gCoreILProgram)
            {
                print_error(error, "clCreateProgramWithIL failed");
            }
            else
            {
                print_error(error, "clCreateProgramWithILKHR failed");
            }
            return error;
        }
    }

    return CL_SUCCESS;
}

static int create_single_kernel_helper_create_program(
    cl_context context, cl_device_id device, cl_program *outProgram,
    unsigned int numKernelLines, const char **kernelProgram,
    const char *buildOptions, CompilationMode compilationMode)
{
    std::lock_guard<std::mutex> compiler_lock(gCompilerMutex);

    std::string filePrefix =
        get_unique_filename_prefix(numKernelLines, kernelProgram, buildOptions);
    bool shouldSaveToDisk = should_save_kernel_source_to_disk(
        compilationMode, gCompilationCacheMode, gCompilationCachePath,
        filePrefix);

    if (shouldSaveToDisk)
    {
        if (CL_SUCCESS
            != save_kernel_source_and_options_to_disk(
                numKernelLines, kernelProgram, buildOptions))
        {
            log_error("Unable to dump kernel source to disk");
            return -1;
        }
    }
    if (compilationMode == kOnline)
    {
        int error = CL_SUCCESS;

        /* Create the program object from source */
        *outProgram = clCreateProgramWithSource(context, numKernelLines,
                                                kernelProgram, NULL, &error);
        if (*outProgram == NULL || error != CL_SUCCESS)
        {
            print_error(error, "clCreateProgramWithSource failed");
            return error;
        }
        return CL_SUCCESS;
    }
    else
    {
        return create_single_kernel_helper_create_program_offline(
            context, device, outProgram, numKernelLines, kernelProgram,
            buildOptions, compilationMode);
    }
}

int create_single_kernel_helper_create_program(cl_context context,
                                               cl_program *outProgram,
                                               unsigned int numKernelLines,
                                               const char **kernelProgram,
                                               const char *buildOptions)
{
    return create_single_kernel_helper_create_program(
        context, NULL, outProgram, numKernelLines, kernelProgram, buildOptions,
        gCompilationMode);
}

int create_single_kernel_helper_create_program_for_device(
    cl_context context, cl_device_id device, cl_program *outProgram,
    unsigned int numKernelLines, const char **kernelProgram,
    const char *buildOptions)
{
    return create_single_kernel_helper_create_program(
        context, device, outProgram, numKernelLines, kernelProgram,
        buildOptions, gCompilationMode);
}

int create_single_kernel_helper_with_build_options(
    cl_context context, cl_program *outProgram, cl_kernel *outKernel,
    unsigned int numKernelLines, const char **kernelProgram,
    const char *kernelName, const char *buildOptions)
{
    return create_single_kernel_helper(context, outProgram, outKernel,
                                       numKernelLines, kernelProgram,
                                       kernelName, buildOptions);
}

// Creates and builds OpenCL C/C++ program, and creates a kernel
int create_single_kernel_helper(cl_context context, cl_program *outProgram,
                                cl_kernel *outKernel,
                                unsigned int numKernelLines,
                                const char **kernelProgram,
                                const char *kernelName,
                                const char *buildOptions)
{
    // For the logic that automatically adds -cl-std it is much cleaner if the
    // build options have RAII. This buffer will store the potentially updated
    // build options, in which case buildOptions will point at the string owned
    // by this buffer.
    std::string build_options_internal{ buildOptions ? buildOptions : "" };

    // Check the build options for the -cl-std option.
    if (!buildOptions || !strstr(buildOptions, "-cl-std"))
    {
        // If the build option isn't present add it using the latest OpenCL-C
        // version supported by the device. This allows calling code to force a
        // particular CL C version if it is required, but also means that
        // callers need not specify a version if they want to assume the most
        // recent CL C.

        auto version = get_max_OpenCL_C_for_context(context);

        std::string cl_std{};
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
            // If the -cl-std build option is not specified, the highest OpenCL
            // C 1.x language version supported by each device is used when
            // compiling the program for each device.
            cl_std = "";
        }
        build_options_internal += ' ';
        build_options_internal += cl_std;
        buildOptions = build_options_internal.c_str();
    }
    int error = create_single_kernel_helper_create_program(
        context, outProgram, numKernelLines, kernelProgram, buildOptions);
    if (error != CL_SUCCESS)
    {
        log_error("Create program failed: %d, line: %d\n", error, __LINE__);
        return error;
    }

    // Remove offline-compiler-only build options
    std::string newBuildOptions;
    if (buildOptions != NULL)
    {
        newBuildOptions = buildOptions;
        std::string offlineCompierOptions[] = {
            "-cl-fp16-enable", "-cl-fp64-enable", "-cl-zero-init-local-mem-vars"
        };
        for (auto &s : offlineCompierOptions)
        {
            std::string::size_type i = newBuildOptions.find(s);
            if (i != std::string::npos) newBuildOptions.erase(i, s.length());
        }
    }
    // Build program and create kernel
    return build_program_create_kernel_helper(
        context, outProgram, outKernel, numKernelLines, kernelProgram,
        kernelName, newBuildOptions.c_str());
}

// Builds OpenCL C/C++ program and creates
int build_program_create_kernel_helper(
    cl_context context, cl_program *outProgram, cl_kernel *outKernel,
    unsigned int numKernelLines, const char **kernelProgram,
    const char *kernelName, const char *buildOptions)
{
    int error;
    /* Compile the program */
    int buildProgramFailed = 0;
    int printedSource = 0;
    error = clBuildProgram(*outProgram, 0, NULL, buildOptions, NULL, NULL);
    if (error != CL_SUCCESS)
    {
        unsigned int i;
        print_error(error, "clBuildProgram failed");
        buildProgramFailed = 1;
        printedSource = 1;
        log_error("Build options: %s\n", buildOptions);
        log_error("Original source is: ------------\n");
        for (i = 0; i < numKernelLines; i++) log_error("%s", kernelProgram[i]);
    }

    // Verify the build status on all devices
    cl_uint deviceCount = 0;
    error = clGetProgramInfo(*outProgram, CL_PROGRAM_NUM_DEVICES,
                             sizeof(deviceCount), &deviceCount, NULL);
    if (error != CL_SUCCESS)
    {
        print_error(error, "clGetProgramInfo CL_PROGRAM_NUM_DEVICES failed");
        return error;
    }

    if (deviceCount == 0)
    {
        log_error("No devices found for program.\n");
        return -1;
    }

    cl_device_id *devices =
        (cl_device_id *)malloc(deviceCount * sizeof(cl_device_id));
    if (NULL == devices) return -1;
    BufferOwningPtr<cl_device_id> devicesBuf(devices);

    memset(devices, 0, deviceCount * sizeof(cl_device_id));
    error = clGetProgramInfo(*outProgram, CL_PROGRAM_DEVICES,
                             sizeof(cl_device_id) * deviceCount, devices, NULL);
    if (error != CL_SUCCESS)
    {
        print_error(error, "clGetProgramInfo CL_PROGRAM_DEVICES failed");
        return error;
    }

    cl_uint z;
    bool buildFailed = false;
    for (z = 0; z < deviceCount; z++)
    {
        char deviceName[4096] = "";
        error = clGetDeviceInfo(devices[z], CL_DEVICE_NAME, sizeof(deviceName),
                                deviceName, NULL);
        if (error != CL_SUCCESS || deviceName[0] == '\0')
        {
            log_error("Device \"%d\" failed to return a name\n", z);
            print_error(error, "clGetDeviceInfo CL_DEVICE_NAME failed");
        }

        cl_build_status buildStatus;
        error = clGetProgramBuildInfo(*outProgram, devices[z],
                                      CL_PROGRAM_BUILD_STATUS,
                                      sizeof(buildStatus), &buildStatus, NULL);
        if (error != CL_SUCCESS)
        {
            print_error(error,
                        "clGetProgramBuildInfo CL_PROGRAM_BUILD_STATUS failed");
            return error;
        }

        if (buildStatus == CL_BUILD_SUCCESS && buildProgramFailed
            && deviceCount == 1)
        {
            buildFailed = true;
            log_error("clBuildProgram returned an error, but buildStatus is "
                      "marked as CL_BUILD_SUCCESS.\n");
        }

        if (buildStatus != CL_BUILD_SUCCESS)
        {

            char statusString[64] = "";
            if (buildStatus == (cl_build_status)CL_BUILD_SUCCESS)
                sprintf(statusString, "CL_BUILD_SUCCESS");
            else if (buildStatus == (cl_build_status)CL_BUILD_NONE)
                sprintf(statusString, "CL_BUILD_NONE");
            else if (buildStatus == (cl_build_status)CL_BUILD_ERROR)
                sprintf(statusString, "CL_BUILD_ERROR");
            else if (buildStatus == (cl_build_status)CL_BUILD_IN_PROGRESS)
                sprintf(statusString, "CL_BUILD_IN_PROGRESS");
            else
                sprintf(statusString, "UNKNOWN (%d)", buildStatus);

            if (buildStatus != CL_BUILD_SUCCESS)
                log_error(
                    "Build not successful for device \"%s\", status: %s\n",
                    deviceName, statusString);
            size_t paramSize = 0;
            error = clGetProgramBuildInfo(*outProgram, devices[z],
                                          CL_PROGRAM_BUILD_LOG, 0, NULL,
                                          &paramSize);
            if (error != CL_SUCCESS)
            {

                print_error(
                    error, "clGetProgramBuildInfo CL_PROGRAM_BUILD_LOG failed");
                return error;
            }

            std::string log;
            log.resize(paramSize / sizeof(char));
            error = clGetProgramBuildInfo(*outProgram, devices[z],
                                          CL_PROGRAM_BUILD_LOG, paramSize,
                                          &log[0], NULL);
            if (error != CL_SUCCESS || log[0] == '\0')
            {
                log_error("Device %d (%s) failed to return a build log\n", z,
                          deviceName);
                if (error)
                {
                    print_error(
                        error,
                        "clGetProgramBuildInfo CL_PROGRAM_BUILD_LOG failed");
                    return error;
                }
                else
                {
                    log_error("clGetProgramBuildInfo returned an empty log.\n");
                    return -1;
                }
            }
            // In this case we've already printed out the code above.
            if (!printedSource)
            {
                unsigned int i;
                log_error("Original source is: ------------\n");
                for (i = 0; i < numKernelLines; i++)
                    log_error("%s", kernelProgram[i]);
                printedSource = 1;
            }
            log_error("Build log for device \"%s\" is: ------------\n",
                      deviceName);
            log_error("%s\n", log.c_str());
            log_error("\n----------\n");
            return -1;
        }
    }

    if (buildFailed)
    {
        return -1;
    }

    /* And create a kernel from it */
    if (kernelName != NULL)
    {
        *outKernel = clCreateKernel(*outProgram, kernelName, &error);
        if (*outKernel == NULL || error != CL_SUCCESS)
        {
            print_error(error, "Unable to create kernel");
            return error;
        }
    }

    return 0;
}

int get_max_allowed_work_group_size(cl_context context, cl_kernel kernel,
                                    size_t *outMaxSize, size_t *outLimits)
{
    cl_device_id *devices;
    size_t size, maxCommonSize = 0;
    int numDevices, i, j, error;
    cl_uint numDims;
    size_t outSize;
    size_t sizeLimit[] = { 1, 1, 1 };


    /* Assume fewer than 16 devices will be returned */
    error = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &outSize);
    test_error(error, "Unable to obtain list of devices size for context");
    devices = (cl_device_id *)malloc(outSize);
    BufferOwningPtr<cl_device_id> devicesBuf(devices);

    error =
        clGetContextInfo(context, CL_CONTEXT_DEVICES, outSize, devices, NULL);
    test_error(error, "Unable to obtain list of devices for context");

    numDevices = (int)(outSize / sizeof(cl_device_id));

    for (i = 0; i < numDevices; i++)
    {
        error = clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_GROUP_SIZE,
                                sizeof(size), &size, NULL);
        test_error(error, "Unable to obtain max work group size for device");
        if (size < maxCommonSize || maxCommonSize == 0) maxCommonSize = size;

        error = clGetKernelWorkGroupInfo(kernel, devices[i],
                                         CL_KERNEL_WORK_GROUP_SIZE,
                                         sizeof(size), &size, NULL);
        test_error(
            error,
            "Unable to obtain max work group size for device and kernel combo");
        if (size < maxCommonSize || maxCommonSize == 0) maxCommonSize = size;

        error = clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                                sizeof(numDims), &numDims, NULL);
        test_error(
            error,
            "clGetDeviceInfo failed for CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS");
        sizeLimit[0] = 1;
        error = clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_ITEM_SIZES,
                                numDims * sizeof(size_t), sizeLimit, NULL);
        test_error(error,
                   "clGetDeviceInfo failed for CL_DEVICE_MAX_WORK_ITEM_SIZES");

        if (outLimits != NULL)
        {
            if (i == 0)
            {
                for (j = 0; j < 3; j++) outLimits[j] = sizeLimit[j];
            }
            else
            {
                for (j = 0; j < (int)numDims; j++)
                {
                    if (sizeLimit[j] < outLimits[j])
                        outLimits[j] = sizeLimit[j];
                }
            }
        }
    }

    *outMaxSize = (unsigned int)maxCommonSize;
    return 0;
}


extern int get_max_allowed_1d_work_group_size_on_device(cl_device_id device,
                                                        cl_kernel kernel,
                                                        size_t *outSize)
{
    cl_uint maxDim;
    size_t maxWgSize;
    size_t *maxWgSizePerDim;
    int error;

    error = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE,
                                     sizeof(size_t), &maxWgSize, NULL);
    test_error(error,
               "clGetKernelWorkGroupInfo CL_KERNEL_WORK_GROUP_SIZE failed");

    error = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                            sizeof(cl_uint), &maxDim, NULL);
    test_error(error,
               "clGetDeviceInfo CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS failed");
    maxWgSizePerDim = (size_t *)malloc(maxDim * sizeof(size_t));
    if (!maxWgSizePerDim)
    {
        log_error("Unable to allocate maxWgSizePerDim\n");
        return -1;
    }

    error = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                            maxDim * sizeof(size_t), maxWgSizePerDim, NULL);
    if (error != CL_SUCCESS)
    {
        log_error("clGetDeviceInfo CL_DEVICE_MAX_WORK_ITEM_SIZES failed\n");
        free(maxWgSizePerDim);
        return error;
    }

    // "maxWgSize" is limited to that of the first dimension.
    if (maxWgSize > maxWgSizePerDim[0])
    {
        maxWgSize = maxWgSizePerDim[0];
    }

    free(maxWgSizePerDim);

    *outSize = maxWgSize;
    return 0;
}


int get_max_common_work_group_size(cl_context context, cl_kernel kernel,
                                   size_t globalThreadSize, size_t *outMaxSize)
{
    size_t sizeLimit[3];
    int error =
        get_max_allowed_work_group_size(context, kernel, outMaxSize, sizeLimit);
    if (error != 0) return error;

    /* Now find the largest factor of globalThreadSize that is <= maxCommonSize
     */
    /* Note for speed, we don't need to check the range of maxCommonSize, b/c
     once it gets to 1, the modulo test will succeed and break the loop anyway
   */
    for (;
         (globalThreadSize % *outMaxSize) != 0 || (*outMaxSize > sizeLimit[0]);
         (*outMaxSize)--)
        ;
    return 0;
}

int get_max_common_2D_work_group_size(cl_context context, cl_kernel kernel,
                                      size_t *globalThreadSizes,
                                      size_t *outMaxSizes)
{
    size_t sizeLimit[3];
    size_t maxSize;
    int error =
        get_max_allowed_work_group_size(context, kernel, &maxSize, sizeLimit);
    if (error != 0) return error;

    /* Now find a set of factors, multiplied together less than maxSize, but
       each a factor of the global sizes */

    /* Simple case */
    if (globalThreadSizes[0] * globalThreadSizes[1] <= maxSize)
    {
        if (globalThreadSizes[0] <= sizeLimit[0]
            && globalThreadSizes[1] <= sizeLimit[1])
        {
            outMaxSizes[0] = globalThreadSizes[0];
            outMaxSizes[1] = globalThreadSizes[1];
            return 0;
        }
    }

    size_t remainingSize, sizeForThisOne;
    remainingSize = maxSize;
    int i, j;
    for (i = 0; i < 2; i++)
    {
        if (globalThreadSizes[i] > remainingSize)
            sizeForThisOne = remainingSize;
        else
            sizeForThisOne = globalThreadSizes[i];
        for (; (globalThreadSizes[i] % sizeForThisOne) != 0
             || (sizeForThisOne > sizeLimit[i]);
             sizeForThisOne--)
            ;
        outMaxSizes[i] = sizeForThisOne;
        remainingSize = maxSize;
        for (j = 0; j <= i; j++) remainingSize /= outMaxSizes[j];
    }

    return 0;
}

int get_max_common_3D_work_group_size(cl_context context, cl_kernel kernel,
                                      size_t *globalThreadSizes,
                                      size_t *outMaxSizes)
{
    size_t sizeLimit[3];
    size_t maxSize;
    int error =
        get_max_allowed_work_group_size(context, kernel, &maxSize, sizeLimit);
    if (error != 0) return error;
    /* Now find a set of factors, multiplied together less than maxSize, but
     each a factor of the global sizes */

    /* Simple case */
    if (globalThreadSizes[0] * globalThreadSizes[1] * globalThreadSizes[2]
        <= maxSize)
    {
        if (globalThreadSizes[0] <= sizeLimit[0]
            && globalThreadSizes[1] <= sizeLimit[1]
            && globalThreadSizes[2] <= sizeLimit[2])
        {
            outMaxSizes[0] = globalThreadSizes[0];
            outMaxSizes[1] = globalThreadSizes[1];
            outMaxSizes[2] = globalThreadSizes[2];
            return 0;
        }
    }

    size_t remainingSize, sizeForThisOne;
    remainingSize = maxSize;
    int i, j;
    for (i = 0; i < 3; i++)
    {
        if (globalThreadSizes[i] > remainingSize)
            sizeForThisOne = remainingSize;
        else
            sizeForThisOne = globalThreadSizes[i];
        for (; (globalThreadSizes[i] % sizeForThisOne) != 0
             || (sizeForThisOne > sizeLimit[i]);
             sizeForThisOne--)
            ;
        outMaxSizes[i] = sizeForThisOne;
        remainingSize = maxSize;
        for (j = 0; j <= i; j++) remainingSize /= outMaxSizes[j];
    }

    return 0;
}

/* Helper to determine if a device supports an image format */
int is_image_format_supported(cl_context context, cl_mem_flags flags,
                              cl_mem_object_type image_type,
                              const cl_image_format *fmt)
{
    cl_image_format *list;
    cl_uint count = 0;
    cl_int err = clGetSupportedImageFormats(context, flags, image_type, 128,
                                            NULL, &count);
    if (count == 0) return 0;

    list = (cl_image_format *)malloc(count * sizeof(cl_image_format));
    if (NULL == list)
    {
        log_error("Error: unable to allocate %zu byte buffer for image format "
                  "list at %s:%d (err = %d)\n",
                  count * sizeof(cl_image_format), __FILE__, __LINE__, err);
        return 0;
    }
    BufferOwningPtr<cl_image_format> listBuf(list);


    cl_int error = clGetSupportedImageFormats(context, flags, image_type, count,
                                              list, NULL);
    if (error)
    {
        log_error("Error: failed to obtain supported image type list at %s:%d "
                  "(err = %d)\n",
                  __FILE__, __LINE__, err);
        return 0;
    }

    // iterate looking for a match.
    cl_uint i;
    for (i = 0; i < count; i++)
    {
        if (fmt->image_channel_data_type == list[i].image_channel_data_type
            && fmt->image_channel_order == list[i].image_channel_order)
            break;
    }

    return (i < count) ? 1 : 0;
}

size_t get_pixel_bytes(const cl_image_format *fmt);
size_t get_pixel_bytes(const cl_image_format *fmt)
{
    size_t chanCount;
    switch (fmt->image_channel_order)
    {
        case CL_R:
        case CL_A:
        case CL_Rx:
        case CL_INTENSITY:
        case CL_LUMINANCE:
        case CL_DEPTH: chanCount = 1; break;
        case CL_RG:
        case CL_RA:
        case CL_RGx: chanCount = 2; break;
        case CL_RGB:
        case CL_RGBx:
        case CL_sRGB:
        case CL_sRGBx: chanCount = 3; break;
        case CL_RGBA:
        case CL_ARGB:
        case CL_BGRA:
        case CL_sBGRA:
        case CL_sRGBA:
#ifdef CL_1RGB_APPLE
        case CL_1RGB_APPLE:
#endif
#ifdef CL_BGR1_APPLE
        case CL_BGR1_APPLE:
#endif
            chanCount = 4;
            break;
        default:
            log_error("Unknown channel order at %s:%d!\n", __FILE__, __LINE__);
            abort();
            break;
    }

    switch (fmt->image_channel_data_type)
    {
        case CL_UNORM_SHORT_565:
        case CL_UNORM_SHORT_555: return 2;

        case CL_UNORM_INT_101010: return 4;

        case CL_SNORM_INT8:
        case CL_UNORM_INT8:
        case CL_SIGNED_INT8:
        case CL_UNSIGNED_INT8: return chanCount;

        case CL_SNORM_INT16:
        case CL_UNORM_INT16:
        case CL_HALF_FLOAT:
        case CL_SIGNED_INT16:
        case CL_UNSIGNED_INT16:
#ifdef CL_SFIXED14_APPLE
        case CL_SFIXED14_APPLE:
#endif
            return chanCount * 2;

        case CL_SIGNED_INT32:
        case CL_UNSIGNED_INT32:
        case CL_FLOAT: return chanCount * 4;

        default:
            log_error("Unknown channel data type at %s:%d!\n", __FILE__,
                      __LINE__);
            abort();
    }

    return 0;
}

test_status verifyImageSupport(cl_device_id device)
{
    int result = checkForImageSupport(device);
    if (result == 0)
    {
        return TEST_PASS;
    }
    if (result == CL_IMAGE_FORMAT_NOT_SUPPORTED)
    {
        log_error("SKIPPED: Device does not supported images as required by "
                  "this test!\n");
        return TEST_SKIP;
    }
    return TEST_FAIL;
}

int checkForImageSupport(cl_device_id device)
{
    cl_uint i;
    int error;


    /* Check the device props to see if images are supported at all first */
    error =
        clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT, sizeof(i), &i, NULL);
    test_error(error, "Unable to query device for image support");
    if (i == 0)
    {
        return CL_IMAGE_FORMAT_NOT_SUPPORTED;
    }

    /* So our support is good */
    return 0;
}

int checkFor3DImageSupport(cl_device_id device)
{
    cl_uint i;
    int error;

    /* Check the device props to see if images are supported at all first */
    error =
        clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT, sizeof(i), &i, NULL);
    test_error(error, "Unable to query device for image support");
    if (i == 0)
    {
        return CL_IMAGE_FORMAT_NOT_SUPPORTED;
    }

    char profile[128];
    error = clGetDeviceInfo(device, CL_DEVICE_PROFILE, sizeof(profile), profile,
                            NULL);
    test_error(error, "Unable to query device for CL_DEVICE_PROFILE");
    if (0 == strcmp(profile, "EMBEDDED_PROFILE"))
    {
        size_t width = -1L;
        size_t height = -1L;
        size_t depth = -1L;
        error = clGetDeviceInfo(device, CL_DEVICE_IMAGE3D_MAX_WIDTH,
                                sizeof(width), &width, NULL);
        test_error(error, "Unable to get CL_DEVICE_IMAGE3D_MAX_WIDTH");
        error = clGetDeviceInfo(device, CL_DEVICE_IMAGE3D_MAX_HEIGHT,
                                sizeof(height), &height, NULL);
        test_error(error, "Unable to get CL_DEVICE_IMAGE3D_MAX_HEIGHT");
        error = clGetDeviceInfo(device, CL_DEVICE_IMAGE3D_MAX_DEPTH,
                                sizeof(depth), &depth, NULL);
        test_error(error, "Unable to get CL_DEVICE_IMAGE3D_MAX_DEPTH");

        if (0 == (height | width | depth)) return CL_IMAGE_FORMAT_NOT_SUPPORTED;
    }

    /* So our support is good */
    return 0;
}

int checkForReadWriteImageSupport(cl_device_id device)
{
    if (checkForImageSupport(device))
    {
        return CL_IMAGE_FORMAT_NOT_SUPPORTED;
    }

    auto device_cl_version = get_device_cl_version(device);
    if (device_cl_version >= Version(3, 0))
    {
        // In OpenCL 3.0, Read-Write images are optional.
        // Check if they are supported.
        cl_uint are_rw_images_supported{};
        test_error(
            clGetDeviceInfo(device, CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS,
                            sizeof(are_rw_images_supported),
                            &are_rw_images_supported, nullptr),
            "clGetDeviceInfo failed for CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS\n");
        if (0 == are_rw_images_supported)
        {
            log_info("READ_WRITE_IMAGE tests skipped, not supported.\n");
            return CL_IMAGE_FORMAT_NOT_SUPPORTED;
        }
    }
    // READ_WRITE images are not supported on 1.X devices.
    else if (device_cl_version < Version(2, 0))
    {
        log_info("READ_WRITE_IMAGE tests skipped, Opencl 2.0+ is requried.");
        return CL_IMAGE_FORMAT_NOT_SUPPORTED;
    }
    // Support for read-write image arguments is required
    // for an 2.X device if the device supports images.

    /* So our support is good */
    return 0;
}

size_t get_min_alignment(cl_context context)
{
    static cl_uint align_size = 0;

    if (0 == align_size)
    {
        cl_device_id *devices;
        size_t devices_size = 0;
        cl_uint result = 0;
        cl_int error;
        int i;

        error = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL,
                                 &devices_size);
        test_error_ret(error, "clGetContextInfo failed", 0);

        devices = (cl_device_id *)malloc(devices_size);
        if (devices == NULL)
        {
            print_error(error, "malloc failed");
            return 0;
        }

        error = clGetContextInfo(context, CL_CONTEXT_DEVICES, devices_size,
                                 (void *)devices, NULL);
        test_error_ret(error, "clGetContextInfo failed", 0);

        for (i = 0; i < (int)(devices_size / sizeof(cl_device_id)); i++)
        {
            cl_uint alignment = 0;

            error = clGetDeviceInfo(devices[i], CL_DEVICE_MEM_BASE_ADDR_ALIGN,
                                    sizeof(cl_uint), (void *)&alignment, NULL);

            if (error == CL_SUCCESS)
            {
                alignment >>= 3; // convert bits to bytes
                result = (alignment > result) ? alignment : result;
            }
            else
                print_error(error, "clGetDeviceInfo failed");
        }

        align_size = result;
        free(devices);
    }

    return align_size;
}

cl_device_fp_config get_default_rounding_mode(cl_device_id device)
{
    char profileStr[128] = "";
    cl_device_fp_config single = 0;
    int error = clGetDeviceInfo(device, CL_DEVICE_SINGLE_FP_CONFIG,
                                sizeof(single), &single, NULL);
    if (error)
        test_error_ret(error, "Unable to get device CL_DEVICE_SINGLE_FP_CONFIG",
                       0);

    if (single & CL_FP_ROUND_TO_NEAREST) return CL_FP_ROUND_TO_NEAREST;

    if (0 == (single & CL_FP_ROUND_TO_ZERO))
        test_error_ret(-1,
                       "FAILURE: device must support either "
                       "CL_DEVICE_SINGLE_FP_CONFIG or CL_FP_ROUND_TO_NEAREST",
                       0);

    // Make sure we are an embedded device before allowing a pass
    if ((error = clGetDeviceInfo(device, CL_DEVICE_PROFILE, sizeof(profileStr),
                                 &profileStr, NULL)))
        test_error_ret(error, "FAILURE: Unable to get CL_DEVICE_PROFILE", 0);

    if (strcmp(profileStr, "EMBEDDED_PROFILE"))
        test_error_ret(error,
                       "FAILURE: non-EMBEDDED_PROFILE devices must support "
                       "CL_FP_ROUND_TO_NEAREST",
                       0);

    return CL_FP_ROUND_TO_ZERO;
}

int checkDeviceForQueueSupport(cl_device_id device,
                               cl_command_queue_properties prop)
{
    cl_command_queue_properties realProps;
    cl_int error = clGetDeviceInfo(device, CL_DEVICE_QUEUE_ON_HOST_PROPERTIES,
                                   sizeof(realProps), &realProps, NULL);
    test_error_ret(error, "FAILURE: Unable to get device queue properties", 0);

    return (realProps & prop) ? 1 : 0;
}

int printDeviceHeader(cl_device_id device)
{
    char deviceName[512], deviceVendor[512], deviceVersion[512],
        cLangVersion[512];
    int error;

    error = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName),
                            deviceName, NULL);
    test_error(error, "Unable to get CL_DEVICE_NAME for device");

    error = clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(deviceVendor),
                            deviceVendor, NULL);
    test_error(error, "Unable to get CL_DEVICE_VENDOR for device");

    error = clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(deviceVersion),
                            deviceVersion, NULL);
    test_error(error, "Unable to get CL_DEVICE_VERSION for device");

    error = clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION,
                            sizeof(cLangVersion), cLangVersion, NULL);
    test_error(error, "Unable to get CL_DEVICE_OPENCL_C_VERSION for device");

    log_info("Compute Device Name = %s, Compute Device Vendor = %s, Compute "
             "Device Version = %s%s%s\n",
             deviceName, deviceVendor, deviceVersion,
             (error == CL_SUCCESS) ? ", CL C Version = " : "",
             (error == CL_SUCCESS) ? cLangVersion : "");

    auto version = get_device_cl_version(device);
    if (version >= Version(3, 0))
    {
        auto ctsVersion = get_device_info_string(
            device, CL_DEVICE_LATEST_CONFORMANCE_VERSION_PASSED);
        log_info("Device latest conformance version passed: %s\n",
                 ctsVersion.c_str());
    }

    return CL_SUCCESS;
}

Version get_device_cl_c_version(cl_device_id device)
{
    auto device_cl_version = get_device_cl_version(device);

    // The second special case is OpenCL-1.0 where CL_DEVICE_OPENCL_C_VERSION
    // did not exist, but since this is just the first version we can
    // return 1.0.
    if (device_cl_version == Version{ 1, 0 })
    {
        return Version{ 1, 0 };
    }

    // Otherwise we know we have a 1.1 <= device_version <= 2.0 where all CL C
    // versions are backwards compatible, hence querying with the
    // CL_DEVICE_OPENCL_C_VERSION query must return the most recent supported
    // OpenCL C version.
    size_t opencl_c_version_size_in_bytes{};
    auto error = clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, 0, nullptr,
                                 &opencl_c_version_size_in_bytes);
    test_error_ret(error,
                   "clGetDeviceInfo failed for CL_DEVICE_OPENCL_C_VERSION\n",
                   (Version{ -1, 0 }));

    std::string opencl_c_version(opencl_c_version_size_in_bytes, '\0');
    error =
        clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION,
                        opencl_c_version.size(), &opencl_c_version[0], nullptr);

    test_error_ret(error,
                   "clGetDeviceInfo failed for CL_DEVICE_OPENCL_C_VERSION\n",
                   (Version{ -1, 0 }));

    // Scrape out the major, minor pair from the string.
    auto major = opencl_c_version[opencl_c_version.find('.') - 1];
    auto minor = opencl_c_version[opencl_c_version.find('.') + 1];

    return Version{ major - '0', minor - '0' };
}

Version get_device_latest_cl_c_version(cl_device_id device)
{
    auto device_cl_version = get_device_cl_version(device);

    // If the device version >= 3.0 it must support the
    // CL_DEVICE_OPENCL_C_ALL_VERSIONS query from which we can extract the most
    // recent CL C version supported by the device.
    if (device_cl_version >= Version{ 3, 0 })
    {
        size_t opencl_c_all_versions_size_in_bytes{};
        auto error =
            clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_ALL_VERSIONS, 0, nullptr,
                            &opencl_c_all_versions_size_in_bytes);
        test_error_ret(
            error, "clGetDeviceInfo failed for CL_DEVICE_OPENCL_C_ALL_VERSIONS",
            (Version{ -1, 0 }));
        std::vector<cl_name_version> name_versions(
            opencl_c_all_versions_size_in_bytes / sizeof(cl_name_version));
        error = clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_ALL_VERSIONS,
                                opencl_c_all_versions_size_in_bytes,
                                name_versions.data(), nullptr);
        test_error_ret(
            error, "clGetDeviceInfo failed for CL_DEVICE_OPENCL_C_ALL_VERSIONS",
            (Version{ -1, 0 }));

        Version max_supported_cl_c_version{};
        for (const auto &name_version : name_versions)
        {
            Version current_version{
                static_cast<int>(CL_VERSION_MAJOR(name_version.version)),
                static_cast<int>(CL_VERSION_MINOR(name_version.version))
            };
            max_supported_cl_c_version =
                (current_version > max_supported_cl_c_version)
                ? current_version
                : max_supported_cl_c_version;
        }
        return max_supported_cl_c_version;
    }

    return get_device_cl_c_version(device);
}

Version get_max_OpenCL_C_for_context(cl_context context)
{
    // Get all the devices in the context and find the maximum
    // universally supported OpenCL C version.
    size_t devices_size_in_bytes{};
    auto error = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, nullptr,
                                  &devices_size_in_bytes);
    test_error_ret(error, "clGetDeviceInfo failed for CL_CONTEXT_DEVICES",
                   (Version{ -1, 0 }));
    std::vector<cl_device_id> devices(devices_size_in_bytes
                                      / sizeof(cl_device_id));
    error = clGetContextInfo(context, CL_CONTEXT_DEVICES, devices_size_in_bytes,
                             devices.data(), nullptr);
    auto current_version = get_device_latest_cl_c_version(devices[0]);
    std::for_each(std::next(devices.begin()), devices.end(),
                  [&current_version](cl_device_id device) {
                      auto device_version =
                          get_device_latest_cl_c_version(device);
                      // OpenCL 3.0 is not backwards compatible with 2.0.
                      // If we have 3.0 and 2.0 in the same driver we
                      // use 1.2.
                      if (((device_version >= Version(2, 0)
                            && device_version < Version(3, 0))
                           && current_version >= Version(3, 0))
                          || (device_version >= Version(3, 0)
                              && (current_version >= Version(2, 0)
                                  && current_version < Version(3, 0))))
                      {
                          current_version = Version(1, 2);
                      }
                      else
                      {
                          current_version =
                              std::min(device_version, current_version);
                      }
                  });
    return current_version;
}

bool device_supports_cl_c_version(cl_device_id device, Version version)
{
    auto device_cl_version = get_device_cl_version(device);

    // In general, a device does not support an OpenCL C version if it is <=
    // CL_DEVICE_OPENCL_C_VERSION AND it does not appear in the
    // CL_DEVICE_OPENCL_C_ALL_VERSIONS query.

    // If the device version >= 3.0 it must support the
    // CL_DEVICE_OPENCL_C_ALL_VERSIONS query, and the version of OpenCL C being
    // used must appear in the query result if it's <=
    // CL_DEVICE_OPENCL_C_VERSION.
    if (device_cl_version >= Version{ 3, 0 })
    {
        size_t opencl_c_all_versions_size_in_bytes{};
        auto error =
            clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_ALL_VERSIONS, 0, nullptr,
                            &opencl_c_all_versions_size_in_bytes);
        test_error_ret(
            error, "clGetDeviceInfo failed for CL_DEVICE_OPENCL_C_ALL_VERSIONS",
            (false));
        std::vector<cl_name_version> name_versions(
            opencl_c_all_versions_size_in_bytes / sizeof(cl_name_version));
        error = clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_ALL_VERSIONS,
                                opencl_c_all_versions_size_in_bytes,
                                name_versions.data(), nullptr);
        test_error_ret(
            error, "clGetDeviceInfo failed for CL_DEVICE_OPENCL_C_ALL_VERSIONS",
            (false));

        for (const auto &name_version : name_versions)
        {
            Version current_version{
                static_cast<int>(CL_VERSION_MAJOR(name_version.version)),
                static_cast<int>(CL_VERSION_MINOR(name_version.version))
            };
            if (current_version == version)
            {
                return true;
            }
        }
    }

    return version <= get_device_cl_c_version(device);
}

bool poll_until(unsigned timeout_ms, unsigned interval_ms,
                std::function<bool()> fn)
{
    unsigned time_spent_ms = 0;
    bool ret = false;

    while (time_spent_ms < timeout_ms)
    {
        ret = fn();
        if (ret)
        {
            break;
        }
        usleep(interval_ms * 1000);
        time_spent_ms += interval_ms;
    }

    return ret;
}

bool device_supports_double(cl_device_id device)
{
    if (is_extension_available(device, "cl_khr_fp64"))
    {
        return true;
    }
    else
    {
        cl_device_fp_config double_fp_config;
        cl_int err = clGetDeviceInfo(device, CL_DEVICE_DOUBLE_FP_CONFIG,
                                     sizeof(double_fp_config),
                                     &double_fp_config, nullptr);
        test_error(err,
                   "clGetDeviceInfo for CL_DEVICE_DOUBLE_FP_CONFIG failed");
        return double_fp_config != 0;
    }
}

bool device_supports_half(cl_device_id device)
{
    return is_extension_available(device, "cl_khr_fp16");
}
