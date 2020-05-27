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
#include "harness/compat.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <sstream>
#include <fstream>
#include <assert.h>
#include <functional>
#include <memory>

#include "harness/errorHelpers.h"
#include "harness/kernelHelpers.h"
#include "harness/typeWrappers.h"
#include "harness/clImageHelper.h"
#include "harness/os_helpers.h"

#include "exceptions.h"
#include "kernelargs.h"
#include "datagen.h"
#include "run_services.h"
#include "run_build_test.h"
#include "../math_brute_force/FunctionList.h"
#include <CL/cl.h>
//
// Task
//
Task::Task(cl_device_id device, const char* options):
m_devid(device) {
  if (options)
    m_options = options;
}

Task::~Task() {}

const char* Task::getErrorLog() const {
  return m_log.c_str();
}

void Task::setErrorLog(cl_program prog) {
    size_t len = 0;
    std::vector<char> log;

    cl_int err_code = clGetProgramBuildInfo(prog, m_devid, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
    if(err_code != CL_SUCCESS)
    {
        m_log = "Error: clGetProgramBuildInfo(CL_PROGRAM_BUILD_LOG, &len) failed.\n";
        return;
    }

    log.resize(len, 0);

    err_code = clGetProgramBuildInfo(prog, m_devid, CL_PROGRAM_BUILD_LOG, len, &log[0], NULL);
    if(err_code != CL_SUCCESS)
    {
        m_log = "Error: clGetProgramBuildInfo(CL_PROGRAM_BUILD_LOG, &log) failed.\n";
        return;
    }
    m_log.append(&log[0]);
}

//
// BuildTask
//
BuildTask::BuildTask(cl_program prog, cl_device_id dev, const char* options) :
    m_program(prog), Task(dev, options) {}

bool BuildTask::execute() {
    cl_int err_code = clBuildProgram(m_program, 0, NULL, m_options.c_str(), NULL, NULL);
    if(CL_SUCCESS == err_code)
        return true;

    setErrorLog(m_program);
    return false;
}

//
// SpirBuildTask
//
SpirBuildTask::SpirBuildTask(cl_program prog, cl_device_id dev, const char* options) :
    BuildTask(prog, dev, options) {}

//
// CompileTask
//

CompileTask::CompileTask(cl_program prog, cl_device_id dev, const char* options) :
    m_program(prog), Task(dev, options) {}

void CompileTask::addHeader(const char* hname, cl_program hprog) {
    m_headers.push_back(std::make_pair(hname, hprog));
}

const char* first(std::pair<const char*,cl_program>& p) {
    return p.first;
}

cl_program second(const std::pair<const char*, cl_program>& p) {
    return p.second;
}

bool CompileTask::execute() {
    // Generating the header names vector.
    std::vector<const char*> names;
    std::transform(m_headers.begin(), m_headers.end(), names.begin(), first);

    // Generating the header programs vector.
    std::vector<cl_program> programs;
    std::transform(m_headers.begin(), m_headers.end(), programs.begin(), second);

    const char** h_names = NULL;
    const cl_program* h_programs = NULL;
    if (!m_headers.empty())
    {
        h_programs = &programs[0];
        h_names    = &names[0];
    }

    // Compiling with the headers.
    cl_int err_code = clCompileProgram(
        m_program,
        1U,
        &m_devid,
        m_options.c_str(),
        m_headers.size(), // # of headers
        h_programs,
        h_names,
        NULL, NULL);
    if (CL_SUCCESS == err_code)
        return true;

    setErrorLog(m_program);
    return false;
}

//
// SpirCompileTask
//
SpirCompileTask::SpirCompileTask(cl_program prog, cl_device_id dev, const char* options) :
    CompileTask(prog, dev, options) {}


//
// LinkTask
//
LinkTask::LinkTask(cl_program* programs, int num_programs, cl_context ctxt,
                   cl_device_id dev, const char* options) :
    m_programs(programs), m_numPrograms(num_programs), m_context(ctxt), m_executable(NULL),
    Task(dev, options) {}

bool LinkTask::execute() {
    cl_int err_code;
    int i;

    for(i = 0; i < m_numPrograms; ++i)
    {
        err_code = clCompileProgram(m_programs[i], 1, &m_devid, "-x spir -spir-std=1.2 -cl-kernel-arg-info", 0, NULL, NULL, NULL, NULL);
        if (CL_SUCCESS != err_code)
        {
            setErrorLog(m_programs[i]);
            return false;
        }
    }

    m_executable = clLinkProgram(m_context, 1, &m_devid, m_options.c_str(), m_numPrograms, m_programs, NULL, NULL, &err_code);
    if (CL_SUCCESS == err_code)
      return true;

    if(m_executable) setErrorLog(m_executable);
    return false;
}

cl_program LinkTask::getExecutable() const {
    return m_executable;
}

LinkTask::~LinkTask() {
    if(m_executable) clReleaseProgram(m_executable);
}

//
// KernelEnumerator
//
void KernelEnumerator::process(cl_program prog) {
    const size_t MAX_KERNEL_NAME = 64;
    size_t num_kernels;

    cl_int err_code = clGetProgramInfo(
        prog,
        CL_PROGRAM_NUM_KERNELS,
        sizeof(size_t),
        &num_kernels,
        NULL
    );
    if (CL_SUCCESS != err_code)
        return;

    // Querying for the number of kernels.
    size_t buffer_len = sizeof(char)*num_kernels*MAX_KERNEL_NAME;
    char* kernel_names = new char[buffer_len];
    memset(kernel_names, '\0', buffer_len);
    size_t str_len = 0;
    err_code = clGetProgramInfo(
        prog,
        CL_PROGRAM_KERNEL_NAMES,
        buffer_len,
        (void *)kernel_names,
        &str_len
    );
    if (CL_SUCCESS != err_code)
        return;

    //parsing the names and inserting them to the list
    std::string names(kernel_names);
    assert (str_len == 1+names.size() && "incompatible string lengths");
    size_t offset = 0;
    for(size_t i=0 ; i<names.size() ; ++i){
        //kernel names are separated by semi colons
        if (names[i] == ';'){
            m_kernels.push_back(names.substr(offset, i-offset));
            offset = i+1;
        }
    }
    m_kernels.push_back(names.substr(offset, names.size()-offset));
    delete[] kernel_names;
}

KernelEnumerator::KernelEnumerator(cl_program prog) {
    process(prog);
}

KernelEnumerator::iterator KernelEnumerator::begin(){
    return m_kernels.begin();
}

KernelEnumerator::iterator KernelEnumerator::end(){
    return m_kernels.end();
}

size_t KernelEnumerator::size() const {
    return m_kernels.size();
}

/**
 Run the single test - run the test for both CL and SPIR versions of the kernel
 */
static bool run_test(cl_context context, cl_command_queue queue, cl_program clprog,
    cl_program bcprog, const std::string& kernel_name, std::string& err, const cl_device_id device,
    float ulps)
{
    WorkSizeInfo ws;
    TestResult cl_result;
    std::unique_ptr<TestResult> bc_result;
    // first, run the single CL test
    {
        // make sure that the kernel will be released before the program
        clKernelWrapper kernel = create_kernel_helper(clprog, kernel_name);
        // based on the kernel characteristics, we are generating and initializing the arguments for both phases (cl and bc executions)
        generate_kernel_data(context, kernel, ws, cl_result);
        bc_result.reset(cl_result.clone(context, ws, kernel, device));
        assert (compare_results(cl_result, *bc_result, ulps) && "not equal?");
        run_kernel( kernel, queue, ws, cl_result );
    }
    // now, run the single BC test
    {
        // make sure that the kernel will be released before the program
        clKernelWrapper kernel = create_kernel_helper(bcprog, kernel_name);
        run_kernel( kernel, queue, ws, *bc_result );
    }

    int error = clFinish(queue);
    if( CL_SUCCESS != error)
    {
        err = "clFinish failed\n";
        return false;
    }

    // compare the results
    if( !compare_results(cl_result, *bc_result, ulps) )
    {
        err = " (result diff in kernel '" + kernel_name + "').";
        return false;
    }
    return true;
}

/**
 Get the maximum relative error defined as ULP of floating-point math functions
 */
static float get_max_ulps(const char *test_name)
{
    float ulps = 0.f;
    // Get ULP values from math_brute_force functionList
    if (strstr(test_name, "math_kernel"))
    {
        for( size_t i = 0; i < functionListCount; i++ )
        {
            char name[64];
            const Func *func = &functionList[ i ];
            sprintf(name, ".%s_float", func->name);
            if (strstr(test_name, name))
            {
                ulps = func->float_ulps;
            }
            else
            {
                sprintf(name, ".%s_double", func->name);
                if (strstr(test_name, name))
                {
                    ulps = func->double_ulps;
                }
            }
        }
    }
    return ulps;
}

TestRunner::TestRunner(EventHandler *success, EventHandler *failure,
                       const OclExtensions& devExt):
    m_successHandler(success), m_failureHandler(failure), m_devExt(&devExt) {}

/**
 Based on the test name build the cl file name, the bc file name and execute
 the kernel for both modes (cl and bc).
 */
bool TestRunner::runBuildTest(cl_device_id device, const char *folder,
                              const char *test_name, cl_uint size_t_width)
{
    int failures = 0;
    // Composing the name of the CSV file.
    char* dir = get_exe_dir();
    std::string csvName(dir);
    csvName.append(dir_sep());
    csvName.append("khr.csv");
    free(dir);

    log_info("%s...\n", test_name);

    float ulps = get_max_ulps(test_name);

    // Figure out whether the test can run on the device. If not, we skip it.
    const KhrSupport& khrDb = *KhrSupport::get(csvName);
    cl_bool images = khrDb.isImagesRequired(folder, test_name);
    cl_bool images3D = khrDb.isImages3DRequired(folder, test_name);

    char deviceProfile[64];
    clGetDeviceInfo(device, CL_DEVICE_PROFILE, sizeof(deviceProfile), &deviceProfile, NULL);
    std::string device_profile(deviceProfile, 64);

    if(images == CL_TRUE && checkForImageSupport(device) != 0)
    {
        (*m_successHandler)(test_name, "");
        std::cout << "Skipped. (Cannot run on device due to Images is not supported)." << std::endl;
        return true;
    }

    if(images3D == CL_TRUE && checkFor3DImageSupport(device) != 0)
    {
        (*m_successHandler)(test_name, "");
        std::cout << "Skipped. (Cannot run on device as 3D images are not supported)." << std::endl;
        return true;
    }

    OclExtensions requiredExt = khrDb.getRequiredExtensions(folder, test_name);
    if(!m_devExt->supports(requiredExt))
    {
        (*m_successHandler)(test_name, "");
        std::cout << "Skipped. (Cannot run on device due to missing extensions: " << m_devExt->get_missing(requiredExt) << " )." << std::endl;
        return true;
    }

    std::string cl_file_path, bc_file;
    // Build cl file name based on the test name
    get_cl_file_path(folder, test_name, cl_file_path);
    // Build bc file name based on the test name
    get_bc_file_path(folder, test_name, bc_file, size_t_width);
    gRG.init(1);
    //
    // Processing each kernel in the program separately
    //
    clContextWrapper context;
    clCommandQueueWrapper queue;
    create_context_and_queue(device, &context, &queue);
    clProgramWrapper clprog = create_program_from_cl(context, cl_file_path);
    clProgramWrapper bcprog = create_program_from_bc(context, bc_file);
    std::string bcoptions = "-x spir -spir-std=1.2 -cl-kernel-arg-info";
    std::string cloptions = "-cl-kernel-arg-info";

    cl_device_fp_config gFloatCapabilities = 0;
    cl_int err;
    if ((err = clGetDeviceInfo(device, CL_DEVICE_SINGLE_FP_CONFIG, sizeof(gFloatCapabilities), &gFloatCapabilities, NULL)))
    {
        log_info("Unable to get device CL_DEVICE_SINGLE_FP_CONFIG. (%d)\n", err);
    }

    if (strstr(test_name, "div_cr") || strstr(test_name, "sqrt_cr")) {
        if ((gFloatCapabilities & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT) == 0) {
            (*m_successHandler)(test_name, "");
            std::cout << "Skipped. (Cannot run on device due to missing CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT property.)" << std::endl;
            return true;
        } else {
            bcoptions += " -cl-fp32-correctly-rounded-divide-sqrt";
            cloptions += " -cl-fp32-correctly-rounded-divide-sqrt";
        }
    }

    // Building the programs.
    BuildTask clBuild(clprog, device, cloptions.c_str());
    if (!clBuild.execute()) {
        std::cerr << clBuild.getErrorLog() << std::endl;
        return false;
    }

    SpirBuildTask bcBuild(bcprog, device, bcoptions.c_str());
    if (!bcBuild.execute()) {
        std::cerr << bcBuild.getErrorLog() << std::endl;
        return false;
    }

    KernelEnumerator clkernel_enumerator(clprog),
                     bckernel_enumerator(bcprog);
    if (clkernel_enumerator.size() != bckernel_enumerator.size()) {
        std::cerr << "number of kernels in test" << test_name
                  << " doesn't match in bc and cl files" << std::endl;
        return false;
    }
    KernelEnumerator::iterator it = clkernel_enumerator.begin(),
        e = clkernel_enumerator.end();
    while (it != e)
    {
        std::string kernel_name = *it++;
        std::string err;
        try
        {
            bool success = run_test(context, queue, clprog, bcprog, kernel_name, err, device, ulps);
            if (success)
            {
                log_info("kernel '%s' passed.\n", kernel_name.c_str());
                (*m_successHandler)(test_name, kernel_name);
            }
            else
            {
                ++failures;
                log_info("kernel '%s' failed.\n", kernel_name.c_str());
                (*m_failureHandler)(test_name, kernel_name);
            }
        }
        catch (std::runtime_error err)
        {
            ++failures;
            log_info("kernel '%s' failed: %s\n", kernel_name.c_str(), err.what());
            (*m_failureHandler)(test_name, kernel_name);
        }
    }

    log_info("%s %s\n", test_name, failures ? "FAILED" : "passed.");
    return failures == 0;
}

