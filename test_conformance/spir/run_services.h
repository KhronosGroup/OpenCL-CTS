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
#ifndef __RUN_SERVICES_H
#define __RUN_SERVICES_H

#include <string>
#include "kernelargs.h"
#include "datagen.h"
#include <list>

void get_cl_file_path(const char *folder, const char *str, std::string &cl_file_path);
void get_bc_file_path(const char *folder, const char *str, std::string &bc_file_path, cl_uint size_t_width);
void get_h_file_path(const char *folder, const char *str, std::string &h_file_path);
void get_kernel_name(const char *test_name, std::string &kernel_name);

cl_device_id get_context_device(cl_context context);

void create_context_and_queue(cl_device_id device, cl_context *out_context, cl_command_queue *out_queue);
cl_program create_program_from_cl(cl_context context, const std::string& file_name);
cl_program create_program_from_bc(cl_context context, const std::string& file_name);
/**
 Retrieves the kernel with the given name from the program
 */
cl_kernel create_kernel_helper(cl_program program, const std::string& kernel_name);

cl_device_id get_program_device (cl_program program);

void generate_kernel_ws( cl_device_id device, cl_kernel kernel, WorkSizeInfo& ws);

/**
 Responsible for holding the result of a single test
 */
class TestResult
{
public:
    TestResult(){};

    KernelArgs& kernelArgs() { return m_kernelArgs; }

    const KernelArgs& kernelArgs() const { return m_kernelArgs; }

    void readToHost(cl_command_queue queue) { m_kernelArgs.readToHost(queue); }

    /*
     * Clones this object to a newly heap-allocated (deeply copied) object.
     */
    TestResult* clone(cl_context ctx, const WorkSizeInfo& ws, const cl_kernel kernel, const cl_device_id device) const;

private:
    KernelArgs m_kernelArgs;
};

template <int i>
struct KhrValue
{
  enum {Mask = (1 << i)};
};

template <>
struct KhrValue<0>
{
  enum {Mask = 1};
};

/*
 * Represents a set of OpenCL extension.
 */
class OclExtensions
{
public:
    static OclExtensions getDeviceCapabilities(cl_device_id);

    static OclExtensions empty();

    #define STRINIGFY(X) #X

    #define RETURN_IF_ENUM(S, E) if(S == STRINIGFY(E)) return E


    static OclExtensions fromString(const std::string&);

    std::string toString();

    // Operators

    // Merges the given extension and this one together, and returns the merged
    // value.
    OclExtensions operator|(const OclExtensions&) const;


    // Indicates whether each extension in this objects also resides in b.
    bool supports(const OclExtensions& b) const;

    // Return list of missing extensions
    OclExtensions get_missing(const OclExtensions& b) const;


    size_t get() const { return m_extVector; }
private:

    OclExtensions(size_t ext) : m_extVector(ext) {}

    enum ClKhrs
    {
        no_extensions = KhrValue<0>::Mask,
        has_cl_khr_int64_base_atomics = KhrValue<1>::Mask,
        has_cl_khr_int64_extended_atomics = KhrValue<2>::Mask,
        has_cl_khr_3d_image_writes = KhrValue<3>::Mask,
        has_cl_khr_fp16 = KhrValue<4>::Mask,
        has_cl_khr_gl_sharing = KhrValue<5>::Mask,
        has_cl_khr_gl_event = KhrValue<6>::Mask,
        has_cl_khr_d3d10_sharing = KhrValue<7>::Mask,
        has_cl_khr_dx9_media_sharing = KhrValue<8>::Mask,
        has_cl_khr_d3d11_sharing = KhrValue<9>::Mask,
        has_cl_khr_depth_images = KhrValue<10>::Mask,
        has_cl_khr_gl_depth_images = KhrValue<11>::Mask,
        has_cl_khr_gl_msaa_sharing = KhrValue<12>::Mask,
        has_cl_khr_image2d_from_buffer = KhrValue<13>::Mask,
        has_cl_khr_initialize_memory = KhrValue<14>::Mask,
        has_cl_khr_context_abort = KhrValue<15>::Mask,
        has_cl_khr_spir = KhrValue<16>::Mask,
        has_cl_khr_fp64 = KhrValue<17>::Mask,
        has_cl_khr_global_int32_base_atomics = KhrValue<18>::Mask,
        has_cl_khr_global_int32_extended_atomics = KhrValue<19>::Mask,
        has_cl_khr_local_int32_base_atomics = KhrValue<20>::Mask,
        has_cl_khr_local_int32_extended_atomics = KhrValue<21>::Mask,
        has_cl_khr_byte_addressable_store = KhrValue<22>::Mask,
        has_cles_khr_int64 = KhrValue<23>::Mask,
        has_cles_khr_2d_image_array_writes = KhrValue<24>::Mask,
    };

    size_t m_extVector;
};

std::ostream& operator<<(std::ostream& os, OclExtensions ext);

/*
 * Indicates whether a given test needs KHR extension.
 */

class DataRow;

class DataTable
{
  std::vector<DataRow*> m_rows;
public:
    size_t getNumRows() const;
    void addTableRow(DataRow*);
    const DataRow& operator[](int index)const;
    DataRow& operator[](int index);
};

class KhrSupport
{
public:
  static const KhrSupport* get(const std::string& csvFile);
  DataRow* parseLine(const std::string&);
  OclExtensions getRequiredExtensions(const char* suite, const char* test) const;
  cl_bool isImagesRequired(const char* suite, const char* test) const;
  cl_bool isImages3DRequired(const char* suite, const char* test) const;

private:
  static const int SUITE_INDEX     = 0;
  static const int TEST_INDEX      = 1;
  static const int EXT_INDEX       = 2;
  static const int IMAGES_INDEX    = 3;
  static const int IMAGES_3D_INDEX = 4;

  void parseCSV(std::fstream&);

  DataTable m_dt;
  static KhrSupport* m_instance;
};

class DataRow
{
    std::vector<std::string> m_row;
    DataRow() {}
public:
    const std::string& operator[](int)const;
    std::string&      operator[](int);

    friend DataRow* KhrSupport::parseLine(const std::string&);
};

/*
 * Generates data for the given kernel.
 * Parameters:
 *   context - The context of the kernel.
 *   kernel  - The kernel to which arguments will be generated
 *   ws(OUT) - generated work size info.
 *   res(OUT)- generated test results.
 */
void generate_kernel_data(cl_context context, cl_kernel kernel,
                          WorkSizeInfo &ws, TestResult& res);

void run_kernel(cl_kernel kernel, cl_command_queue queue, WorkSizeInfo &ws, TestResult& result);
bool compare_results(const TestResult& lhs, const TestResult& rhs, float ulps);

#endif
