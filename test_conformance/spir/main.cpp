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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <memory>
#include <sstream>
#include <iterator>

#include "harness/errorHelpers.h"
#include "harness/kernelHelpers.h"
#include "harness/typeWrappers.h"
#include "harness/os_helpers.h"

#include "exceptions.h"
#include "run_build_test.h"
#include "run_services.h"

#include <list>
#include <algorithm>
#include "miniz/miniz.h"

#if defined(_WIN32)
#include <windows.h>
#include <direct.h>
#else // !_WIN32
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#endif

static int no_unzip = 0;

class custom_cout : public std::streambuf
{
private:
    std::stringstream ss;

    std::streamsize xsputn (const char* s, std::streamsize n)
    {
        ss.write(s, n);
        return n;
    }

    int overflow(int c)
    {
        if(c > 0 && c < 256) ss.put(c);
        return c;
    }

    int sync()
    {
        log_info("%s", ss.str().c_str());
        ss.str("");
        return 0;
    }
};

class custom_cerr : public std::streambuf
{
private:
    std::stringstream ss;

    std::streamsize xsputn (const char* s, std::streamsize n)
    {
        ss.write(s, n);
        return n;
    }

    int overflow(int c)
    {
        if(c > 0 && c < 256) ss.put(c);
        return c;
    }

    int sync()
    {
        log_error("%s", ss.str().c_str());
        ss.str("");
        return 0;
    }
};

class override_buff
{
    std::ostream* stream;
    std::streambuf* buff;

public:
    override_buff(std::ostream& s, std::streambuf& b)
    {
        stream = &s;
        buff = stream->rdbuf();
        stream->rdbuf(&b);
    }

    ~override_buff()
    {
        stream->rdbuf(buff);
    }
};

typedef bool (*testfn)(cl_device_id device, cl_uint size_t_width, const char *folder);

template <typename T>
void dealloc(T *p)
{
    if (p) delete p;
}

static bool is_dir_exits(const char* path)
{
    assert(path && "NULL directory");
#if defined(_WIN32)
    DWORD ftyp = GetFileAttributesA(path);
    if (ftyp != INVALID_FILE_ATTRIBUTES && (ftyp & FILE_ATTRIBUTE_DIRECTORY))
    return true;
#else // Linux assumed here.
    if (DIR *pDir = opendir(path))
    {
        closedir(pDir);
        return true;
    }
#endif
    return false;
}

static void get_spir_version(cl_device_id device,
                             std::vector<Version> &versions)
{
    char version[64] = {0};
    cl_int err;
    size_t size = 0;

    if ((err = clGetDeviceInfo(device, CL_DEVICE_SPIR_VERSIONS, sizeof(version),
                               (void *)version, &size)))
    {
        log_error( "Error: failed to obtain SPIR version at %s:%d (err = %d)\n",
                  __FILE__, __LINE__, err );
        return;
    }

    assert(size && "Empty version string");

    std::list<std::string> versionVector;
    std::stringstream versionStream(version);
    std::copy(std::istream_iterator<std::string>(versionStream),
              std::istream_iterator<std::string>(),
              std::back_inserter(versionVector));
    for (auto &v : versionVector)
    {
        auto major = v[v.find('.') - 1];
        auto minor = v[v.find('.') + 1];
        versions.push_back(Version{ major - '0', minor - '0' });
    }
}

struct CounterEventHandler: EventHandler{
    unsigned int& Counter;
    unsigned int TN;
    std::string LastTest;

    //N - counter of successful tests.
    //T - total number of tests in the suite
    CounterEventHandler(unsigned int& N, unsigned int T): Counter(N), TN(T){}

    void operator ()(const std::string& testName, const std::string& kernelName) {
        if (testName != LastTest){
            ++Counter;
            LastTest = testName;
        }
    }
};

class AccumulatorEventHandler: public EventHandler{
  std::list<std::string>& m_list;
  const std::string m_name;
public:
  AccumulatorEventHandler(std::list<std::string>& L, const std::string N):
  m_list(L), m_name(N){}

  void operator ()(const std::string& T, const std::string& K){
    std::cerr << "\nTest " << T << "\t Kernel " << K << " failed" << std::endl;
    m_list.push_back(m_name + "\t" + T + "\t" + K);
  }
};

static void printError(const std::string& S){
  std::cerr << S << std::endl;
}

static bool extractKernelAttribute(std::string& kernel_attributes,
    const std::string& attribute, std::vector<std::string>& attribute_vector) {
  size_t start = kernel_attributes.find(attribute + "(");
  if (start == 0) {
    size_t end = kernel_attributes.find(")", start);
    if (end != std::string::npos) {
      size_t length = end-start+1;
      attribute_vector.push_back(kernel_attributes.substr(start, length));
      kernel_attributes.erase(start, length);
      return true;
    }
  }
  return false;
}

// Extracts suite with the given name, and saves it to disk.
static void extract_suite(const char *suiteName)
{
  mz_zip_archive zip_archive;

  // Composing the name of the archive.
  char* dir = get_exe_dir();
  std::string archiveName(dir);
  archiveName.append(dir_sep());
  archiveName.append(suiteName);
  archiveName.append(".zip");
  free(dir);

#if defined(_WIN32)
      _mkdir(suiteName);
#else
      mkdir(suiteName, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
#endif

  memset(&zip_archive, 0, sizeof(zip_archive));
  if (!mz_zip_reader_init_file(&zip_archive, archiveName.c_str(), 0))
      throw Exceptions::ArchiveError(MZ_DATA_ERROR);

  // Get and print information about each file in the archive.
  for (size_t i = 0; i < mz_zip_reader_get_num_files(&zip_archive); i++)
  {
      mz_zip_archive_file_stat fileStat;
      size_t fileSize = 0;

      if (!mz_zip_reader_file_stat(&zip_archive, i, &fileStat))
      {
          mz_zip_reader_end(&zip_archive);
          throw Exceptions::ArchiveError(MZ_DATA_ERROR);
      }
      const std::string fileName = fileStat.m_filename;

      // If the file is a directory, skip it. We create suite folder at the beggining.
      if (mz_zip_reader_is_file_a_directory(&zip_archive, fileStat.m_file_index))
      {
          continue;
      }

      // Extracting the file.
      void *p = mz_zip_reader_extract_file_to_heap(&zip_archive,
                                                   fileName.c_str(),
                                                   &fileSize, 0);
      if (!p)
      {
          mz_zip_reader_end(&zip_archive);
          throw std::runtime_error("mz_zip_reader_extract_file_to_heap() failed!\n");
      }

      // Writing the file back to the disk
      std::fstream file(fileName.c_str(),
                        std::ios_base::trunc | std::ios_base::in |
                        std::ios_base::out | std::ios_base::binary);
      if (!file.is_open())
      {
          std::string msg = "Failed to open ";
          msg.append(fileName);
          throw Exceptions::TestError(msg);
      }

      file.write((const char*)p, fileSize);
      if (file.bad())
      {
          std::string msg("Failed to write into ");
          msg.append(fileName);
          throw Exceptions::TestError(msg);
      }

      // Cleanup.
      file.flush();
      file.close();
      free(p);
  }
  mz_zip_reader_end(&zip_archive);
}

//
// Extracts the given suite package if needed.
// return true if the suite was extracted, false otherwise.
//
static bool try_extract(const char* suite)
{
    if(no_unzip == 0)
    {
        std::cout << "Extracting test suite " << suite << std::endl;
        extract_suite(suite);
        std::cout << "Done." << std::endl;
    }
    return true;
}

bool test_suite(cl_device_id device, cl_uint size_t_width, const char *folder,
                const char *test_name[], unsigned int number_of_tests,
                const char *extension)
{
    // If the folder doesn't exist, we extract in from the archive.
    try_extract(folder);

    std::cout << "Running tests:" << std::endl;

    OclExtensions deviceCapabilities = OclExtensions::getDeviceCapabilities(device);
    unsigned int tests_passed = 0;
    CounterEventHandler SuccE(tests_passed, number_of_tests);
    std::list<std::string> ErrList;
    for (unsigned int i = 0; i < number_of_tests; ++i)
    {
        AccumulatorEventHandler FailE(ErrList, test_name[i]);
        if((strlen(extension) != 0) && (!is_extension_available(device, extension)))
        {
            (SuccE)(test_name[i], "");
            std::cout << test_name[i] << "... Skipped. (Cannot run on device due to missing extension: " << extension << " )." << std::endl;
            continue;
        }
        TestRunner testRunner(&SuccE, &FailE, deviceCapabilities);
        testRunner.runBuildTest(device, folder, test_name[i], size_t_width);
    }

    std::cout << std::endl;
    std::cout << "PASSED " << tests_passed << " of " << number_of_tests << " tests.\n" << std::endl;

    if (!ErrList.empty())
    {
        std::cout << "Failed tests:" << std::endl;
        std::for_each(ErrList.begin(), ErrList.end(), printError);
        std::cout << std::endl;
        return false;
    }
    std::cout << std::endl;
    return true;
}

static std::string getTestFolder(const std::string& TS)
{
  const std::string DOUBLE("_double");
  if (TS.size() < DOUBLE.size())
    return TS;

  const size_t prefixLen = TS.size() - DOUBLE.size();
  const std::string postfix = TS.substr(prefixLen, DOUBLE.size());
  if (DOUBLE == postfix)
      return TS.substr(0, prefixLen);

  return TS;
}

bool test_api (cl_device_id device, cl_uint size_t_width, const char *folder)
{
    static const char* test_name[] = {
        "const_derived_d",
        "const_scalar_d",
        "const_vector16_d",
        "const_vector2_d",
        "const_vector3_d",
        "const_vector4_d",
        "const_vector8_d",
        "constant_derived_p0",
        "constant_derived_p1",
        "constant_derived_restrict_p0",
        "constant_derived_restrict_p1",
        "constant_scalar_p0",
        "constant_scalar_p1",
        "constant_scalar_p2",
        "constant_scalar_p3",
        "constant_scalar_restrict_p0",
        "constant_scalar_restrict_p1",
        "constant_scalar_restrict_p2",
        "constant_scalar_restrict_p3",
        "constant_vector16_p0",
        "constant_vector16_p1",
        "constant_vector16_p2",
        "constant_vector16_restrict_p0",
        "constant_vector16_restrict_p2",
        "constant_vector2_p0",
        "constant_vector2_p1",
        "constant_vector2_restrict_p0",
        "constant_vector2_restrict_p1",
        "constant_vector2_restrict_p2",
        "constant_vector3_p0",
        "constant_vector3_p1",
        "constant_vector3_p2",
        "constant_vector3_restrict_p0",
        "constant_vector3_restrict_p1",
        "constant_vector3_restrict_p2",
        "constant_vector4_p0",
        "constant_vector4_p1",
        "constant_vector4_p2",
        "constant_vector4_restrict_p0",
        "constant_vector4_restrict_p1",
        "constant_vector4_restrict_p2",
        "constant_vector8_p0",
        "constant_vector8_p1",
        "constant_vector8_p2",
        "constant_vector8_restrict_p0",
        "constant_vector8_restrict_p1",
        "constant_vector8_restrict_p2",
        "derived_d",
        "global_const_derived_p",
        "global_const_derived_restrict_p",
        "global_const_scalar_p",
        "global_const_scalar_restrict_p",
        "global_const_vector16_p",
        "global_const_vector16_restrict_p",
        "global_const_vector2_p",
        "global_const_vector2_restrict_p",
        "global_const_vector3_p",
        "global_const_vector3_restrict_p",
        "global_const_vector4_p",
        "global_const_vector4_restrict_p",
        "global_const_vector8_p",
        "global_const_vector8_restrict_p",
        "global_const_volatile_derived_p",
        "global_const_volatile_derived_restrict_p",
        "global_const_volatile_scalar_p",
        "global_const_volatile_scalar_restrict_p",
        "global_const_volatile_vector16_p",
        "global_const_volatile_vector16_restrict_p",
        "global_const_volatile_vector2_p",
        "global_const_volatile_vector2_restrict_p",
        "global_const_volatile_vector3_p",
        "global_const_volatile_vector3_restrict_p",
        "global_const_volatile_vector4_p",
        "global_const_volatile_vector4_restrict_p",
        "global_const_volatile_vector8_p",
        "global_const_volatile_vector8_restrict_p",
        "global_derived_p",
        "global_derived_restrict_p",
        "global_scalar_p",
        "global_scalar_restrict_p",
        "global_vector16_p",
        "global_vector16_restrict_p",
        "global_vector2_p",
        "global_vector2_restrict_p",
        "global_vector3_p",
        "global_vector3_restrict_p",
        "global_vector4_p",
        "global_vector4_restrict_p",
        "global_vector8_p",
        "global_vector8_restrict_p",
        "global_volatile_derived_p",
        "global_volatile_derived_restrict_p",
        "global_volatile_scalar_p",
        "global_volatile_scalar_restrict_p",
        "global_volatile_vector16_p",
        "global_volatile_vector16_restrict_p",
        "global_volatile_vector2_p",
        "global_volatile_vector2_restrict_p",
        "global_volatile_vector3_p",
        "global_volatile_vector3_restrict_p",
        "global_volatile_vector4_p",
        "global_volatile_vector4_restrict_p",
        "global_volatile_vector8_p",
        "global_volatile_vector8_restrict_p",
        "local_const_derived_p",
        "local_const_derived_restrict_p",
        "local_const_scalar_p",
        "local_const_scalar_restrict_p",
        "local_const_vector16_p",
        "local_const_vector16_restrict_p",
        "local_const_vector2_p",
        "local_const_vector2_restrict_p",
        "local_const_vector3_p",
        "local_const_vector3_restrict_p",
        "local_const_vector4_p",
        "local_const_vector4_restrict_p",
        "local_const_vector8_p",
        "local_const_vector8_restrict_p",
        "local_const_volatile_derived_p",
        "local_const_volatile_derived_restrict_p",
        "local_const_volatile_scalar_p",
        "local_const_volatile_scalar_restrict_p",
        "local_const_volatile_vector16_p",
        "local_const_volatile_vector16_restrict_p",
        "local_const_volatile_vector2_p",
        "local_const_volatile_vector2_restrict_p",
        "local_const_volatile_vector3_p",
        "local_const_volatile_vector3_restrict_p",
        "local_const_volatile_vector4_p",
        "local_const_volatile_vector4_restrict_p",
        "local_const_volatile_vector8_p",
        "local_const_volatile_vector8_restrict_p",
        "local_derived_p",
        "local_derived_restrict_p",
        "local_scalar_p",
        "local_scalar_restrict_p",
        "local_vector16_p",
        "local_vector16_restrict_p",
        "local_vector2_p",
        "local_vector2_restrict_p",
        "local_vector3_p",
        "local_vector3_restrict_p",
        "local_vector4_p",
        "local_vector4_restrict_p",
        "local_vector8_p",
        "local_vector8_restrict_p",
        "local_volatile_derived_p",
        "local_volatile_derived_restrict_p",
        "local_volatile_scalar_p",
        "local_volatile_scalar_restrict_p",
        "local_volatile_vector16_p",
        "local_volatile_vector16_restrict_p",
        "local_volatile_vector2_p",
        "local_volatile_vector2_restrict_p",
        "local_volatile_vector3_p",
        "local_volatile_vector3_restrict_p",
        "local_volatile_vector4_p",
        "local_volatile_vector4_restrict_p",
        "local_volatile_vector8_p",
        "local_volatile_vector8_restrict_p",
        "private_const_derived_d",
        "private_const_scalar_d",
        "private_const_vector16_d",
        "private_const_vector2_d",
        "private_const_vector3_d",
        "private_const_vector4_d",
        "private_const_vector8_d",
        "private_derived_d",
        "private_scalar_d",
        "private_vector16_d",
        "private_vector2_d",
        "private_vector3_d",
        "private_vector4_d",
        "private_vector8_d",
        "scalar_d",
        "vector16_d",
        "vector2_d",
        "vector3_d",
        "vector4_d",
        "vector8_d",
        "image_d",
        "image_d_write_array",
        "image_d_3d",
        "sample_test.min_max_read_image_args",
        "kernel_with_bool",
        "bool_scalar_d",
        "long_constant_scalar_p2",
        "long_const_scalar_d",
        "long_const_vector16_d",
        "long_const_vector2_d",
        "long_const_vector3_d",
        "long_const_vector4_d",
        "long_const_vector8_d",
        "long_constant_scalar_p3",
        "long_constant_scalar_restrict_p2",
        "long_constant_scalar_restrict_p3",
        "long_constant_vector16_p1",
        "long_constant_vector16_restrict_p1",
        "long_constant_vector2_p1",
        "long_constant_vector2_restrict_p1",
        "long_constant_vector3_p1",
        "long_constant_vector3_restrict_p1",
        "long_constant_vector4_p1",
        "long_constant_vector4_restrict_p1",
        "long_constant_vector8_p1",
        "long_constant_vector8_restrict_p1",
        "long_global_const_scalar_p",
        "long_global_const_scalar_restrict_p",
        "long_global_const_vector16_p",
        "long_global_const_vector16_restrict_p",
        "long_global_const_vector2_p",
        "long_global_const_vector2_restrict_p",
        "long_global_const_vector3_p",
        "long_global_const_vector3_restrict_p",
        "long_global_const_vector4_p",
        "long_global_const_vector4_restrict_p",
        "long_global_const_vector8_p",
        "long_global_const_vector8_restrict_p",
        "long_global_const_volatile_scalar_p",
        "long_global_const_volatile_scalar_restrict_p",
        "long_global_const_volatile_vector16_p",
        "long_global_const_volatile_vector16_restrict_p",
        "long_global_const_volatile_vector2_p",
        "long_global_const_volatile_vector2_restrict_p",
        "long_global_const_volatile_vector3_p",
        "long_global_const_volatile_vector3_restrict_p",
        "long_global_const_volatile_vector4_p",
        "long_global_const_volatile_vector4_restrict_p",
        "long_global_const_volatile_vector8_p",
        "long_global_const_volatile_vector8_restrict_p",
        "long_global_scalar_p",
        "long_global_scalar_restrict_p",
        "long_global_vector16_p",
        "long_global_vector16_restrict_p",
        "long_global_vector2_p",
        "long_global_vector2_restrict_p",
        "long_global_vector3_p",
        "long_global_vector3_restrict_p",
        "long_global_vector4_p",
        "long_global_vector4_restrict_p",
        "long_global_vector8_p",
        "long_global_vector8_restrict_p",
        "long_global_volatile_scalar_p",
        "long_global_volatile_scalar_restrict_p",
        "long_global_volatile_vector16_p",
        "long_global_volatile_vector16_restrict_p",
        "long_global_volatile_vector2_p",
        "long_global_volatile_vector2_restrict_p",
        "long_global_volatile_vector3_p",
        "long_global_volatile_vector3_restrict_p",
        "long_global_volatile_vector4_p",
        "long_global_volatile_vector4_restrict_p",
        "long_global_volatile_vector8_p",
        "long_global_volatile_vector8_restrict_p",
        "long_local_const_scalar_p",
        "long_local_const_scalar_restrict_p",
        "long_local_const_vector16_p",
        "long_local_const_vector16_restrict_p",
        "long_local_const_vector2_p",
        "long_local_const_vector2_restrict_p",
        "long_local_const_vector3_p",
        "long_local_const_vector3_restrict_p",
        "long_local_const_vector4_p",
        "long_local_const_vector4_restrict_p",
        "long_local_const_vector8_p",
        "long_local_const_vector8_restrict_p",
        "long_local_const_volatile_scalar_p",
        "long_local_const_volatile_scalar_restrict_p",
        "long_local_const_volatile_vector16_p",
        "long_local_const_volatile_vector16_restrict_p",
        "long_local_const_volatile_vector2_p",
        "long_local_const_volatile_vector2_restrict_p",
        "long_local_const_volatile_vector3_p",
        "long_local_const_volatile_vector3_restrict_p",
        "long_local_const_volatile_vector4_p",
        "long_local_const_volatile_vector4_restrict_p",
        "long_local_const_volatile_vector8_p",
        "long_local_const_volatile_vector8_restrict_p",
        "long_local_scalar_p",
        "long_local_scalar_restrict_p",
        "long_local_vector16_p",
        "long_local_vector16_restrict_p",
        "long_local_vector2_p",
        "long_local_vector2_restrict_p",
        "long_local_vector3_p",
        "long_local_vector3_restrict_p",
        "long_local_vector4_p",
        "long_local_vector4_restrict_p",
        "long_local_vector8_p",
        "long_local_vector8_restrict_p",
        "long_local_volatile_scalar_p",
        "long_local_volatile_scalar_restrict_p",
        "long_local_volatile_vector16_p",
        "long_local_volatile_vector16_restrict_p",
        "long_local_volatile_vector2_p",
        "long_local_volatile_vector2_restrict_p",
        "long_local_volatile_vector3_p",
        "long_local_volatile_vector3_restrict_p",
        "long_local_volatile_vector4_p",
        "long_local_volatile_vector4_restrict_p",
        "long_local_volatile_vector8_p",
        "long_local_volatile_vector8_restrict_p",
        "long_private_const_scalar_d",
        "long_private_const_vector16_d",
        "long_private_const_vector2_d",
        "long_private_const_vector3_d",
        "long_private_const_vector4_d",
        "long_private_const_vector8_d",
        "long_private_scalar_d",
        "long_private_vector16_d",
        "long_private_vector2_d",
        "long_private_vector3_d",
        "long_private_vector4_d",
        "long_private_vector8_d",
        "long_scalar_d",
        "long_vector16_d",
        "long_vector2_d",
        "long_vector3_d",
        "long_vector4_d",
        "long_vector8_d",
    };

    log_info("test_api\n");
    return test_suite(device, size_t_width, folder, test_name, sizeof(test_name) / sizeof(const char *), "");
}


bool test_api_double (cl_device_id device, cl_uint size_t_width, const char *folder)
{
    static const char* test_name[] = {
        "double_scalar_p",
        "double_scalar_p2",
        "double_scalar_d",
        "double_vector2_p",
        "double_vector2_p2",
        "double_vector2_d",
        "double_vector3_p",
        "double_vector3_p2",
        "double_vector3_d",
        "double_vector4_p",
        "double_vector4_p2",
        "double_vector4_d",
        "double_vector8_p",
        "double_vector8_p2",
        "double_vector8_d",
        "double_vector16_p",
        "double_vector16_p2",
        "double_vector16_d",
    };

    log_info("test_api_double\n");
    return test_suite(device, size_t_width, folder, test_name, sizeof(test_name) / sizeof(const char *), "cl_khr_fp64");
}


bool test_atomics (cl_device_id device, cl_uint size_t_width, const char *folder)
{
    static const char* test_name[] = {
        "test_atomic_fn.atomic_add_global_int",
        "test_atomic_fn.atomic_add_global_uint",
        "test_atomic_fn.atomic_sub_global_int",
        "test_atomic_fn.atomic_sub_global_uint",
        "test_atomic_fn.atomic_xchg_global_int",
        "test_atomic_fn.atomic_xchg_global_uint",
        "test_atomic_fn.atomic_xchg_global_float",
        "test_atomic_fn.atomic_min_global_int",
        "test_atomic_fn.atomic_min_global_uint",
        "test_atomic_fn.atomic_max_global_int",
        "test_atomic_fn.atomic_max_global_uint",
        "test_atomic_fn.atomic_inc_global_int",
        "test_atomic_fn.atomic_inc_global_uint",
        "test_atomic_fn.atomic_dec_global_int",
        "test_atomic_fn.atomic_dec_global_uint",
        "test_atomic_fn.atomic_cmpxchg_global_int",
        "test_atomic_fn.atomic_cmpxchg_global_uint",
        "test_atomic_fn.atomic_and_global_int",
        "test_atomic_fn.atomic_and_global_uint",
        "test_atomic_fn.atomic_or_global_int",
        "test_atomic_fn.atomic_or_global_uint",
        "test_atomic_fn.atomic_xor_global_int",
        "test_atomic_fn.atomic_xor_global_uint",
        "test_atomic_fn.atomic_add_local_int",
        "test_atomic_fn.atomic_add_local_uint",
        "test_atomic_fn.atomic_sub_local_int",
        "test_atomic_fn.atomic_sub_local_uint",
        "test_atomic_fn.atomic_xchg_local_int",
        "test_atomic_fn.atomic_xchg_local_uint",
        "test_atomic_fn.atomic_xchg_local_float",
        "test_atomic_fn.atomic_min_local_int",
        "test_atomic_fn.atomic_min_local_uint",
        "test_atomic_fn.atomic_max_local_int",
        "test_atomic_fn.atomic_max_local_uint",
        "test_atomic_fn.atomic_inc_local_int",
        "test_atomic_fn.atomic_inc_local_uint",
        "test_atomic_fn.atomic_dec_local_int",
        "test_atomic_fn.atomic_dec_local_uint",
        "test_atomic_fn.atomic_cmpxchg_local_int",
        "test_atomic_fn.atomic_cmpxchg_local_uint",
        "test_atomic_fn.atomic_and_local_int",
        "test_atomic_fn.atomic_and_local_uint",
        "test_atomic_fn.atomic_or_local_int",
        "test_atomic_fn.atomic_or_local_uint",
        "test_atomic_fn.atomic_xor_local_int",
        "test_atomic_fn.atomic_xor_local_uint",
    };

    log_info("test_atomics\n");
    return test_suite(device, size_t_width, folder, test_name, sizeof(test_name) / sizeof(const char *), "");
}


bool test_basic (cl_device_id device, cl_uint size_t_width, const char *folder)
{
    static const char* test_name[] = {
        "sample_kernel.work_item_functions",
        "test_sizeof.sizeof_char",
        "test_sizeof.sizeof_uchar",
        "test_sizeof.sizeof_unsigned_char",
        "test_sizeof.sizeof_short",
        "test_sizeof.sizeof_ushort",
        "test_sizeof.sizeof_unsigned_short",
        "test_sizeof.sizeof_int",
        "test_sizeof.sizeof_uint",
        "test_sizeof.sizeof_unsigned_int",
        "test_sizeof.sizeof_float",
        "test_sizeof.sizeof_long",
        "test_sizeof.sizeof_ulong",
        "test_sizeof.sizeof_unsigned_long",
        "test_sizeof.sizeof_char2",
        "test_sizeof.sizeof_uchar2",
        "test_sizeof.sizeof_short2",
        "test_sizeof.sizeof_ushort2",
        "test_sizeof.sizeof_int2",
        "test_sizeof.sizeof_uint2",
        "test_sizeof.sizeof_float2",
        "test_sizeof.sizeof_long2",
        "test_sizeof.sizeof_ulong2",
        "test_sizeof.sizeof_char4",
        "test_sizeof.sizeof_uchar4",
        "test_sizeof.sizeof_short4",
        "test_sizeof.sizeof_ushort4",
        "test_sizeof.sizeof_int4",
        "test_sizeof.sizeof_uint4",
        "test_sizeof.sizeof_float4",
        "test_sizeof.sizeof_long4",
        "test_sizeof.sizeof_ulong4",
        "test_sizeof.sizeof_char8",
        "test_sizeof.sizeof_uchar8",
        "test_sizeof.sizeof_short8",
        "test_sizeof.sizeof_ushort8",
        "test_sizeof.sizeof_int8",
        "test_sizeof.sizeof_uint8",
        "test_sizeof.sizeof_float8",
        "test_sizeof.sizeof_long8",
        "test_sizeof.sizeof_ulong8",
        "test_sizeof.sizeof_char16",
        "test_sizeof.sizeof_uchar16",
        "test_sizeof.sizeof_short16",
        "test_sizeof.sizeof_ushort16",
        "test_sizeof.sizeof_int16",
        "test_sizeof.sizeof_uint16",
        "test_sizeof.sizeof_float16",
        "test_sizeof.sizeof_long16",
        "test_sizeof.sizeof_ulong16",
        "test_sizeof.sizeof_void_p",
        "test_sizeof.sizeof_size_t",
        "test_sizeof.sizeof_sizeof_int",
        "test_sizeof.sizeof_ptrdiff_t",
        "test_sizeof.sizeof_intptr_t",
        "test_sizeof.sizeof_uintptr_t",
        "test_sizeof.sizeof_image2d_t",
        "test_sizeof.sizeof_image3d_t",
        "test_sizeof.sizeof_double",
        "test_sizeof.sizeof_double2",
        "test_sizeof.sizeof_double4",
        "test_sizeof.sizeof_double8",
        "test_sizeof.sizeof_double16",
        "sample_test.vec_type_hint_char",
        "sample_test.vec_type_hint_char2",
        "sample_test.vec_type_hint_char4",
        "sample_test.vec_type_hint_char8",
        "sample_test.vec_type_hint_char16",
        "sample_test.vec_type_hint_uchar",
        "sample_test.vec_type_hint_uchar2",
        "sample_test.vec_type_hint_uchar4",
        "sample_test.vec_type_hint_uchar8",
        "sample_test.vec_type_hint_uchar16",
        "sample_test.vec_type_hint_short",
        "sample_test.vec_type_hint_short2",
        "sample_test.vec_type_hint_short4",
        "sample_test.vec_type_hint_short8",
        "sample_test.vec_type_hint_short16",
        "sample_test.vec_type_hint_ushort",
        "sample_test.vec_type_hint_ushort2",
        "sample_test.vec_type_hint_ushort4",
        "sample_test.vec_type_hint_ushort8",
        "sample_test.vec_type_hint_ushort16",
        "sample_test.vec_type_hint_int",
        "sample_test.vec_type_hint_int2",
        "sample_test.vec_type_hint_int4",
        "sample_test.vec_type_hint_int8",
        "sample_test.vec_type_hint_int16",
        "sample_test.vec_type_hint_uint",
        "sample_test.vec_type_hint_uint2",
        "sample_test.vec_type_hint_uint4",
        "sample_test.vec_type_hint_uint8",
        "sample_test.vec_type_hint_uint16",
        "sample_test.vec_type_hint_long",
        "sample_test.vec_type_hint_long2",
        "sample_test.vec_type_hint_long4",
        "sample_test.vec_type_hint_long8",
        "sample_test.vec_type_hint_long16",
        "sample_test.vec_type_hint_ulong",
        "sample_test.vec_type_hint_ulong2",
        "sample_test.vec_type_hint_ulong4",
        "sample_test.vec_type_hint_ulong8",
        "sample_test.vec_type_hint_ulong16",
        "sample_test.vec_type_hint_float",
        "sample_test.vec_type_hint_float2",
        "sample_test.vec_type_hint_float4",
        "sample_test.vec_type_hint_float8",
        "sample_test.vec_type_hint_float16",
        "test.kernel_memory_alignment_private_char",
        "test.kernel_memory_alignment_private_uchar",
        "test.kernel_memory_alignment_private_short",
        "test.kernel_memory_alignment_private_ushort",
        "test.kernel_memory_alignment_private_int",
        "test.kernel_memory_alignment_private_uint",
        "test.kernel_memory_alignment_private_long",
        "test.kernel_memory_alignment_private_ulong",
        "test.kernel_memory_alignment_private_float",
        "test_fn.vload_global_char2",
        "test_fn.vload_global_char3",
        "test_fn.vload_global_char4",
        "test_fn.vload_global_char8",
        "test_fn.vload_global_char16",
        "test_fn.vload_global_uchar2",
        "test_fn.vload_global_uchar3",
        "test_fn.vload_global_uchar4",
        "test_fn.vload_global_uchar8",
        "test_fn.vload_global_uchar16",
        "test_fn.vload_global_short2",
        "test_fn.vload_global_short3",
        "test_fn.vload_global_short4",
        "test_fn.vload_global_short8",
        "test_fn.vload_global_short16",
        "test_fn.vload_global_ushort2",
        "test_fn.vload_global_ushort3",
        "test_fn.vload_global_ushort4",
        "test_fn.vload_global_ushort8",
        "test_fn.vload_global_ushort16",
        "test_fn.vload_global_int2",
        "test_fn.vload_global_int3",
        "test_fn.vload_global_int4",
        "test_fn.vload_global_int8",
        "test_fn.vload_global_int16",
        "test_fn.vload_global_uint2",
        "test_fn.vload_global_uint3",
        "test_fn.vload_global_uint4",
        "test_fn.vload_global_uint8",
        "test_fn.vload_global_uint16",
        "test_fn.vload_global_long2",
        "test_fn.vload_global_long3",
        "test_fn.vload_global_long4",
        "test_fn.vload_global_long8",
        "test_fn.vload_global_long16",
        "test_fn.vload_global_ulong2",
        "test_fn.vload_global_ulong3",
        "test_fn.vload_global_ulong4",
        "test_fn.vload_global_ulong8",
        "test_fn.vload_global_ulong16",
        "test_fn.vload_global_float2",
        "test_fn.vload_global_float3",
        "test_fn.vload_global_float4",
        "test_fn.vload_global_float8",
        "test_fn.vload_global_float16",
        "test_fn.vload_constant_char2",
        "test_fn.vload_constant_char3",
        "test_fn.vload_constant_char4",
        "test_fn.vload_constant_char8",
        "test_fn.vload_constant_char16",
        "test_fn.vload_constant_uchar2",
        "test_fn.vload_constant_uchar3",
        "test_fn.vload_constant_uchar4",
        "test_fn.vload_constant_uchar8",
        "test_fn.vload_constant_uchar16",
        "test_fn.vload_constant_short2",
        "test_fn.vload_constant_short3",
        "test_fn.vload_constant_short4",
        "test_fn.vload_constant_short8",
        "test_fn.vload_constant_short16",
        "test_fn.vload_constant_ushort2",
        "test_fn.vload_constant_ushort3",
        "test_fn.vload_constant_ushort4",
        "test_fn.vload_constant_ushort8",
        "test_fn.vload_constant_ushort16",
        "test_fn.vload_constant_int2",
        "test_fn.vload_constant_int3",
        "test_fn.vload_constant_int4",
        "test_fn.vload_constant_int8",
        "test_fn.vload_constant_int16",
        "test_fn.vload_constant_uint2",
        "test_fn.vload_constant_uint3",
        "test_fn.vload_constant_uint4",
        "test_fn.vload_constant_uint8",
        "test_fn.vload_constant_uint16",
        "test_fn.vload_constant_long2",
        "test_fn.vload_constant_long3",
        "test_fn.vload_constant_long4",
        "test_fn.vload_constant_long8",
        "test_fn.vload_constant_long16",
        "test_fn.vload_constant_ulong2",
        "test_fn.vload_constant_ulong3",
        "test_fn.vload_constant_ulong4",
        "test_fn.vload_constant_ulong8",
        "test_fn.vload_constant_ulong16",
        "test_fn.vload_constant_float2",
        "test_fn.vload_constant_float3",
        "test_fn.vload_constant_float4",
        "test_fn.vload_constant_float8",
        "test_fn.vload_constant_float16",
        "test_fn.vload_private_char2",
        "test_fn.vload_private_char3",
        "test_fn.vload_private_char4",
        "test_fn.vload_private_char8",
        "test_fn.vload_private_char16",
        "test_fn.vload_private_uchar2",
        "test_fn.vload_private_uchar3",
        "test_fn.vload_private_uchar4",
        "test_fn.vload_private_uchar8",
        "test_fn.vload_private_uchar16",
        "test_fn.vload_private_short2",
        "test_fn.vload_private_short3",
        "test_fn.vload_private_short4",
        "test_fn.vload_private_short8",
        "test_fn.vload_private_short16",
        "test_fn.vload_private_ushort2",
        "test_fn.vload_private_ushort3",
        "test_fn.vload_private_ushort4",
        "test_fn.vload_private_ushort8",
        "test_fn.vload_private_ushort16",
        "test_fn.vload_private_int2",
        "test_fn.vload_private_int3",
        "test_fn.vload_private_int4",
        "test_fn.vload_private_int8",
        "test_fn.vload_private_int16",
        "test_fn.vload_private_uint2",
        "test_fn.vload_private_uint3",
        "test_fn.vload_private_uint4",
        "test_fn.vload_private_uint8",
        "test_fn.vload_private_uint16",
        "test_fn.vload_private_long2",
        "test_fn.vload_private_long3",
        "test_fn.vload_private_long4",
        "test_fn.vload_private_long8",
        "test_fn.vload_private_long16",
        "test_fn.vload_private_ulong2",
        "test_fn.vload_private_ulong3",
        "test_fn.vload_private_ulong4",
        "test_fn.vload_private_ulong8",
        "test_fn.vload_private_ulong16",
        "test_fn.vload_private_float2",
        "test_fn.vload_private_float3",
        "test_fn.vload_private_float4",
        "test_fn.vload_private_float8",
        "test_fn.vload_private_float16",
        "test_fn.vload_local_char2",
        "test_fn.vload_local_char3",
        "test_fn.vload_local_char4",
        "test_fn.vload_local_char8",
        "test_fn.vload_local_char16",
        "test_fn.vload_local_uchar2",
        "test_fn.vload_local_uchar3",
        "test_fn.vload_local_uchar4",
        "test_fn.vload_local_uchar8",
        "test_fn.vload_local_uchar16",
        "test_fn.vload_local_short2",
        "test_fn.vload_local_short3",
        "test_fn.vload_local_short4",
        "test_fn.vload_local_short8",
        "test_fn.vload_local_short16",
        "test_fn.vload_local_ushort2",
        "test_fn.vload_local_ushort3",
        "test_fn.vload_local_ushort4",
        "test_fn.vload_local_ushort8",
        "test_fn.vload_local_ushort16",
        "test_fn.vload_local_int2",
        "test_fn.vload_local_int3",
        "test_fn.vload_local_int4",
        "test_fn.vload_local_int8",
        "test_fn.vload_local_int16",
        "test_fn.vload_local_uint2",
        "test_fn.vload_local_uint3",
        "test_fn.vload_local_uint4",
        "test_fn.vload_local_uint8",
        "test_fn.vload_local_uint16",
        "test_fn.vload_local_long2",
        "test_fn.vload_local_long3",
        "test_fn.vload_local_long4",
        "test_fn.vload_local_long8",
        "test_fn.vload_local_long16",
        "test_fn.vload_local_ulong2",
        "test_fn.vload_local_ulong3",
        "test_fn.vload_local_ulong4",
        "test_fn.vload_local_ulong8",
        "test_fn.vload_local_ulong16",
        "test_fn.vload_local_float2",
        "test_fn.vload_local_float3",
        "test_fn.vload_local_float4",
        "test_fn.vload_local_float8",
        "test_fn.vload_local_float16",
        "test_fn.vstore_global_char2",
        "test_fn.vstore_global_char3",
        "test_fn.vstore_global_char4",
        "test_fn.vstore_global_char8",
        "test_fn.vstore_global_char16",
        "test_fn.vstore_global_uchar2",
        "test_fn.vstore_global_uchar3",
        "test_fn.vstore_global_uchar4",
        "test_fn.vstore_global_uchar8",
        "test_fn.vstore_global_uchar16",
        "test_fn.vstore_global_short2",
        "test_fn.vstore_global_short3",
        "test_fn.vstore_global_short4",
        "test_fn.vstore_global_short8",
        "test_fn.vstore_global_short16",
        "test_fn.vstore_global_ushort2",
        "test_fn.vstore_global_ushort3",
        "test_fn.vstore_global_ushort4",
        "test_fn.vstore_global_ushort8",
        "test_fn.vstore_global_ushort16",
        "test_fn.vstore_global_int2",
        "test_fn.vstore_global_int3",
        "test_fn.vstore_global_int4",
        "test_fn.vstore_global_int8",
        "test_fn.vstore_global_int16",
        "test_fn.vstore_global_uint2",
        "test_fn.vstore_global_uint3",
        "test_fn.vstore_global_uint4",
        "test_fn.vstore_global_uint8",
        "test_fn.vstore_global_uint16",
        "test_fn.vstore_global_long2",
        "test_fn.vstore_global_long3",
        "test_fn.vstore_global_long4",
        "test_fn.vstore_global_long8",
        "test_fn.vstore_global_long16",
        "test_fn.vstore_global_ulong2",
        "test_fn.vstore_global_ulong3",
        "test_fn.vstore_global_ulong4",
        "test_fn.vstore_global_ulong8",
        "test_fn.vstore_global_ulong16",
        "test_fn.vstore_global_float2",
        "test_fn.vstore_global_float3",
        "test_fn.vstore_global_float4",
        "test_fn.vstore_global_float8",
        "test_fn.vstore_global_float16",
        "test_fn.vstore_local_char2",
        "test_fn.vstore_local_char3",
        "test_fn.vstore_local_char4",
        "test_fn.vstore_local_char8",
        "test_fn.vstore_local_char16",
        "test_fn.vstore_local_uchar2",
        "test_fn.vstore_local_uchar3",
        "test_fn.vstore_local_uchar4",
        "test_fn.vstore_local_uchar8",
        "test_fn.vstore_local_uchar16",
        "test_fn.vstore_local_short2",
        "test_fn.vstore_local_short3",
        "test_fn.vstore_local_short4",
        "test_fn.vstore_local_short8",
        "test_fn.vstore_local_short16",
        "test_fn.vstore_local_ushort2",
        "test_fn.vstore_local_ushort3",
        "test_fn.vstore_local_ushort4",
        "test_fn.vstore_local_ushort8",
        "test_fn.vstore_local_ushort16",
        "test_fn.vstore_local_int2",
        "test_fn.vstore_local_int3",
        "test_fn.vstore_local_int4",
        "test_fn.vstore_local_int8",
        "test_fn.vstore_local_int16",
        "test_fn.vstore_local_uint2",
        "test_fn.vstore_local_uint3",
        "test_fn.vstore_local_uint4",
        "test_fn.vstore_local_uint8",
        "test_fn.vstore_local_uint16",
        "test_fn.vstore_local_long2",
        "test_fn.vstore_local_long3",
        "test_fn.vstore_local_long4",
        "test_fn.vstore_local_long8",
        "test_fn.vstore_local_long16",
        "test_fn.vstore_local_ulong2",
        "test_fn.vstore_local_ulong3",
        "test_fn.vstore_local_ulong4",
        "test_fn.vstore_local_ulong8",
        "test_fn.vstore_local_ulong16",
        "test_fn.vstore_local_float2",
        "test_fn.vstore_local_float3",
        "test_fn.vstore_local_float4",
        "test_fn.vstore_local_float8",
        "test_fn.vstore_local_float16",
        "test_fn.vstore_private_char2",
        "test_fn.vstore_private_char3",
        "test_fn.vstore_private_char4",
        "test_fn.vstore_private_char8",
        "test_fn.vstore_private_char16",
        "test_fn.vstore_private_uchar2",
        "test_fn.vstore_private_uchar3",
        "test_fn.vstore_private_uchar4",
        "test_fn.vstore_private_uchar8",
        "test_fn.vstore_private_uchar16",
        "test_fn.vstore_private_short2",
        "test_fn.vstore_private_short3",
        "test_fn.vstore_private_short4",
        "test_fn.vstore_private_short8",
        "test_fn.vstore_private_short16",
        "test_fn.vstore_private_ushort2",
        "test_fn.vstore_private_ushort3",
        "test_fn.vstore_private_ushort4",
        "test_fn.vstore_private_ushort8",
        "test_fn.vstore_private_ushort16",
        "test_fn.vstore_private_int2",
        "test_fn.vstore_private_int3",
        "test_fn.vstore_private_int4",
        "test_fn.vstore_private_int8",
        "test_fn.vstore_private_int16",
        "test_fn.vstore_private_uint2",
        "test_fn.vstore_private_uint3",
        "test_fn.vstore_private_uint4",
        "test_fn.vstore_private_uint8",
        "test_fn.vstore_private_uint16",
        "test_fn.vstore_private_long2",
        "test_fn.vstore_private_long3",
        "test_fn.vstore_private_long4",
        "test_fn.vstore_private_long8",
        "test_fn.vstore_private_long16",
        "test_fn.vstore_private_ulong2",
        "test_fn.vstore_private_ulong3",
        "test_fn.vstore_private_ulong4",
        "test_fn.vstore_private_ulong8",
        "test_fn.vstore_private_ulong16",
        "test_fn.vstore_private_float2",
        "test_fn.vstore_private_float3",
        "test_fn.vstore_private_float4",
        "test_fn.vstore_private_float8",
        "test_fn.vstore_private_float16",
        "test_fn.async_copy_global_to_local_char",
        "test_fn.async_copy_global_to_local_char2",
        "test_fn.async_copy_global_to_local_char4",
        "test_fn.async_copy_global_to_local_char8",
        "test_fn.async_copy_global_to_local_char16",
        "test_fn.async_copy_global_to_local_uchar",
        "test_fn.async_copy_global_to_local_uchar2",
        "test_fn.async_copy_global_to_local_uchar4",
        "test_fn.async_copy_global_to_local_uchar8",
        "test_fn.async_copy_global_to_local_uchar16",
        "test_fn.async_copy_global_to_local_short",
        "test_fn.async_copy_global_to_local_short2",
        "test_fn.async_copy_global_to_local_short4",
        "test_fn.async_copy_global_to_local_short8",
        "test_fn.async_copy_global_to_local_short16",
        "test_fn.async_copy_global_to_local_ushort",
        "test_fn.async_copy_global_to_local_ushort2",
        "test_fn.async_copy_global_to_local_ushort4",
        "test_fn.async_copy_global_to_local_ushort8",
        "test_fn.async_copy_global_to_local_ushort16",
        "test_fn.async_copy_global_to_local_int",
        "test_fn.async_copy_global_to_local_int2",
        "test_fn.async_copy_global_to_local_int4",
        "test_fn.async_copy_global_to_local_int8",
        "test_fn.async_copy_global_to_local_int16",
        "test_fn.async_copy_global_to_local_uint",
        "test_fn.async_copy_global_to_local_uint2",
        "test_fn.async_copy_global_to_local_uint4",
        "test_fn.async_copy_global_to_local_uint8",
        "test_fn.async_copy_global_to_local_uint16",
        "test_fn.async_copy_global_to_local_long",
        "test_fn.async_copy_global_to_local_long2",
        "test_fn.async_copy_global_to_local_long4",
        "test_fn.async_copy_global_to_local_long8",
        "test_fn.async_copy_global_to_local_long16",
        "test_fn.async_copy_global_to_local_ulong",
        "test_fn.async_copy_global_to_local_ulong2",
        "test_fn.async_copy_global_to_local_ulong4",
        "test_fn.async_copy_global_to_local_ulong8",
        "test_fn.async_copy_global_to_local_ulong16",
        "test_fn.async_copy_global_to_local_float",
        "test_fn.async_copy_global_to_local_float2",
        "test_fn.async_copy_global_to_local_float4",
        "test_fn.async_copy_global_to_local_float8",
        "test_fn.async_copy_global_to_local_float16",
        "test_fn.async_copy_global_to_local_double",
        "test_fn.async_copy_global_to_local_double2",
        "test_fn.async_copy_global_to_local_double4",
        "test_fn.async_copy_global_to_local_double8",
        "test_fn.async_copy_global_to_local_double16",
        "test_fn.async_copy_local_to_global_char",
        "test_fn.async_copy_local_to_global_char2",
        "test_fn.async_copy_local_to_global_char4",
        "test_fn.async_copy_local_to_global_char8",
        "test_fn.async_copy_local_to_global_char16",
        "test_fn.async_copy_local_to_global_uchar",
        "test_fn.async_copy_local_to_global_uchar2",
        "test_fn.async_copy_local_to_global_uchar4",
        "test_fn.async_copy_local_to_global_uchar8",
        "test_fn.async_copy_local_to_global_uchar16",
        "test_fn.async_copy_local_to_global_short",
        "test_fn.async_copy_local_to_global_short2",
        "test_fn.async_copy_local_to_global_short4",
        "test_fn.async_copy_local_to_global_short8",
        "test_fn.async_copy_local_to_global_short16",
        "test_fn.async_copy_local_to_global_ushort",
        "test_fn.async_copy_local_to_global_ushort2",
        "test_fn.async_copy_local_to_global_ushort4",
        "test_fn.async_copy_local_to_global_ushort8",
        "test_fn.async_copy_local_to_global_ushort16",
        "test_fn.async_copy_local_to_global_int",
        "test_fn.async_copy_local_to_global_int2",
        "test_fn.async_copy_local_to_global_int4",
        "test_fn.async_copy_local_to_global_int8",
        "test_fn.async_copy_local_to_global_int16",
        "test_fn.async_copy_local_to_global_uint",
        "test_fn.async_copy_local_to_global_uint2",
        "test_fn.async_copy_local_to_global_uint4",
        "test_fn.async_copy_local_to_global_uint8",
        "test_fn.async_copy_local_to_global_uint16",
        "test_fn.async_copy_local_to_global_long",
        "test_fn.async_copy_local_to_global_long2",
        "test_fn.async_copy_local_to_global_long4",
        "test_fn.async_copy_local_to_global_long8",
        "test_fn.async_copy_local_to_global_long16",
        "test_fn.async_copy_local_to_global_ulong",
        "test_fn.async_copy_local_to_global_ulong2",
        "test_fn.async_copy_local_to_global_ulong4",
        "test_fn.async_copy_local_to_global_ulong8",
        "test_fn.async_copy_local_to_global_ulong16",
        "test_fn.async_copy_local_to_global_float",
        "test_fn.async_copy_local_to_global_float2",
        "test_fn.async_copy_local_to_global_float4",
        "test_fn.async_copy_local_to_global_float8",
        "test_fn.async_copy_local_to_global_float16",
        "test_fn.async_strided_copy_global_to_local_char",
        "test_fn.async_strided_copy_global_to_local_char2",
        "test_fn.async_strided_copy_global_to_local_char4",
        "test_fn.async_strided_copy_global_to_local_char8",
        "test_fn.async_strided_copy_global_to_local_char16",
        "test_fn.async_strided_copy_global_to_local_uchar",
        "test_fn.async_strided_copy_global_to_local_uchar2",
        "test_fn.async_strided_copy_global_to_local_uchar4",
        "test_fn.async_strided_copy_global_to_local_uchar8",
        "test_fn.async_strided_copy_global_to_local_uchar16",
        "test_fn.async_strided_copy_global_to_local_short",
        "test_fn.async_strided_copy_global_to_local_short2",
        "test_fn.async_strided_copy_global_to_local_short4",
        "test_fn.async_strided_copy_global_to_local_short8",
        "test_fn.async_strided_copy_global_to_local_short16",
        "test_fn.async_strided_copy_global_to_local_ushort",
        "test_fn.async_strided_copy_global_to_local_ushort2",
        "test_fn.async_strided_copy_global_to_local_ushort4",
        "test_fn.async_strided_copy_global_to_local_ushort8",
        "test_fn.async_strided_copy_global_to_local_ushort16",
        "test_fn.async_strided_copy_global_to_local_int",
        "test_fn.async_strided_copy_global_to_local_int2",
        "test_fn.async_strided_copy_global_to_local_int4",
        "test_fn.async_strided_copy_global_to_local_int8",
        "test_fn.async_strided_copy_global_to_local_int16",
        "test_fn.async_strided_copy_global_to_local_uint",
        "test_fn.async_strided_copy_global_to_local_uint2",
        "test_fn.async_strided_copy_global_to_local_uint4",
        "test_fn.async_strided_copy_global_to_local_uint8",
        "test_fn.async_strided_copy_global_to_local_uint16",
        "test_fn.async_strided_copy_global_to_local_long",
        "test_fn.async_strided_copy_global_to_local_long2",
        "test_fn.async_strided_copy_global_to_local_long4",
        "test_fn.async_strided_copy_global_to_local_long8",
        "test_fn.async_strided_copy_global_to_local_long16",
        "test_fn.async_strided_copy_global_to_local_ulong",
        "test_fn.async_strided_copy_global_to_local_ulong2",
        "test_fn.async_strided_copy_global_to_local_ulong4",
        "test_fn.async_strided_copy_global_to_local_ulong8",
        "test_fn.async_strided_copy_global_to_local_ulong16",
        "test_fn.async_strided_copy_global_to_local_float",
        "test_fn.async_strided_copy_global_to_local_float2",
        "test_fn.async_strided_copy_global_to_local_float4",
        "test_fn.async_strided_copy_global_to_local_float8",
        "test_fn.async_strided_copy_global_to_local_float16",
        "test_fn.async_strided_copy_local_to_global_char",
        "test_fn.async_strided_copy_local_to_global_char2",
        "test_fn.async_strided_copy_local_to_global_char4",
        "test_fn.async_strided_copy_local_to_global_char8",
        "test_fn.async_strided_copy_local_to_global_char16",
        "test_fn.async_strided_copy_local_to_global_uchar",
        "test_fn.async_strided_copy_local_to_global_uchar2",
        "test_fn.async_strided_copy_local_to_global_uchar4",
        "test_fn.async_strided_copy_local_to_global_uchar8",
        "test_fn.async_strided_copy_local_to_global_uchar16",
        "test_fn.async_strided_copy_local_to_global_short",
        "test_fn.async_strided_copy_local_to_global_short2",
        "test_fn.async_strided_copy_local_to_global_short4",
        "test_fn.async_strided_copy_local_to_global_short8",
        "test_fn.async_strided_copy_local_to_global_short16",
        "test_fn.async_strided_copy_local_to_global_ushort",
        "test_fn.async_strided_copy_local_to_global_ushort2",
        "test_fn.async_strided_copy_local_to_global_ushort4",
        "test_fn.async_strided_copy_local_to_global_ushort8",
        "test_fn.async_strided_copy_local_to_global_ushort16",
        "test_fn.async_strided_copy_local_to_global_int",
        "test_fn.async_strided_copy_local_to_global_int2",
        "test_fn.async_strided_copy_local_to_global_int4",
        "test_fn.async_strided_copy_local_to_global_int8",
        "test_fn.async_strided_copy_local_to_global_int16",
        "test_fn.async_strided_copy_local_to_global_uint",
        "test_fn.async_strided_copy_local_to_global_uint2",
        "test_fn.async_strided_copy_local_to_global_uint4",
        "test_fn.async_strided_copy_local_to_global_uint8",
        "test_fn.async_strided_copy_local_to_global_uint16",
        "test_fn.async_strided_copy_local_to_global_long",
        "test_fn.async_strided_copy_local_to_global_long2",
        "test_fn.async_strided_copy_local_to_global_long4",
        "test_fn.async_strided_copy_local_to_global_long8",
        "test_fn.async_strided_copy_local_to_global_long16",
        "test_fn.async_strided_copy_local_to_global_ulong",
        "test_fn.async_strided_copy_local_to_global_ulong2",
        "test_fn.async_strided_copy_local_to_global_ulong4",
        "test_fn.async_strided_copy_local_to_global_ulong8",
        "test_fn.async_strided_copy_local_to_global_ulong16",
        "test_fn.async_strided_copy_local_to_global_float",
        "test_fn.async_strided_copy_local_to_global_float2",
        "test_fn.async_strided_copy_local_to_global_float4",
        "test_fn.async_strided_copy_local_to_global_float8",
        "test_fn.async_strided_copy_local_to_global_float16",
        "test_fn.prefetch_char",
        "test_fn.prefetch_char2",
        "test_fn.prefetch_char4",
        "test_fn.prefetch_char8",
        "test_fn.prefetch_char16",
        "test_fn.prefetch_uchar",
        "test_fn.prefetch_uchar2",
        "test_fn.prefetch_uchar4",
        "test_fn.prefetch_uchar8",
        "test_fn.prefetch_uchar16",
        "test_fn.prefetch_short",
        "test_fn.prefetch_short2",
        "test_fn.prefetch_short4",
        "test_fn.prefetch_short8",
        "test_fn.prefetch_short16",
        "test_fn.prefetch_ushort",
        "test_fn.prefetch_ushort2",
        "test_fn.prefetch_ushort4",
        "test_fn.prefetch_ushort8",
        "test_fn.prefetch_ushort16",
        "test_fn.prefetch_int",
        "test_fn.prefetch_int2",
        "test_fn.prefetch_int4",
        "test_fn.prefetch_int8",
        "test_fn.prefetch_int16",
        "test_fn.prefetch_uint",
        "test_fn.prefetch_uint2",
        "test_fn.prefetch_uint4",
        "test_fn.prefetch_uint8",
        "test_fn.prefetch_uint16",
        "test_fn.prefetch_long",
        "test_fn.prefetch_long2",
        "test_fn.prefetch_long4",
        "test_fn.prefetch_long8",
        "test_fn.prefetch_long16",
        "test_fn.prefetch_ulong",
        "test_fn.prefetch_ulong2",
        "test_fn.prefetch_ulong4",
        "test_fn.prefetch_ulong8",
        "test_fn.prefetch_ulong16",
        "test_fn.prefetch_float",
        "test_fn.prefetch_float2",
        "test_fn.prefetch_float4",
        "test_fn.prefetch_float8",
        "test_fn.prefetch_float16",
    };

    log_info("test_basic\n");
    return test_suite(device, size_t_width, folder, test_name, sizeof(test_name) / sizeof(const char *), "");
}

bool test_basic_double (cl_device_id device, cl_uint size_t_width, const char *folder)
{
    static const char* test_name[] = {
        "sample_test.vec_type_hint_double",
        "sample_test.vec_type_hint_double2",
        "sample_test.vec_type_hint_double4",
        "sample_test.vec_type_hint_double8",
        "sample_test.vec_type_hint_double16",
        "test.kernel_memory_alignment_private_double",
        "test_fn.vload_global_double2",
        "test_fn.vload_global_double3",
        "test_fn.vload_global_double4",
        "test_fn.vload_global_double8",
        "test_fn.vload_global_double16",
        "test_fn.vload_constant_double2",
        "test_fn.vload_constant_double3",
        "test_fn.vload_constant_double4",
        "test_fn.vload_constant_double8",
        "test_fn.vload_constant_double16",
        "test_fn.vstore_global_double2",
        "test_fn.vstore_global_double3",
        "test_fn.vstore_global_double4",
        "test_fn.vstore_global_double8",
        "test_fn.vstore_global_double16",
        "test_fn.vload_local_double2",
        "test_fn.vload_local_double3",
        "test_fn.vload_local_double4",
        "test_fn.vload_local_double8",
        "test_fn.vload_local_double16",
        "test_fn.vstore_global_double2",
        "test_fn.vstore_global_double3",
        "test_fn.vstore_global_double4",
        "test_fn.vstore_global_double8",
        "test_fn.vstore_global_double16",
        "test_fn.vstore_local_double2",
        "test_fn.vstore_local_double3",
        "test_fn.vstore_local_double4",
        "test_fn.vstore_local_double8",
        "test_fn.vstore_local_double16",
        "test_fn.vstore_private_double2",
        "test_fn.vstore_private_double3",
        "test_fn.vstore_private_double4",
        "test_fn.vstore_private_double8",
        "test_fn.vstore_private_double16",
        "test_fn.async_copy_local_to_global_double",
        "test_fn.async_copy_local_to_global_double2",
        "test_fn.async_copy_local_to_global_double4",
        "test_fn.async_copy_local_to_global_double8",
        "test_fn.async_copy_local_to_global_double16",
        "test_fn.async_strided_copy_global_to_local_double",
        "test_fn.async_strided_copy_global_to_local_double2",
        "test_fn.async_strided_copy_global_to_local_double4",
        "test_fn.async_strided_copy_global_to_local_double8",
        "test_fn.async_strided_copy_global_to_local_double16",
        "test_fn.async_strided_copy_local_to_global_double",
        "test_fn.async_strided_copy_local_to_global_double2",
        "test_fn.async_strided_copy_local_to_global_double4",
        "test_fn.async_strided_copy_local_to_global_double8",
        "test_fn.async_strided_copy_local_to_global_double16",
        "test_fn.prefetch_double",
        "test_fn.prefetch_double2",
        "test_fn.prefetch_double4",
        "test_fn.prefetch_double8",
        "test_fn.prefetch_double16",
    };

    log_info("test_basic_double\n");
    return test_suite(device, size_t_width, folder, test_name, sizeof(test_name) / sizeof(const char *), "cl_khr_fp64");
}


bool test_commonfns (cl_device_id device, cl_uint size_t_width, const char *folder)
{
    static const char* test_name[] = {
        "test_clamp.test_clamp_float",
        "test_clamp.test_clamp_float2",
        "test_clamp.test_clamp_float4",
        "test_clamp.test_clamp_float8",
        "test_clamp.test_clamp_float16",
        "test_clamp.test_clamp_float3",
        "test_degrees",
        "test_degrees2",
        "test_degrees4",
        "test_degrees8",
        "test_degrees16",
        "test_degrees3",
        "test_fmax",
        "test_fmax2",
        "test_fmax4",
        "test_fmax8",
        "test_fmax16",
        "test_fmax3",
        "test_fmin",
        "test_fmin2",
        "test_fmin4",
        "test_fmin8",
        "test_fmin16",
        "test_fmin3",
        "test_fn.test_max_float",
        "test_fn.test_max_float2",
        "test_fn.test_max_float4",
        "test_fn.test_max_float8",
        "test_fn.test_max_float16",
        "test_fn.test_max_float3",
        "test_fn.test_min_float",
        "test_fn.test_min_float2",
        "test_fn.test_min_float4",
        "test_fn.test_min_float8",
        "test_fn.test_min_float16",
        "test_fn.test_min_float3",
        "test_mix",
        "test_radians",
        "test_radians2",
        "test_radians4",
        "test_radians8",
        "test_radians16",
        "test_radians3",
        "test_step",
        "test_step2",
        "test_step4",
        "test_step8",
        "test_step16",
        "test_step3",
        "test_smoothstep",
        "test_smoothstep2",
        "test_smoothstep4",
        "test_smoothstep8",
        "test_smoothstep16",
        "test_smoothstep3",
        "test_sign",
        "test_sign2",
        "test_sign4",
        "test_sign8",
        "test_sign16",
        "test_sign3",
    };

    log_info("test_commonfns\n");
    return test_suite(device, size_t_width, folder, test_name, sizeof(test_name) / sizeof(const char *), "");
}


bool test_commonfns_double (cl_device_id device, cl_uint size_t_width, const char *folder)
{
    static const char* test_name[] = {
        "test_clamp.test_clamp_double",
        "test_clamp.test_clamp_double2",
        "test_clamp.test_clamp_double4",
        "test_clamp.test_clamp_double8",
        "test_clamp.test_clamp_double16",
        "test_clamp.test_clamp_double3",
        "test_degrees_double",
        "test_degrees2_double",
        "test_degrees4_double",
        "test_degrees8_double",
        "test_degrees16_double",
        "test_degrees3_double",
        "test_fn.test_max_double",
        "test_fn.test_max_double2",
        "test_fn.test_max_double4",
        "test_fn.test_max_double8",
        "test_fn.test_max_double16",
        "test_fn.test_max_double3",
        "test_fn.test_min_double",
        "test_fn.test_min_double2",
        "test_fn.test_min_double4",
        "test_fn.test_min_double8",
        "test_fn.test_min_double16",
        "test_fn.test_min_double3",
        "test_radians_double",
        "test_radians2_double",
        "test_radians4_double",
        "test_radians8_double",
        "test_radians16_double",
        "test_radians3_double",
        "test_step_double",
        "test_step2_double",
        "test_step4_double",
        "test_step8_double",
        "test_step16_double",
        "test_step3_double",
        "test_sign_double",
        "test_sign2_double",
        "test_sign4_double",
        "test_sign8_double",
        "test_sign16_double",
        "test_sign3_double",
    };

    log_info("test_commonfns_double\n");
    return test_suite(device, size_t_width, folder, test_name, sizeof(test_name) / sizeof(const char *), "cl_khr_fp64");
}

bool test_conversions (cl_device_id device, cl_uint size_t_width, const char *folder)
{
    static const char* test_name[] = {
        "convert2_type_roundingmode_type_f",
        "convert3_type_roundingmode_type_f",
        "convert4_type_roundingmode_type_f",
        "convert8_type_roundingmode_type_f",
        "convert16_type_roundingmode_type_f",
        "test_implicit_uchar_uchar",
        "test_convert_uchar_uchar",
        "test_convert_uchar_rte_uchar",
        "test_convert_uchar_rtp_uchar",
        "test_convert_uchar_rtn_uchar",
        "test_convert_uchar_rtz_uchar",
        "test_convert_uchar_sat_uchar",
        "test_convert_uchar_sat_rte_uchar",
        "test_convert_uchar_sat_rtp_uchar",
        "test_convert_uchar_sat_rtn_uchar",
        "test_convert_uchar_sat_rtz_uchar",
        "test_implicit_uchar_char",
        "test_convert_uchar_char",
        "test_convert_uchar_rte_char",
        "test_convert_uchar_rtp_char",
        "test_convert_uchar_rtn_char",
        "test_convert_uchar_rtz_char",
        "test_convert_uchar_sat_char",
        "test_convert_uchar_sat_rte_char",
        "test_convert_uchar_sat_rtp_char",
        "test_convert_uchar_sat_rtn_char",
        "test_convert_uchar_sat_rtz_char",
        "test_implicit_uchar_ushort",
        "test_convert_uchar_ushort",
        "test_convert_uchar_rte_ushort",
        "test_convert_uchar_rtp_ushort",
        "test_convert_uchar_rtn_ushort",
        "test_convert_uchar_rtz_ushort",
        "test_convert_uchar_sat_ushort",
        "test_convert_uchar_sat_rte_ushort",
        "test_convert_uchar_sat_rtp_ushort",
        "test_convert_uchar_sat_rtn_ushort",
        "test_convert_uchar_sat_rtz_ushort",
        "test_implicit_uchar_short",
        "test_convert_uchar_short",
        "test_convert_uchar_rte_short",
        "test_convert_uchar_rtp_short",
        "test_convert_uchar_rtn_short",
        "test_convert_uchar_rtz_short",
        "test_convert_uchar_sat_short",
        "test_convert_uchar_sat_rte_short",
        "test_convert_uchar_sat_rtp_short",
        "test_convert_uchar_sat_rtn_short",
        "test_convert_uchar_sat_rtz_short",
        "test_implicit_uchar_uint",
        "test_convert_uchar_uint",
        "test_convert_uchar_rte_uint",
        "test_convert_uchar_rtp_uint",
        "test_convert_uchar_rtn_uint",
        "test_convert_uchar_rtz_uint",
        "test_convert_uchar_sat_uint",
        "test_convert_uchar_sat_rte_uint",
        "test_convert_uchar_sat_rtp_uint",
        "test_convert_uchar_sat_rtn_uint",
        "test_convert_uchar_sat_rtz_uint",
        "test_implicit_uchar_int",
        "test_convert_uchar_int",
        "test_convert_uchar_rte_int",
        "test_convert_uchar_rtp_int",
        "test_convert_uchar_rtn_int",
        "test_convert_uchar_rtz_int",
        "test_convert_uchar_sat_int",
        "test_convert_uchar_sat_rte_int",
        "test_convert_uchar_sat_rtp_int",
        "test_convert_uchar_sat_rtn_int",
        "test_convert_uchar_sat_rtz_int",
        "test_implicit_uchar_float",
        "test_convert_uchar_float",
        "test_convert_uchar_rte_float",
        "test_convert_uchar_rtp_float",
        "test_convert_uchar_rtn_float",
        "test_convert_uchar_rtz_float",
        "test_convert_uchar_sat_float",
        "test_convert_uchar_sat_rte_float",
        "test_convert_uchar_sat_rtp_float",
        "test_convert_uchar_sat_rtn_float",
        "test_convert_uchar_sat_rtz_float",
        "test_implicit_uchar_ulong",
        "test_convert_uchar_ulong",
        "test_convert_uchar_rte_ulong",
        "test_convert_uchar_rtp_ulong",
        "test_convert_uchar_rtn_ulong",
        "test_convert_uchar_rtz_ulong",
        "test_convert_uchar_sat_ulong",
        "test_convert_uchar_sat_rte_ulong",
        "test_convert_uchar_sat_rtp_ulong",
        "test_convert_uchar_sat_rtn_ulong",
        "test_convert_uchar_sat_rtz_ulong",
        "test_implicit_uchar_long",
        "test_convert_uchar_long",
        "test_convert_uchar_rte_long",
        "test_convert_uchar_rtp_long",
        "test_convert_uchar_rtn_long",
        "test_convert_uchar_rtz_long",
        "test_convert_uchar_sat_long",
        "test_convert_uchar_sat_rte_long",
        "test_convert_uchar_sat_rtp_long",
        "test_convert_uchar_sat_rtn_long",
        "test_convert_uchar_sat_rtz_long",
        "test_implicit_char_uchar",
        "test_convert_char_uchar",
        "test_convert_char_rte_uchar",
        "test_convert_char_rtp_uchar",
        "test_convert_char_rtn_uchar",
        "test_convert_char_rtz_uchar",
        "test_convert_char_sat_uchar",
        "test_convert_char_sat_rte_uchar",
        "test_convert_char_sat_rtp_uchar",
        "test_convert_char_sat_rtn_uchar",
        "test_convert_char_sat_rtz_uchar",
        "test_implicit_char_char",
        "test_convert_char_char",
        "test_convert_char_rte_char",
        "test_convert_char_rtp_char",
        "test_convert_char_rtn_char",
        "test_convert_char_rtz_char",
        "test_convert_char_sat_char",
        "test_convert_char_sat_rte_char",
        "test_convert_char_sat_rtp_char",
        "test_convert_char_sat_rtn_char",
        "test_convert_char_sat_rtz_char",
        "test_implicit_char_ushort",
        "test_convert_char_ushort",
        "test_convert_char_rte_ushort",
        "test_convert_char_rtp_ushort",
        "test_convert_char_rtn_ushort",
        "test_convert_char_rtz_ushort",
        "test_convert_char_sat_ushort",
        "test_convert_char_sat_rte_ushort",
        "test_convert_char_sat_rtp_ushort",
        "test_convert_char_sat_rtn_ushort",
        "test_convert_char_sat_rtz_ushort",
        "test_implicit_char_short",
        "test_convert_char_short",
        "test_convert_char_rte_short",
        "test_convert_char_rtp_short",
        "test_convert_char_rtn_short",
        "test_convert_char_rtz_short",
        "test_convert_char_sat_short",
        "test_convert_char_sat_rte_short",
        "test_convert_char_sat_rtp_short",
        "test_convert_char_sat_rtn_short",
        "test_convert_char_sat_rtz_short",
        "test_implicit_char_uint",
        "test_convert_char_uint",
        "test_convert_char_rte_uint",
        "test_convert_char_rtp_uint",
        "test_convert_char_rtn_uint",
        "test_convert_char_rtz_uint",
        "test_convert_char_sat_uint",
        "test_convert_char_sat_rte_uint",
        "test_convert_char_sat_rtp_uint",
        "test_convert_char_sat_rtn_uint",
        "test_convert_char_sat_rtz_uint",
        "test_implicit_char_int",
        "test_convert_char_int",
        "test_convert_char_rte_int",
        "test_convert_char_rtp_int",
        "test_convert_char_rtn_int",
        "test_convert_char_rtz_int",
        "test_convert_char_sat_int",
        "test_convert_char_sat_rte_int",
        "test_convert_char_sat_rtp_int",
        "test_convert_char_sat_rtn_int",
        "test_convert_char_sat_rtz_int",
        "test_implicit_char_float",
        "test_convert_char_float",
        "test_convert_char_rte_float",
        "test_convert_char_rtp_float",
        "test_convert_char_rtn_float",
        "test_convert_char_rtz_float",
        "test_convert_char_sat_float",
        "test_convert_char_sat_rte_float",
        "test_convert_char_sat_rtp_float",
        "test_convert_char_sat_rtn_float",
        "test_convert_char_sat_rtz_float",
        "test_implicit_char_ulong",
        "test_convert_char_ulong",
        "test_convert_char_rte_ulong",
        "test_convert_char_rtp_ulong",
        "test_convert_char_rtn_ulong",
        "test_convert_char_rtz_ulong",
        "test_convert_char_sat_ulong",
        "test_convert_char_sat_rte_ulong",
        "test_convert_char_sat_rtp_ulong",
        "test_convert_char_sat_rtn_ulong",
        "test_convert_char_sat_rtz_ulong",
        "test_implicit_char_long",
        "test_convert_char_long",
        "test_convert_char_rte_long",
        "test_convert_char_rtp_long",
        "test_convert_char_rtn_long",
        "test_convert_char_rtz_long",
        "test_convert_char_sat_long",
        "test_convert_char_sat_rte_long",
        "test_convert_char_sat_rtp_long",
        "test_convert_char_sat_rtn_long",
        "test_convert_char_sat_rtz_long",
        "test_implicit_ushort_uchar",
        "test_convert_ushort_uchar",
        "test_convert_ushort_rte_uchar",
        "test_convert_ushort_rtp_uchar",
        "test_convert_ushort_rtn_uchar",
        "test_convert_ushort_rtz_uchar",
        "test_convert_ushort_sat_uchar",
        "test_convert_ushort_sat_rte_uchar",
        "test_convert_ushort_sat_rtp_uchar",
        "test_convert_ushort_sat_rtn_uchar",
        "test_convert_ushort_sat_rtz_uchar",
        "test_implicit_ushort_char",
        "test_convert_ushort_char",
        "test_convert_ushort_rte_char",
        "test_convert_ushort_rtp_char",
        "test_convert_ushort_rtn_char",
        "test_convert_ushort_rtz_char",
        "test_convert_ushort_sat_char",
        "test_convert_ushort_sat_rte_char",
        "test_convert_ushort_sat_rtp_char",
        "test_convert_ushort_sat_rtn_char",
        "test_convert_ushort_sat_rtz_char",
        "test_implicit_ushort_ushort",
        "test_convert_ushort_ushort",
        "test_convert_ushort_rte_ushort",
        "test_convert_ushort_rtp_ushort",
        "test_convert_ushort_rtn_ushort",
        "test_convert_ushort_rtz_ushort",
        "test_convert_ushort_sat_ushort",
        "test_convert_ushort_sat_rte_ushort",
        "test_convert_ushort_sat_rtp_ushort",
        "test_convert_ushort_sat_rtn_ushort",
        "test_convert_ushort_sat_rtz_ushort",
        "test_implicit_ushort_short",
        "test_convert_ushort_short",
        "test_convert_ushort_rte_short",
        "test_convert_ushort_rtp_short",
        "test_convert_ushort_rtn_short",
        "test_convert_ushort_rtz_short",
        "test_convert_ushort_sat_short",
        "test_convert_ushort_sat_rte_short",
        "test_convert_ushort_sat_rtp_short",
        "test_convert_ushort_sat_rtn_short",
        "test_convert_ushort_sat_rtz_short",
        "test_implicit_ushort_uint",
        "test_convert_ushort_uint",
        "test_convert_ushort_rte_uint",
        "test_convert_ushort_rtp_uint",
        "test_convert_ushort_rtn_uint",
        "test_convert_ushort_rtz_uint",
        "test_convert_ushort_sat_uint",
        "test_convert_ushort_sat_rte_uint",
        "test_convert_ushort_sat_rtp_uint",
        "test_convert_ushort_sat_rtn_uint",
        "test_convert_ushort_sat_rtz_uint",
        "test_implicit_ushort_int",
        "test_convert_ushort_int",
        "test_convert_ushort_rte_int",
        "test_convert_ushort_rtp_int",
        "test_convert_ushort_rtn_int",
        "test_convert_ushort_rtz_int",
        "test_convert_ushort_sat_int",
        "test_convert_ushort_sat_rte_int",
        "test_convert_ushort_sat_rtp_int",
        "test_convert_ushort_sat_rtn_int",
        "test_convert_ushort_sat_rtz_int",
        "test_implicit_ushort_float",
        "test_convert_ushort_float",
        "test_convert_ushort_rte_float",
        "test_convert_ushort_rtp_float",
        "test_convert_ushort_rtn_float",
        "test_convert_ushort_rtz_float",
        "test_convert_ushort_sat_float",
        "test_convert_ushort_sat_rte_float",
        "test_convert_ushort_sat_rtp_float",
        "test_convert_ushort_sat_rtn_float",
        "test_convert_ushort_sat_rtz_float",
        "test_implicit_ushort_ulong",
        "test_convert_ushort_ulong",
        "test_convert_ushort_rte_ulong",
        "test_convert_ushort_rtp_ulong",
        "test_convert_ushort_rtn_ulong",
        "test_convert_ushort_rtz_ulong",
        "test_convert_ushort_sat_ulong",
        "test_convert_ushort_sat_rte_ulong",
        "test_convert_ushort_sat_rtp_ulong",
        "test_convert_ushort_sat_rtn_ulong",
        "test_convert_ushort_sat_rtz_ulong",
        "test_implicit_ushort_long",
        "test_convert_ushort_long",
        "test_convert_ushort_rte_long",
        "test_convert_ushort_rtp_long",
        "test_convert_ushort_rtn_long",
        "test_convert_ushort_rtz_long",
        "test_convert_ushort_sat_long",
        "test_convert_ushort_sat_rte_long",
        "test_convert_ushort_sat_rtp_long",
        "test_convert_ushort_sat_rtn_long",
        "test_convert_ushort_sat_rtz_long",
        "test_implicit_short_uchar",
        "test_convert_short_uchar",
        "test_convert_short_rte_uchar",
        "test_convert_short_rtp_uchar",
        "test_convert_short_rtn_uchar",
        "test_convert_short_rtz_uchar",
        "test_convert_short_sat_uchar",
        "test_convert_short_sat_rte_uchar",
        "test_convert_short_sat_rtp_uchar",
        "test_convert_short_sat_rtn_uchar",
        "test_convert_short_sat_rtz_uchar",
        "test_implicit_short_char",
        "test_convert_short_char",
        "test_convert_short_rte_char",
        "test_convert_short_rtp_char",
        "test_convert_short_rtn_char",
        "test_convert_short_rtz_char",
        "test_convert_short_sat_char",
        "test_convert_short_sat_rte_char",
        "test_convert_short_sat_rtp_char",
        "test_convert_short_sat_rtn_char",
        "test_convert_short_sat_rtz_char",
        "test_implicit_short_ushort",
        "test_convert_short_ushort",
        "test_convert_short_rte_ushort",
        "test_convert_short_rtp_ushort",
        "test_convert_short_rtn_ushort",
        "test_convert_short_rtz_ushort",
        "test_convert_short_sat_ushort",
        "test_convert_short_sat_rte_ushort",
        "test_convert_short_sat_rtp_ushort",
        "test_convert_short_sat_rtn_ushort",
        "test_convert_short_sat_rtz_ushort",
        "test_implicit_short_short",
        "test_convert_short_short",
        "test_convert_short_rte_short",
        "test_convert_short_rtp_short",
        "test_convert_short_rtn_short",
        "test_convert_short_rtz_short",
        "test_convert_short_sat_short",
        "test_convert_short_sat_rte_short",
        "test_convert_short_sat_rtp_short",
        "test_convert_short_sat_rtn_short",
        "test_convert_short_sat_rtz_short",
        "test_implicit_short_uint",
        "test_convert_short_uint",
        "test_convert_short_rte_uint",
        "test_convert_short_rtp_uint",
        "test_convert_short_rtn_uint",
        "test_convert_short_rtz_uint",
        "test_convert_short_sat_uint",
        "test_convert_short_sat_rte_uint",
        "test_convert_short_sat_rtp_uint",
        "test_convert_short_sat_rtn_uint",
        "test_convert_short_sat_rtz_uint",
        "test_implicit_short_int",
        "test_convert_short_int",
        "test_convert_short_rte_int",
        "test_convert_short_rtp_int",
        "test_convert_short_rtn_int",
        "test_convert_short_rtz_int",
        "test_convert_short_sat_int",
        "test_convert_short_sat_rte_int",
        "test_convert_short_sat_rtp_int",
        "test_convert_short_sat_rtn_int",
        "test_convert_short_sat_rtz_int",
        "test_implicit_short_float",
        "test_convert_short_float",
        "test_convert_short_rte_float",
        "test_convert_short_rtp_float",
        "test_convert_short_rtn_float",
        "test_convert_short_rtz_float",
        "test_convert_short_sat_float",
        "test_convert_short_sat_rte_float",
        "test_convert_short_sat_rtp_float",
        "test_convert_short_sat_rtn_float",
        "test_convert_short_sat_rtz_float",
        "test_implicit_short_ulong",
        "test_convert_short_ulong",
        "test_convert_short_rte_ulong",
        "test_convert_short_rtp_ulong",
        "test_convert_short_rtn_ulong",
        "test_convert_short_rtz_ulong",
        "test_convert_short_sat_ulong",
        "test_convert_short_sat_rte_ulong",
        "test_convert_short_sat_rtp_ulong",
        "test_convert_short_sat_rtn_ulong",
        "test_convert_short_sat_rtz_ulong",
        "test_implicit_short_long",
        "test_convert_short_long",
        "test_convert_short_rte_long",
        "test_convert_short_rtp_long",
        "test_convert_short_rtn_long",
        "test_convert_short_rtz_long",
        "test_convert_short_sat_long",
        "test_convert_short_sat_rte_long",
        "test_convert_short_sat_rtp_long",
        "test_convert_short_sat_rtn_long",
        "test_convert_short_sat_rtz_long",
        "test_implicit_uint_uchar",
        "test_convert_uint_uchar",
        "test_convert_uint_rte_uchar",
        "test_convert_uint_rtp_uchar",
        "test_convert_uint_rtn_uchar",
        "test_convert_uint_rtz_uchar",
        "test_convert_uint_sat_uchar",
        "test_convert_uint_sat_rte_uchar",
        "test_convert_uint_sat_rtp_uchar",
        "test_convert_uint_sat_rtn_uchar",
        "test_convert_uint_sat_rtz_uchar",
        "test_implicit_uint_char",
        "test_convert_uint_char",
        "test_convert_uint_rte_char",
        "test_convert_uint_rtp_char",
        "test_convert_uint_rtn_char",
        "test_convert_uint_rtz_char",
        "test_convert_uint_sat_char",
        "test_convert_uint_sat_rte_char",
        "test_convert_uint_sat_rtp_char",
        "test_convert_uint_sat_rtn_char",
        "test_convert_uint_sat_rtz_char",
        "test_implicit_uint_ushort",
        "test_convert_uint_ushort",
        "test_convert_uint_rte_ushort",
        "test_convert_uint_rtp_ushort",
        "test_convert_uint_rtn_ushort",
        "test_convert_uint_rtz_ushort",
        "test_convert_uint_sat_ushort",
        "test_convert_uint_sat_rte_ushort",
        "test_convert_uint_sat_rtp_ushort",
        "test_convert_uint_sat_rtn_ushort",
        "test_convert_uint_sat_rtz_ushort",
        "test_implicit_uint_short",
        "test_convert_uint_short",
        "test_convert_uint_rte_short",
        "test_convert_uint_rtp_short",
        "test_convert_uint_rtn_short",
        "test_convert_uint_rtz_short",
        "test_convert_uint_sat_short",
        "test_convert_uint_sat_rte_short",
        "test_convert_uint_sat_rtp_short",
        "test_convert_uint_sat_rtn_short",
        "test_convert_uint_sat_rtz_short",
        "test_implicit_uint_uint",
        "test_convert_uint_uint",
        "test_convert_uint_rte_uint",
        "test_convert_uint_rtp_uint",
        "test_convert_uint_rtn_uint",
        "test_convert_uint_rtz_uint",
        "test_convert_uint_sat_uint",
        "test_convert_uint_sat_rte_uint",
        "test_convert_uint_sat_rtp_uint",
        "test_convert_uint_sat_rtn_uint",
        "test_convert_uint_sat_rtz_uint",
        "test_implicit_uint_int",
        "test_convert_uint_int",
        "test_convert_uint_rte_int",
        "test_convert_uint_rtp_int",
        "test_convert_uint_rtn_int",
        "test_convert_uint_rtz_int",
        "test_convert_uint_sat_int",
        "test_convert_uint_sat_rte_int",
        "test_convert_uint_sat_rtp_int",
        "test_convert_uint_sat_rtn_int",
        "test_convert_uint_sat_rtz_int",
        "test_implicit_uint_float",
        "test_convert_uint_float",
        "test_convert_uint_rte_float",
        "test_convert_uint_rtp_float",
        "test_convert_uint_rtn_float",
        "test_convert_uint_rtz_float",
        "test_convert_uint_sat_float",
        "test_convert_uint_sat_rte_float",
        "test_convert_uint_sat_rtp_float",
        "test_convert_uint_sat_rtn_float",
        "test_convert_uint_sat_rtz_float",
        "test_implicit_uint_ulong",
        "test_convert_uint_ulong",
        "test_convert_uint_rte_ulong",
        "test_convert_uint_rtp_ulong",
        "test_convert_uint_rtn_ulong",
        "test_convert_uint_rtz_ulong",
        "test_convert_uint_sat_ulong",
        "test_convert_uint_sat_rte_ulong",
        "test_convert_uint_sat_rtp_ulong",
        "test_convert_uint_sat_rtn_ulong",
        "test_convert_uint_sat_rtz_ulong",
        "test_implicit_uint_long",
        "test_convert_uint_long",
        "test_convert_uint_rte_long",
        "test_convert_uint_rtp_long",
        "test_convert_uint_rtn_long",
        "test_convert_uint_rtz_long",
        "test_convert_uint_sat_long",
        "test_convert_uint_sat_rte_long",
        "test_convert_uint_sat_rtp_long",
        "test_convert_uint_sat_rtn_long",
        "test_convert_uint_sat_rtz_long",
        "test_implicit_int_uchar",
        "test_convert_int_uchar",
        "test_convert_int_rte_uchar",
        "test_convert_int_rtp_uchar",
        "test_convert_int_rtn_uchar",
        "test_convert_int_rtz_uchar",
        "test_convert_int_sat_uchar",
        "test_convert_int_sat_rte_uchar",
        "test_convert_int_sat_rtp_uchar",
        "test_convert_int_sat_rtn_uchar",
        "test_convert_int_sat_rtz_uchar",
        "test_implicit_int_char",
        "test_convert_int_char",
        "test_convert_int_rte_char",
        "test_convert_int_rtp_char",
        "test_convert_int_rtn_char",
        "test_convert_int_rtz_char",
        "test_convert_int_sat_char",
        "test_convert_int_sat_rte_char",
        "test_convert_int_sat_rtp_char",
        "test_convert_int_sat_rtn_char",
        "test_convert_int_sat_rtz_char",
        "test_implicit_int_ushort",
        "test_convert_int_ushort",
        "test_convert_int_rte_ushort",
        "test_convert_int_rtp_ushort",
        "test_convert_int_rtn_ushort",
        "test_convert_int_rtz_ushort",
        "test_convert_int_sat_ushort",
        "test_convert_int_sat_rte_ushort",
        "test_convert_int_sat_rtp_ushort",
        "test_convert_int_sat_rtn_ushort",
        "test_convert_int_sat_rtz_ushort",
        "test_implicit_int_short",
        "test_convert_int_short",
        "test_convert_int_rte_short",
        "test_convert_int_rtp_short",
        "test_convert_int_rtn_short",
        "test_convert_int_rtz_short",
        "test_convert_int_sat_short",
        "test_convert_int_sat_rte_short",
        "test_convert_int_sat_rtp_short",
        "test_convert_int_sat_rtn_short",
        "test_convert_int_sat_rtz_short",
        "test_implicit_int_uint",
        "test_convert_int_uint",
        "test_convert_int_rte_uint",
        "test_convert_int_rtp_uint",
        "test_convert_int_rtn_uint",
        "test_convert_int_rtz_uint",
        "test_convert_int_sat_uint",
        "test_convert_int_sat_rte_uint",
        "test_convert_int_sat_rtp_uint",
        "test_convert_int_sat_rtn_uint",
        "test_convert_int_sat_rtz_uint",
        "test_implicit_int_int",
        "test_convert_int_int",
        "test_convert_int_rte_int",
        "test_convert_int_rtp_int",
        "test_convert_int_rtn_int",
        "test_convert_int_rtz_int",
        "test_convert_int_sat_int",
        "test_convert_int_sat_rte_int",
        "test_convert_int_sat_rtp_int",
        "test_convert_int_sat_rtn_int",
        "test_convert_int_sat_rtz_int",
        "test_implicit_int_float",
        "test_convert_int_float",
        "test_convert_int_rte_float",
        "test_convert_int_rtp_float",
        "test_convert_int_rtn_float",
        "test_convert_int_rtz_float",
        "test_convert_int_sat_float",
        "test_convert_int_sat_rte_float",
        "test_convert_int_sat_rtp_float",
        "test_convert_int_sat_rtn_float",
        "test_convert_int_sat_rtz_float",
        "test_implicit_int_ulong",
        "test_convert_int_ulong",
        "test_convert_int_rte_ulong",
        "test_convert_int_rtp_ulong",
        "test_convert_int_rtn_ulong",
        "test_convert_int_rtz_ulong",
        "test_convert_int_sat_ulong",
        "test_convert_int_sat_rte_ulong",
        "test_convert_int_sat_rtp_ulong",
        "test_convert_int_sat_rtn_ulong",
        "test_convert_int_sat_rtz_ulong",
        "test_implicit_int_long",
        "test_convert_int_long",
        "test_convert_int_rte_long",
        "test_convert_int_rtp_long",
        "test_convert_int_rtn_long",
        "test_convert_int_rtz_long",
        "test_convert_int_sat_long",
        "test_convert_int_sat_rte_long",
        "test_convert_int_sat_rtp_long",
        "test_convert_int_sat_rtn_long",
        "test_convert_int_sat_rtz_long",
        "test_implicit_float_uchar",
        "test_convert_float_uchar",
        "test_convert_float_rte_uchar",
        "test_convert_float_rtp_uchar",
        "test_convert_float_rtn_uchar",
        "test_convert_float_rtz_uchar",
        "test_implicit_float_char",
        "test_convert_float_char",
        "test_convert_float_rte_char",
        "test_convert_float_rtp_char",
        "test_convert_float_rtn_char",
        "test_convert_float_rtz_char",
        "test_implicit_float_ushort",
        "test_convert_float_ushort",
        "test_convert_float_rte_ushort",
        "test_convert_float_rtp_ushort",
        "test_convert_float_rtn_ushort",
        "test_convert_float_rtz_ushort",
        "test_implicit_float_short",
        "test_convert_float_short",
        "test_convert_float_rte_short",
        "test_convert_float_rtp_short",
        "test_convert_float_rtn_short",
        "test_convert_float_rtz_short",
        "test_implicit_float_uint",
        "test_convert_float_uint",
        "test_convert_float_rte_uint",
        "test_convert_float_rtp_uint",
        "test_convert_float_rtn_uint",
        "test_convert_float_rtz_uint",
        "test_implicit_float_int",
        "test_convert_float_int",
        "test_convert_float_rte_int",
        "test_convert_float_rtp_int",
        "test_convert_float_rtn_int",
        "test_convert_float_rtz_int",
        "test_implicit_float_float",
        "test_convert_float_float",
        "test_convert_float_rte_float",
        "test_convert_float_rtp_float",
        "test_convert_float_rtn_float",
        "test_convert_float_rtz_float",
        "test_implicit_float_ulong",
        "test_convert_float_ulong",
        "test_convert_float_rte_ulong",
        "test_convert_float_rtp_ulong",
        "test_convert_float_rtn_ulong",
        "test_convert_float_rtz_ulong",
        "test_implicit_float_long",
        "test_convert_float_long",
        "test_convert_float_rte_long",
        "test_convert_float_rtp_long",
        "test_convert_float_rtn_long",
        "test_convert_float_rtz_long",
        "test_implicit_ulong_uchar",
        "test_convert_ulong_uchar",
        "test_convert_ulong_rte_uchar",
        "test_convert_ulong_rtp_uchar",
        "test_convert_ulong_rtn_uchar",
        "test_convert_ulong_rtz_uchar",
        "test_convert_ulong_sat_uchar",
        "test_convert_ulong_sat_rte_uchar",
        "test_convert_ulong_sat_rtp_uchar",
        "test_convert_ulong_sat_rtn_uchar",
        "test_convert_ulong_sat_rtz_uchar",
        "test_implicit_ulong_char",
        "test_convert_ulong_char",
        "test_convert_ulong_rte_char",
        "test_convert_ulong_rtp_char",
        "test_convert_ulong_rtn_char",
        "test_convert_ulong_rtz_char",
        "test_convert_ulong_sat_char",
        "test_convert_ulong_sat_rte_char",
        "test_convert_ulong_sat_rtp_char",
        "test_convert_ulong_sat_rtn_char",
        "test_convert_ulong_sat_rtz_char",
        "test_implicit_ulong_ushort",
        "test_convert_ulong_ushort",
        "test_convert_ulong_rte_ushort",
        "test_convert_ulong_rtp_ushort",
        "test_convert_ulong_rtn_ushort",
        "test_convert_ulong_rtz_ushort",
        "test_convert_ulong_sat_ushort",
        "test_convert_ulong_sat_rte_ushort",
        "test_convert_ulong_sat_rtp_ushort",
        "test_convert_ulong_sat_rtn_ushort",
        "test_convert_ulong_sat_rtz_ushort",
        "test_implicit_ulong_short",
        "test_convert_ulong_short",
        "test_convert_ulong_rte_short",
        "test_convert_ulong_rtp_short",
        "test_convert_ulong_rtn_short",
        "test_convert_ulong_rtz_short",
        "test_convert_ulong_sat_short",
        "test_convert_ulong_sat_rte_short",
        "test_convert_ulong_sat_rtp_short",
        "test_convert_ulong_sat_rtn_short",
        "test_convert_ulong_sat_rtz_short",
        "test_implicit_ulong_uint",
        "test_convert_ulong_uint",
        "test_convert_ulong_rte_uint",
        "test_convert_ulong_rtp_uint",
        "test_convert_ulong_rtn_uint",
        "test_convert_ulong_rtz_uint",
        "test_convert_ulong_sat_uint",
        "test_convert_ulong_sat_rte_uint",
        "test_convert_ulong_sat_rtp_uint",
        "test_convert_ulong_sat_rtn_uint",
        "test_convert_ulong_sat_rtz_uint",
        "test_implicit_ulong_int",
        "test_convert_ulong_int",
        "test_convert_ulong_rte_int",
        "test_convert_ulong_rtp_int",
        "test_convert_ulong_rtn_int",
        "test_convert_ulong_rtz_int",
        "test_convert_ulong_sat_int",
        "test_convert_ulong_sat_rte_int",
        "test_convert_ulong_sat_rtp_int",
        "test_convert_ulong_sat_rtn_int",
        "test_convert_ulong_sat_rtz_int",
        "test_implicit_ulong_float",
        "test_convert_ulong_float",
        "test_convert_ulong_rte_float",
        "test_convert_ulong_rtp_float",
        "test_convert_ulong_rtn_float",
        "test_convert_ulong_rtz_float",
        "test_convert_ulong_sat_float",
        "test_convert_ulong_sat_rte_float",
        "test_convert_ulong_sat_rtp_float",
        "test_convert_ulong_sat_rtn_float",
        "test_convert_ulong_sat_rtz_float",
        "test_implicit_ulong_ulong",
        "test_convert_ulong_ulong",
        "test_convert_ulong_rte_ulong",
        "test_convert_ulong_rtp_ulong",
        "test_convert_ulong_rtn_ulong",
        "test_convert_ulong_rtz_ulong",
        "test_convert_ulong_sat_ulong",
        "test_convert_ulong_sat_rte_ulong",
        "test_convert_ulong_sat_rtp_ulong",
        "test_convert_ulong_sat_rtn_ulong",
        "test_convert_ulong_sat_rtz_ulong",
        "test_implicit_ulong_long",
        "test_convert_ulong_long",
        "test_convert_ulong_rte_long",
        "test_convert_ulong_rtp_long",
        "test_convert_ulong_rtn_long",
        "test_convert_ulong_rtz_long",
        "test_convert_ulong_sat_long",
        "test_convert_ulong_sat_rte_long",
        "test_convert_ulong_sat_rtp_long",
        "test_convert_ulong_sat_rtn_long",
        "test_convert_ulong_sat_rtz_long",
        "test_implicit_long_uchar",
        "test_convert_long_uchar",
        "test_convert_long_rte_uchar",
        "test_convert_long_rtp_uchar",
        "test_convert_long_rtn_uchar",
        "test_convert_long_rtz_uchar",
        "test_convert_long_sat_uchar",
        "test_convert_long_sat_rte_uchar",
        "test_convert_long_sat_rtp_uchar",
        "test_convert_long_sat_rtn_uchar",
        "test_convert_long_sat_rtz_uchar",
        "test_implicit_long_char",
        "test_convert_long_char",
        "test_convert_long_rte_char",
        "test_convert_long_rtp_char",
        "test_convert_long_rtn_char",
        "test_convert_long_rtz_char",
        "test_convert_long_sat_char",
        "test_convert_long_sat_rte_char",
        "test_convert_long_sat_rtp_char",
        "test_convert_long_sat_rtn_char",
        "test_convert_long_sat_rtz_char",
        "test_implicit_long_ushort",
        "test_convert_long_ushort",
        "test_convert_long_rte_ushort",
        "test_convert_long_rtp_ushort",
        "test_convert_long_rtn_ushort",
        "test_convert_long_rtz_ushort",
        "test_convert_long_sat_ushort",
        "test_convert_long_sat_rte_ushort",
        "test_convert_long_sat_rtp_ushort",
        "test_convert_long_sat_rtn_ushort",
        "test_convert_long_sat_rtz_ushort",
        "test_implicit_long_short",
        "test_convert_long_short",
        "test_convert_long_rte_short",
        "test_convert_long_rtp_short",
        "test_convert_long_rtn_short",
        "test_convert_long_rtz_short",
        "test_convert_long_sat_short",
        "test_convert_long_sat_rte_short",
        "test_convert_long_sat_rtp_short",
        "test_convert_long_sat_rtn_short",
        "test_convert_long_sat_rtz_short",
        "test_implicit_long_uint",
        "test_convert_long_uint",
        "test_convert_long_rte_uint",
        "test_convert_long_rtp_uint",
        "test_convert_long_rtn_uint",
        "test_convert_long_rtz_uint",
        "test_convert_long_sat_uint",
        "test_convert_long_sat_rte_uint",
        "test_convert_long_sat_rtp_uint",
        "test_convert_long_sat_rtn_uint",
        "test_convert_long_sat_rtz_uint",
        "test_implicit_long_int",
        "test_convert_long_int",
        "test_convert_long_rte_int",
        "test_convert_long_rtp_int",
        "test_convert_long_rtn_int",
        "test_convert_long_rtz_int",
        "test_convert_long_sat_int",
        "test_convert_long_sat_rte_int",
        "test_convert_long_sat_rtp_int",
        "test_convert_long_sat_rtn_int",
        "test_convert_long_sat_rtz_int",
        "test_implicit_long_float",
        "test_convert_long_float",
        "test_convert_long_rte_float",
        "test_convert_long_rtp_float",
        "test_convert_long_rtn_float",
        "test_convert_long_rtz_float",
        "test_convert_long_sat_float",
        "test_convert_long_sat_rte_float",
        "test_convert_long_sat_rtp_float",
        "test_convert_long_sat_rtn_float",
        "test_convert_long_sat_rtz_float",
        "test_implicit_long_ulong",
        "test_convert_long_ulong",
        "test_convert_long_rte_ulong",
        "test_convert_long_rtp_ulong",
        "test_convert_long_rtn_ulong",
        "test_convert_long_rtz_ulong",
        "test_convert_long_sat_ulong",
        "test_convert_long_sat_rte_ulong",
        "test_convert_long_sat_rtp_ulong",
        "test_convert_long_sat_rtn_ulong",
        "test_convert_long_sat_rtz_ulong",
        "test_implicit_long_long",
        "test_convert_long_long",
        "test_convert_long_rte_long",
        "test_convert_long_rtp_long",
        "test_convert_long_rtn_long",
        "test_convert_long_rtz_long",
        "test_convert_long_sat_long",
        "test_convert_long_sat_rte_long",
        "test_convert_long_sat_rtp_long",
        "test_convert_long_sat_rtn_long",
        "test_convert_long_sat_rtz_long",
        "long_convert2_type_roundingmode_type_f",
        "long_convert3_type_roundingmode_type_f",
        "long_convert4_type_roundingmode_type_f",
        "long_convert8_type_roundingmode_type_f",
        "long_convert16_type_roundingmode_type_f",
    };

    log_info("test_conversions\n");
    return test_suite(device, size_t_width, folder, test_name, sizeof(test_name) / sizeof(const char *), "");
}


bool test_conversions_double (cl_device_id device, cl_uint size_t_width, const char *folder)
{
    static const char* test_name[] = {
        "convert2_type_roundingmode_type_d",
        "convert3_type_roundingmode_type_d",
        "convert4_type_roundingmode_type_d",
        "convert8_type_roundingmode_type_d",
        "convert16_type_roundingmode_type_d",
        "test_implicit_uchar_double",
        "test_convert_uchar_double",
        "test_convert_uchar_rte_double",
        "test_convert_uchar_rtp_double",
        "test_convert_uchar_rtn_double",
        "test_convert_uchar_rtz_double",
        "test_convert_uchar_sat_double",
        "test_convert_uchar_sat_rte_double",
        "test_convert_uchar_sat_rtp_double",
        "test_convert_uchar_sat_rtn_double",
        "test_convert_uchar_sat_rtz_double",
        "test_implicit_char_double",
        "test_convert_char_double",
        "test_convert_char_rte_double",
        "test_convert_char_rtp_double",
        "test_convert_char_rtn_double",
        "test_convert_char_rtz_double",
        "test_convert_char_sat_double",
        "test_convert_char_sat_rte_double",
        "test_convert_char_sat_rtp_double",
        "test_convert_char_sat_rtn_double",
        "test_convert_char_sat_rtz_double",
        "test_implicit_ushort_double",
        "test_convert_ushort_double",
        "test_convert_ushort_rte_double",
        "test_convert_ushort_rtp_double",
        "test_convert_ushort_rtn_double",
        "test_convert_ushort_rtz_double",
        "test_convert_ushort_sat_double",
        "test_convert_ushort_sat_rte_double",
        "test_convert_ushort_sat_rtp_double",
        "test_convert_ushort_sat_rtn_double",
        "test_convert_ushort_sat_rtz_double",
        "test_implicit_short_double",
        "test_convert_short_double",
        "test_convert_short_rte_double",
        "test_convert_short_rtp_double",
        "test_convert_short_rtn_double",
        "test_convert_short_rtz_double",
        "test_convert_short_sat_double",
        "test_convert_short_sat_rte_double",
        "test_convert_short_sat_rtp_double",
        "test_convert_short_sat_rtn_double",
        "test_convert_short_sat_rtz_double",
        "test_implicit_uint_double",
        "test_convert_uint_double",
        "test_convert_uint_rte_double",
        "test_convert_uint_rtp_double",
        "test_convert_uint_rtn_double",
        "test_convert_uint_rtz_double",
        "test_convert_uint_sat_double",
        "test_convert_uint_sat_rte_double",
        "test_convert_uint_sat_rtp_double",
        "test_convert_uint_sat_rtn_double",
        "test_convert_uint_sat_rtz_double",
        "test_implicit_int_double",
        "test_convert_int_double",
        "test_convert_int_rte_double",
        "test_convert_int_rtp_double",
        "test_convert_int_rtn_double",
        "test_convert_int_rtz_double",
        "test_convert_int_sat_double",
        "test_convert_int_sat_rte_double",
        "test_convert_int_sat_rtp_double",
        "test_convert_int_sat_rtn_double",
        "test_convert_int_sat_rtz_double",
        "test_implicit_float_double",
        "test_convert_float_double",
        "test_convert_float_rte_double",
        "test_convert_float_rtp_double",
        "test_convert_float_rtn_double",
        "test_convert_float_rtz_double",
        "test_implicit_double_uchar",
        "test_convert_double_uchar",
        "test_convert_double_rte_uchar",
        "test_convert_double_rtp_uchar",
        "test_convert_double_rtn_uchar",
        "test_convert_double_rtz_uchar",
        "test_implicit_double_char",
        "test_convert_double_char",
        "test_convert_double_rte_char",
        "test_convert_double_rtp_char",
        "test_convert_double_rtn_char",
        "test_convert_double_rtz_char",
        "test_implicit_double_ushort",
        "test_convert_double_ushort",
        "test_convert_double_rte_ushort",
        "test_convert_double_rtp_ushort",
        "test_convert_double_rtn_ushort",
        "test_convert_double_rtz_ushort",
        "test_implicit_double_short",
        "test_convert_double_short",
        "test_convert_double_rte_short",
        "test_convert_double_rtp_short",
        "test_convert_double_rtn_short",
        "test_convert_double_rtz_short",
        "test_implicit_double_uint",
        "test_convert_double_uint",
        "test_convert_double_rte_uint",
        "test_convert_double_rtp_uint",
        "test_convert_double_rtn_uint",
        "test_convert_double_rtz_uint",
        "test_implicit_double_int",
        "test_convert_double_int",
        "test_convert_double_rte_int",
        "test_convert_double_rtp_int",
        "test_convert_double_rtn_int",
        "test_convert_double_rtz_int",
        "test_implicit_double_float",
        "test_convert_double_float",
        "test_convert_double_rte_float",
        "test_convert_double_rtp_float",
        "test_convert_double_rtn_float",
        "test_convert_double_rtz_float",
        "test_implicit_double_double",
        "test_convert_double_double",
        "test_convert_double_rte_double",
        "test_convert_double_rtp_double",
        "test_convert_double_rtn_double",
        "test_convert_double_rtz_double",
        "test_implicit_double_ulong",
        "test_convert_double_ulong",
        "test_convert_double_rte_ulong",
        "test_convert_double_rtp_ulong",
        "test_convert_double_rtn_ulong",
        "test_convert_double_rtz_ulong",
        "test_implicit_double_long",
        "test_convert_double_long",
        "test_convert_double_rte_long",
        "test_convert_double_rtp_long",
        "test_convert_double_rtn_long",
        "test_convert_double_rtz_long",
        "test_implicit_ulong_double",
        "test_convert_ulong_double",
        "test_convert_ulong_rte_double",
        "test_convert_ulong_rtp_double",
        "test_convert_ulong_rtn_double",
        "test_convert_ulong_rtz_double",
        "test_convert_ulong_sat_double",
        "test_convert_ulong_sat_rte_double",
        "test_convert_ulong_sat_rtp_double",
        "test_convert_ulong_sat_rtn_double",
        "test_convert_ulong_sat_rtz_double",
        "test_implicit_long_double",
        "test_convert_long_double",
        "test_convert_long_rte_double",
        "test_convert_long_rtp_double",
        "test_convert_long_rtn_double",
        "test_convert_long_rtz_double",
        "test_convert_long_sat_double",
        "test_convert_long_sat_rte_double",
        "test_convert_long_sat_rtp_double",
        "test_convert_long_sat_rtn_double",
        "test_convert_long_sat_rtz_double",
    };

    log_info("test_conversions_double\n");
    return test_suite(device, size_t_width, folder, test_name, sizeof(test_name) / sizeof(const char *), "cl_khr_fp64");
}


bool test_geometrics (cl_device_id device, cl_uint size_t_width, const char *folder)
{
    static const char* test_name[] = {
        "sample_test.geom_cross_float3",
        "sample_test.geom_cross_float4",
        "sample_test.geom_dot_float",
        "sample_test.geom_dot_float2",
        "sample_test.geom_dot_float3",
        "sample_test.geom_dot_float4",
        "sample_test.geom_distance_float",
        "sample_test.geom_distance_float2",
        "sample_test.geom_distance_float3",
        "sample_test.geom_distance_float4",
        "sample_test.geom_fast_distance_float",
        "sample_test.geom_fast_distance_float2",
        "sample_test.geom_fast_distance_float3",
        "sample_test.geom_fast_distance_float4",
        "sample_test.geom_length_float",
        "sample_test.geom_length_float2",
        "sample_test.geom_length_float3",
        "sample_test.geom_length_float4",
        "sample_test.geom_fast_length_float",
        "sample_test.geom_fast_length_float2",
        "sample_test.geom_fast_length_float3",
        "sample_test.geom_fast_length_float4",
        "sample_test.geom_normalize_float",
        "sample_test.geom_normalize_float2",
        "sample_test.geom_normalize_float3",
        "sample_test.geom_normalize_float4",
        "sample_test.geom_fast_normalize_float",
        "sample_test.geom_fast_normalize_float2",
        "sample_test.geom_fast_normalize_float3",
        "sample_test.geom_fast_normalize_float4",
    };

    log_info("test_geometrics\n");
    return test_suite(device, size_t_width, folder, test_name, sizeof(test_name) / sizeof(const char *), "");
}


bool test_geometrics_double (cl_device_id device, cl_uint size_t_width, const char *folder)
{
    static const char* test_name[] = {
        "sample_test.geom_cross_double3",
        "sample_test.geom_cross_double4",
        "sample_test.geom_dot_double",
        "sample_test.geom_dot_double2",
        "sample_test.geom_dot_double3",
        "sample_test.geom_dot_double4",
        "sample_test.geom_distance_double",
        "sample_test.geom_distance_double2",
        "sample_test.geom_distance_double3",
        "sample_test.geom_distance_double4",
        "sample_test.geom_length_double",
        "sample_test.geom_length_double2",
        "sample_test.geom_length_double3",
        "sample_test.geom_length_double4",
        "sample_test.geom_normalize_double",
        "sample_test.geom_normalize_double2",
        "sample_test.geom_normalize_double3",
        "sample_test.geom_normalize_double4",
    };

    log_info("test_geometrics_double\n");
    return test_suite(device, size_t_width, folder, test_name, sizeof(test_name) / sizeof(const char *), "cl_khr_fp64");
}


bool test_half (cl_device_id device, cl_uint size_t_width, const char *folder)
{
    static const char* test_name[] = {
        "test.vload_half_global",
        "test.vload_half_private",
        "test.vload_half_local",
        "test.vload_half_constant",
        "test.vload_half2_global",
        "test.vload_half2_private",
        "test.vload_half2_local",
        "test.vload_half2_constant",
        "test.vload_half4_global",
        "test.vload_half4_private",
        "test.vload_half4_local",
        "test.vload_half4_constant",
        "test.vload_half8_global",
        "test.vload_half8_private",
        "test.vload_half8_local",
        "test.vload_half8_constant",
        "test.vload_half16_global",
        "test.vload_half16_private",
        "test.vload_half16_local",
        "test.vload_half16_constant",
        "test.vload_half3_global",
        "test.vload_half3_private",
        "test.vload_half3_local",
        "test.vload_half3_constant",
        "test.vloada_half_global",
        "test.vloada_half_private",
        "test.vloada_half_local",
        "test.vloada_half_constant",
        "test.vloada_half2_global",
        "test.vloada_half2_private",
        "test.vloada_half2_local",
        "test.vloada_half2_constant",
        "test.vloada_half4_global",
        "test.vloada_half4_private",
        "test.vloada_half4_local",
        "test.vloada_half4_constant",
        "test.vloada_half8_global",
        "test.vloada_half8_private",
        "test.vloada_half8_local",
        "test.vloada_half8_constant",
        "test.vloada_half16_global",
        "test.vloada_half16_private",
        "test.vloada_half16_local",
        "test.vloada_half16_constant",
        "test.vloada_half3_global",
        "test.vloada_half3_private",
        "test.vloada_half3_local",
        "test.vloada_half3_constant",
        "test.vstore_half_global_float",
        "test.vstore_half_private_float",
        "test.vstore_half_local_float",
        "test.vstore_half_global_float2",
        "test.vstore_half_private_float2",
        "test.vstore_half_local_float2",
        "test.vstore_half_global_float4",
        "test.vstore_half_private_float4",
        "test.vstore_half_local_float4",
        "test.vstore_half_global_float8",
        "test.vstore_half_private_float8",
        "test.vstore_half_local_float8",
        "test.vstore_half_global_float16",
        "test.vstore_half_private_float16",
        "test.vstore_half_local_float16",
        "test.vstore_half_global_float3",
        "test.vstore_half_private_float3",
        "test.vstore_half_local_float3",
        "test.vstorea_half_global_float2",
        "test.vstorea_half_private_float2",
        "test.vstorea_half_local_float2",
        "test.vstorea_half_global_float4",
        "test.vstorea_half_private_float4",
        "test.vstorea_half_local_float4",
        "test.vstorea_half_global_float8",
        "test.vstorea_half_private_float8",
        "test.vstorea_half_local_float8",
        "test.vstorea_half_global_float16",
        "test.vstorea_half_private_float16",
        "test.vstorea_half_local_float16",
        "test.vstorea_half_global_float3",
        "test.vstorea_half_private_float3",
        "test.vstorea_half_local_float3",
        "test.vstore_half_rte_global_float",
        "test.vstore_half_rte_private_float",
        "test.vstore_half_rte_local_float",
        "test.vstore_half_rte_global_float2",
        "test.vstore_half_rte_private_float2",
        "test.vstore_half_rte_local_float2",
        "test.vstore_half_rte_global_float4",
        "test.vstore_half_rte_private_float4",
        "test.vstore_half_rte_local_float4",
        "test.vstore_half_rte_global_float8",
        "test.vstore_half_rte_private_float8",
        "test.vstore_half_rte_local_float8",
        "test.vstore_half_rte_global_float16",
        "test.vstore_half_rte_private_float16",
        "test.vstore_half_rte_local_float16",
        "test.vstore_half_rte_global_float3",
        "test.vstore_half_rte_private_float3",
        "test.vstore_half_rte_local_float3",
        "test.vstorea_half_rte_global_float2",
        "test.vstorea_half_rte_private_float2",
        "test.vstorea_half_rte_local_float2",
        "test.vstorea_half_rte_global_float4",
        "test.vstorea_half_rte_private_float4",
        "test.vstorea_half_rte_local_float4",
        "test.vstorea_half_rte_global_float8",
        "test.vstorea_half_rte_private_float8",
        "test.vstorea_half_rte_local_float8",
        "test.vstorea_half_rte_global_float16",
        "test.vstorea_half_rte_private_float16",
        "test.vstorea_half_rte_local_float16",
        "test.vstorea_half_rte_global_float3",
        "test.vstorea_half_rte_private_float3",
        "test.vstorea_half_rte_local_float3",
        "test.vstore_half_rtz_global_float",
        "test.vstore_half_rtz_private_float",
        "test.vstore_half_rtz_local_float",
        "test.vstore_half_rtz_global_float2",
        "test.vstore_half_rtz_private_float2",
        "test.vstore_half_rtz_local_float2",
        "test.vstore_half_rtz_global_float4",
        "test.vstore_half_rtz_private_float4",
        "test.vstore_half_rtz_local_float4",
        "test.vstore_half_rtz_global_float8",
        "test.vstore_half_rtz_private_float8",
        "test.vstore_half_rtz_local_float8",
        "test.vstore_half_rtz_global_float16",
        "test.vstore_half_rtz_private_float16",
        "test.vstore_half_rtz_local_float16",
        "test.vstore_half_rtz_global_float3",
        "test.vstore_half_rtz_private_float3",
        "test.vstore_half_rtz_local_float3",
        "test.vstorea_half_rtz_global_float2",
        "test.vstorea_half_rtz_private_float2",
        "test.vstorea_half_rtz_local_float2",
        "test.vstorea_half_rtz_global_float4",
        "test.vstorea_half_rtz_private_float4",
        "test.vstorea_half_rtz_local_float4",
        "test.vstorea_half_rtz_global_float8",
        "test.vstorea_half_rtz_private_float8",
        "test.vstorea_half_rtz_local_float8",
        "test.vstorea_half_rtz_global_float16",
        "test.vstorea_half_rtz_private_float16",
        "test.vstorea_half_rtz_local_float16",
        "test.vstorea_half_rtz_global_float3",
        "test.vstorea_half_rtz_private_float3",
        "test.vstorea_half_rtz_local_float3",
        "test.vstore_half_rtp_global_float",
        "test.vstore_half_rtp_private_float",
        "test.vstore_half_rtp_local_float",
        "test.vstore_half_rtp_global_float2",
        "test.vstore_half_rtp_private_float2",
        "test.vstore_half_rtp_local_float2",
        "test.vstore_half_rtp_global_float4",
        "test.vstore_half_rtp_private_float4",
        "test.vstore_half_rtp_local_float4",
        "test.vstore_half_rtp_global_float8",
        "test.vstore_half_rtp_private_float8",
        "test.vstore_half_rtp_local_float8",
        "test.vstore_half_rtp_global_float16",
        "test.vstore_half_rtp_private_float16",
        "test.vstore_half_rtp_local_float16",
        "test.vstore_half_rtp_global_float3",
        "test.vstore_half_rtp_private_float3",
        "test.vstore_half_rtp_local_float3",
        "test.vstorea_half_rtp_global_float2",
        "test.vstorea_half_rtp_private_float2",
        "test.vstorea_half_rtp_local_float2",
        "test.vstorea_half_rtp_global_float4",
        "test.vstorea_half_rtp_private_float4",
        "test.vstorea_half_rtp_local_float4",
        "test.vstorea_half_rtp_global_float8",
        "test.vstorea_half_rtp_private_float8",
        "test.vstorea_half_rtp_local_float8",
        "test.vstorea_half_rtp_global_float16",
        "test.vstorea_half_rtp_private_float16",
        "test.vstorea_half_rtp_local_float16",
        "test.vstorea_half_rtp_global_float3",
        "test.vstorea_half_rtp_private_float3",
        "test.vstorea_half_rtp_local_float3",
        "test.vstore_half_rtn_global_float",
        "test.vstore_half_rtn_private_float",
        "test.vstore_half_rtn_local_float",
        "test.vstore_half_rtn_global_float2",
        "test.vstore_half_rtn_private_float2",
        "test.vstore_half_rtn_local_float2",
        "test.vstore_half_rtn_global_float4",
        "test.vstore_half_rtn_private_float4",
        "test.vstore_half_rtn_local_float4",
        "test.vstore_half_rtn_global_float8",
        "test.vstore_half_rtn_private_float8",
        "test.vstore_half_rtn_local_float8",
        "test.vstore_half_rtn_global_float16",
        "test.vstore_half_rtn_private_float16",
        "test.vstore_half_rtn_local_float16",
        "test.vstore_half_rtn_global_float3",
        "test.vstore_half_rtn_private_float3",
        "test.vstore_half_rtn_local_float3",
        "test.vstorea_half_rtn_global_float2",
        "test.vstorea_half_rtn_private_float2",
        "test.vstorea_half_rtn_local_float2",
        "test.vstorea_half_rtn_global_float4",
        "test.vstorea_half_rtn_private_float4",
        "test.vstorea_half_rtn_local_float4",
        "test.vstorea_half_rtn_global_float8",
        "test.vstorea_half_rtn_private_float8",
        "test.vstorea_half_rtn_local_float8",
        "test.vstorea_half_rtn_global_float16",
        "test.vstorea_half_rtn_private_float16",
        "test.vstorea_half_rtn_local_float16",
        "test.vstorea_half_rtn_global_float3",
        "test.vstorea_half_rtn_private_float3",
        "test.vstorea_half_rtn_local_float3",
    };

    log_info("test_half\n");
    return test_suite(device, size_t_width, folder, test_name, sizeof(test_name) / sizeof(const char *), "");
}


bool test_half_double (cl_device_id device, cl_uint size_t_width, const char *folder)
{
    static const char* test_name[] = {
        "test.vstore_half_global_double",
        "test.vstore_half_private_double",
        "test.vstore_half_local_double",
        "test.vstore_half_global_double2",
        "test.vstore_half_private_double2",
        "test.vstore_half_local_double2",
        "test.vstore_half_global_double4",
        "test.vstore_half_private_double4",
        "test.vstore_half_local_double4",
        "test.vstore_half_global_double8",
        "test.vstore_half_private_double8",
        "test.vstore_half_local_double8",
        "test.vstore_half_global_double16",
        "test.vstore_half_private_double16",
        "test.vstore_half_local_double16",
        "test.vstore_half_global_double3",
        "test.vstore_half_private_double3",
        "test.vstore_half_local_double3",
        "test.vstorea_half_global_double2",
        "test.vstorea_half_private_double2",
        "test.vstorea_half_local_double2",
        "test.vstorea_half_global_double4",
        "test.vstorea_half_private_double4",
        "test.vstorea_half_local_double4",
        "test.vstorea_half_global_double8",
        "test.vstorea_half_private_double8",
        "test.vstorea_half_local_double8",
        "test.vstorea_half_global_double16",
        "test.vstorea_half_private_double16",
        "test.vstorea_half_local_double16",
        "test.vstorea_half_global_double3",
        "test.vstorea_half_private_double3",
        "test.vstorea_half_local_double3",
        "test.vstore_half_rte_global_double",
        "test.vstore_half_rte_private_double",
        "test.vstore_half_rte_local_double",
        "test.vstore_half_rte_global_double2",
        "test.vstore_half_rte_private_double2",
        "test.vstore_half_rte_local_double2",
        "test.vstore_half_rte_global_double4",
        "test.vstore_half_rte_private_double4",
        "test.vstore_half_rte_local_double4",
        "test.vstore_half_rte_global_double8",
        "test.vstore_half_rte_private_double8",
        "test.vstore_half_rte_local_double8",
        "test.vstore_half_rte_global_double16",
        "test.vstore_half_rte_private_double16",
        "test.vstore_half_rte_local_double16",
        "test.vstore_half_rte_global_double3",
        "test.vstore_half_rte_private_double3",
        "test.vstore_half_rte_local_double3",
        "test.vstorea_half_rte_global_double2",
        "test.vstorea_half_rte_private_double2",
        "test.vstorea_half_rte_local_double2",
        "test.vstorea_half_rte_global_double4",
        "test.vstorea_half_rte_private_double4",
        "test.vstorea_half_rte_local_double4",
        "test.vstorea_half_rte_global_double8",
        "test.vstorea_half_rte_private_double8",
        "test.vstorea_half_rte_local_double8",
        "test.vstorea_half_rte_global_double16",
        "test.vstorea_half_rte_private_double16",
        "test.vstorea_half_rte_local_double16",
        "test.vstorea_half_rte_global_double3",
        "test.vstorea_half_rte_private_double3",
        "test.vstorea_half_rte_local_double3",
        "test.vstore_half_rtz_global_double",
        "test.vstore_half_rtz_private_double",
        "test.vstore_half_rtz_local_double",
        "test.vstore_half_rtz_global_double2",
        "test.vstore_half_rtz_private_double2",
        "test.vstore_half_rtz_local_double2",
        "test.vstore_half_rtz_global_double4",
        "test.vstore_half_rtz_private_double4",
        "test.vstore_half_rtz_local_double4",
        "test.vstore_half_rtz_global_double8",
        "test.vstore_half_rtz_private_double8",
        "test.vstore_half_rtz_local_double8",
        "test.vstore_half_rtz_global_double16",
        "test.vstore_half_rtz_private_double16",
        "test.vstore_half_rtz_local_double16",
        "test.vstore_half_rtz_global_double3",
        "test.vstore_half_rtz_private_double3",
        "test.vstore_half_rtz_local_double3",
        "test.vstorea_half_rtz_global_double2",
        "test.vstorea_half_rtz_private_double2",
        "test.vstorea_half_rtz_local_double2",
        "test.vstorea_half_rtz_global_double4",
        "test.vstorea_half_rtz_private_double4",
        "test.vstorea_half_rtz_local_double4",
        "test.vstorea_half_rtz_global_double8",
        "test.vstorea_half_rtz_private_double8",
        "test.vstorea_half_rtz_local_double8",
        "test.vstorea_half_rtz_global_double16",
        "test.vstorea_half_rtz_private_double16",
        "test.vstorea_half_rtz_local_double16",
        "test.vstorea_half_rtz_global_double3",
        "test.vstorea_half_rtz_private_double3",
        "test.vstorea_half_rtz_local_double3",
        "test.vstore_half_rtp_global_double",
        "test.vstore_half_rtp_private_double",
        "test.vstore_half_rtp_local_double",
        "test.vstore_half_rtp_global_double2",
        "test.vstore_half_rtp_private_double2",
        "test.vstore_half_rtp_local_double2",
        "test.vstore_half_rtp_global_double4",
        "test.vstore_half_rtp_private_double4",
        "test.vstore_half_rtp_local_double4",
        "test.vstore_half_rtp_global_double8",
        "test.vstore_half_rtp_private_double8",
        "test.vstore_half_rtp_local_double8",
        "test.vstore_half_rtp_global_double16",
        "test.vstore_half_rtp_private_double16",
        "test.vstore_half_rtp_local_double16",
        "test.vstore_half_rtp_global_double3",
        "test.vstore_half_rtp_private_double3",
        "test.vstore_half_rtp_local_double3",
        "test.vstorea_half_rtp_global_double2",
        "test.vstorea_half_rtp_private_double2",
        "test.vstorea_half_rtp_local_double2",
        "test.vstorea_half_rtp_global_double4",
        "test.vstorea_half_rtp_private_double4",
        "test.vstorea_half_rtp_local_double4",
        "test.vstorea_half_rtp_global_double8",
        "test.vstorea_half_rtp_private_double8",
        "test.vstorea_half_rtp_local_double8",
        "test.vstorea_half_rtp_global_double16",
        "test.vstorea_half_rtp_private_double16",
        "test.vstorea_half_rtp_local_double16",
        "test.vstorea_half_rtp_global_double3",
        "test.vstorea_half_rtp_private_double3",
        "test.vstorea_half_rtp_local_double3",
        "test.vstore_half_rtn_global_double",
        "test.vstore_half_rtn_private_double",
        "test.vstore_half_rtn_local_double",
        "test.vstore_half_rtn_global_double2",
        "test.vstore_half_rtn_private_double2",
        "test.vstore_half_rtn_local_double2",
        "test.vstore_half_rtn_global_double4",
        "test.vstore_half_rtn_private_double4",
        "test.vstore_half_rtn_local_double4",
        "test.vstore_half_rtn_global_double8",
        "test.vstore_half_rtn_private_double8",
        "test.vstore_half_rtn_local_double8",
        "test.vstore_half_rtn_global_double16",
        "test.vstore_half_rtn_private_double16",
        "test.vstore_half_rtn_local_double16",
        "test.vstore_half_rtn_global_double3",
        "test.vstore_half_rtn_private_double3",
        "test.vstore_half_rtn_local_double3",
        "test.vstorea_half_rtn_global_double2",
        "test.vstorea_half_rtn_private_double2",
        "test.vstorea_half_rtn_local_double2",
        "test.vstorea_half_rtn_global_double4",
        "test.vstorea_half_rtn_private_double4",
        "test.vstorea_half_rtn_local_double4",
        "test.vstorea_half_rtn_global_double8",
        "test.vstorea_half_rtn_private_double8",
        "test.vstorea_half_rtn_local_double8",
        "test.vstorea_half_rtn_global_double16",
        "test.vstorea_half_rtn_private_double16",
        "test.vstorea_half_rtn_local_double16",
        "test.vstorea_half_rtn_global_double3",
        "test.vstorea_half_rtn_private_double3",
        "test.vstorea_half_rtn_local_double3",
    };

    log_info("test_half_double\n");
    return test_suite(device, size_t_width, folder, test_name, sizeof(test_name) / sizeof(const char *), "cl_khr_fp64");
}


bool test_kernel_image_methods (cl_device_id device, cl_uint size_t_width, const char *folder)
{
    static const char* test_name[] = {
        "sample_kernel.get_image_info_1D",
        "sample_kernel.get_image_info_2D",
        "sample_kernel.get_image_info_3D",
        "sample_kernel.get_image_info_1D_array",
        "sample_kernel.get_image_info_2D_array",
    };

    log_info("test_kernel_image_methods\n");
    return test_suite(device, size_t_width, folder, test_name, sizeof(test_name) / sizeof(const char *), "");
}


bool test_images_kernel_read_write (cl_device_id device, cl_uint size_t_width, const char *folder)
{
    static const char* test_name[] = {
        "sample_kernel.read_image_set_1D_fint",
        "sample_kernel.read_image_set_1D_ffloat",
        "sample_kernel.read_image_set_1D_iint",
        "sample_kernel.read_image_set_1D_ifloat",
        "sample_kernel.read_image_set_1D_uiint",
        "sample_kernel.read_image_set_1D_uifloat",
        "sample_kernel.write_image_1D_set_float",
        "sample_kernel.write_image_1D_set_int",
        "sample_kernel.write_image_1D_set_uint",
        "sample_kernel.read_image_set_2D_fint",
        "sample_kernel.read_image_set_2D_ffloat",
        "sample_kernel.read_image_set_2D_iint",
        "sample_kernel.read_image_set_2D_ifloat",
        "sample_kernel.read_image_set_2D_uiint",
        "sample_kernel.read_image_set_2D_uifloat",
        "sample_kernel.write_image_2D_set_float",
        "sample_kernel.write_image_2D_set_int",
        "sample_kernel.write_image_2D_set_uint",
        "sample_kernel.read_image_set_3D_fint",
        "sample_kernel.read_image_set_3D_ffloat",
        "sample_kernel.read_image_set_3D_iint",
        "sample_kernel.read_image_set_3D_ifloat",
        "sample_kernel.read_image_set_3D_uiint",
        "sample_kernel.read_image_set_3D_uifloat",
        "sample_kernel.read_image_set_1D_array_fint",
        "sample_kernel.read_image_set_1D_array_ffloat",
        "sample_kernel.read_image_set_1D_array_iint",
        "sample_kernel.read_image_set_1D_array_ifloat",
        "sample_kernel.read_image_set_1D_array_uiint",
        "sample_kernel.read_image_set_1D_array_uifloat",
        "sample_kernel.write_image_1D_array_set_float",
        "sample_kernel.write_image_1D_array_set_int",
        "sample_kernel.write_image_1D_array_set_uint",
        "sample_kernel.read_image_set_2D_array_fint",
        "sample_kernel.read_image_set_2D_array_ffloat",
        "sample_kernel.read_image_set_2D_array_iint",
        "sample_kernel.read_image_set_2D_array_ifloat",
        "sample_kernel.read_image_set_2D_array_uiint",
        "sample_kernel.read_image_set_2D_array_uifloat",
        "sample_kernel.write_image_2D_array_set_float",
        "sample_kernel.write_image_2D_array_set_int",
        "sample_kernel.write_image_2D_array_set_uint",
    };

    log_info("test_images_kernel_read_write\n");
    return test_suite(device, size_t_width, folder, test_name, sizeof(test_name) / sizeof(const char *), "");
}


bool test_images_samplerless_read (cl_device_id device, cl_uint size_t_width, const char *folder)
{
    static const char* test_name[] = {
        "sample_kernel.read_image_set_1D_float",
        "sample_kernel.read_image_set_1D_int",
        "sample_kernel.read_image_set_1D_uint",
        "sample_kernel.read_image_set_1D_buffer_float",
        "sample_kernel.read_image_set_1D_buffer_int",
        "sample_kernel.read_image_set_1D_buffer_uint",
        "sample_kernel.read_image_set_2D_float",
        "sample_kernel.read_image_set_2D_int",
        "sample_kernel.read_image_set_2D_uint",
        "sample_kernel.read_image_set_3D_float",
        "sample_kernel.read_image_set_3D_int",
        "sample_kernel.read_image_set_3D_uint",
        "sample_kernel.read_image_set_1D_array_float",
        "sample_kernel.read_image_set_1D_array_int",
        "sample_kernel.read_image_set_1D_array_uint",
        "sample_kernel.read_image_set_2D_array_float",
        "sample_kernel.read_image_set_2D_array_int",
        "sample_kernel.read_image_set_2D_array_uint",
    };

    log_info("test_images_samplerless_read\n");
    return test_suite(device, size_t_width, folder, test_name, sizeof(test_name) / sizeof(const char *), "");
}


bool test_integer_ops (cl_device_id device, cl_uint size_t_width, const char *folder)
{
    static const char* test_name[] = {
        "sample_test.integer_clz_char",
        "sample_test.integer_clz_char2",
        "sample_test.integer_clz_char3",
        "sample_test.integer_clz_char4",
        "sample_test.integer_clz_char8",
        "sample_test.integer_clz_char16",
        "sample_test.integer_clz_uchar",
        "sample_test.integer_clz_uchar2",
        "sample_test.integer_clz_uchar3",
        "sample_test.integer_clz_uchar4",
        "sample_test.integer_clz_uchar8",
        "sample_test.integer_clz_uchar16",
        "sample_test.integer_clz_short",
        "sample_test.integer_clz_short2",
        "sample_test.integer_clz_short3",
        "sample_test.integer_clz_short4",
        "sample_test.integer_clz_short8",
        "sample_test.integer_clz_short16",
        "sample_test.integer_clz_ushort",
        "sample_test.integer_clz_ushort2",
        "sample_test.integer_clz_ushort3",
        "sample_test.integer_clz_ushort4",
        "sample_test.integer_clz_ushort8",
        "sample_test.integer_clz_ushort16",
        "sample_test.integer_clz_int",
        "sample_test.integer_clz_int2",
        "sample_test.integer_clz_int3",
        "sample_test.integer_clz_int4",
        "sample_test.integer_clz_int8",
        "sample_test.integer_clz_int16",
        "sample_test.integer_clz_uint",
        "sample_test.integer_clz_uint2",
        "sample_test.integer_clz_uint3",
        "sample_test.integer_clz_uint4",
        "sample_test.integer_clz_uint8",
        "sample_test.integer_clz_uint16",
        "sample_test.integer_clz_long",
        "sample_test.integer_clz_long2",
        "sample_test.integer_clz_long3",
        "sample_test.integer_clz_long4",
        "sample_test.integer_clz_long8",
        "sample_test.integer_clz_long16",
        "sample_test.integer_clz_ulong",
        "sample_test.integer_clz_ulong2",
        "sample_test.integer_clz_ulong3",
        "sample_test.integer_clz_ulong4",
        "sample_test.integer_clz_ulong8",
        "sample_test.integer_clz_ulong16",
        "sample_test.integer_hadd_char",
        "sample_test.integer_hadd_char2",
        "sample_test.integer_hadd_char3",
        "sample_test.integer_hadd_char4",
        "sample_test.integer_hadd_char8",
        "sample_test.integer_hadd_char16",
        "sample_test.integer_hadd_uchar",
        "sample_test.integer_hadd_uchar2",
        "sample_test.integer_hadd_uchar3",
        "sample_test.integer_hadd_uchar4",
        "sample_test.integer_hadd_uchar8",
        "sample_test.integer_hadd_uchar16",
        "sample_test.integer_hadd_short",
        "sample_test.integer_hadd_short2",
        "sample_test.integer_hadd_short3",
        "sample_test.integer_hadd_short4",
        "sample_test.integer_hadd_short8",
        "sample_test.integer_hadd_short16",
        "sample_test.integer_hadd_ushort",
        "sample_test.integer_hadd_ushort2",
        "sample_test.integer_hadd_ushort3",
        "sample_test.integer_hadd_ushort4",
        "sample_test.integer_hadd_ushort8",
        "sample_test.integer_hadd_ushort16",
        "sample_test.integer_hadd_int",
        "sample_test.integer_hadd_int2",
        "sample_test.integer_hadd_int3",
        "sample_test.integer_hadd_int4",
        "sample_test.integer_hadd_int8",
        "sample_test.integer_hadd_int16",
        "sample_test.integer_hadd_uint",
        "sample_test.integer_hadd_uint2",
        "sample_test.integer_hadd_uint3",
        "sample_test.integer_hadd_uint4",
        "sample_test.integer_hadd_uint8",
        "sample_test.integer_hadd_uint16",
        "sample_test.integer_hadd_long",
        "sample_test.integer_hadd_long2",
        "sample_test.integer_hadd_long3",
        "sample_test.integer_hadd_long4",
        "sample_test.integer_hadd_long8",
        "sample_test.integer_hadd_long16",
        "sample_test.integer_hadd_ulong",
        "sample_test.integer_hadd_ulong2",
        "sample_test.integer_hadd_ulong3",
        "sample_test.integer_hadd_ulong4",
        "sample_test.integer_hadd_ulong8",
        "sample_test.integer_hadd_ulong16",
        "sample_test.integer_rhadd_char",
        "sample_test.integer_rhadd_char2",
        "sample_test.integer_rhadd_char3",
        "sample_test.integer_rhadd_char4",
        "sample_test.integer_rhadd_char8",
        "sample_test.integer_rhadd_char16",
        "sample_test.integer_rhadd_uchar",
        "sample_test.integer_rhadd_uchar2",
        "sample_test.integer_rhadd_uchar3",
        "sample_test.integer_rhadd_uchar4",
        "sample_test.integer_rhadd_uchar8",
        "sample_test.integer_rhadd_uchar16",
        "sample_test.integer_rhadd_short",
        "sample_test.integer_rhadd_short2",
        "sample_test.integer_rhadd_short3",
        "sample_test.integer_rhadd_short4",
        "sample_test.integer_rhadd_short8",
        "sample_test.integer_rhadd_short16",
        "sample_test.integer_rhadd_ushort",
        "sample_test.integer_rhadd_ushort2",
        "sample_test.integer_rhadd_ushort3",
        "sample_test.integer_rhadd_ushort4",
        "sample_test.integer_rhadd_ushort8",
        "sample_test.integer_rhadd_ushort16",
        "sample_test.integer_rhadd_int",
        "sample_test.integer_rhadd_int2",
        "sample_test.integer_rhadd_int3",
        "sample_test.integer_rhadd_int4",
        "sample_test.integer_rhadd_int8",
        "sample_test.integer_rhadd_int16",
        "sample_test.integer_rhadd_uint",
        "sample_test.integer_rhadd_uint2",
        "sample_test.integer_rhadd_uint3",
        "sample_test.integer_rhadd_uint4",
        "sample_test.integer_rhadd_uint8",
        "sample_test.integer_rhadd_uint16",
        "sample_test.integer_rhadd_long",
        "sample_test.integer_rhadd_long2",
        "sample_test.integer_rhadd_long3",
        "sample_test.integer_rhadd_long4",
        "sample_test.integer_rhadd_long8",
        "sample_test.integer_rhadd_long16",
        "sample_test.integer_rhadd_ulong",
        "sample_test.integer_rhadd_ulong2",
        "sample_test.integer_rhadd_ulong3",
        "sample_test.integer_rhadd_ulong4",
        "sample_test.integer_rhadd_ulong8",
        "sample_test.integer_rhadd_ulong16",
        "sample_test.integer_mul_hi_char",
        "sample_test.integer_mul_hi_char2",
        "sample_test.integer_mul_hi_char3",
        "sample_test.integer_mul_hi_char4",
        "sample_test.integer_mul_hi_char8",
        "sample_test.integer_mul_hi_char16",
        "sample_test.integer_mul_hi_uchar",
        "sample_test.integer_mul_hi_uchar2",
        "sample_test.integer_mul_hi_uchar3",
        "sample_test.integer_mul_hi_uchar4",
        "sample_test.integer_mul_hi_uchar8",
        "sample_test.integer_mul_hi_uchar16",
        "sample_test.integer_mul_hi_short",
        "sample_test.integer_mul_hi_short2",
        "sample_test.integer_mul_hi_short3",
        "sample_test.integer_mul_hi_short4",
        "sample_test.integer_mul_hi_short8",
        "sample_test.integer_mul_hi_short16",
        "sample_test.integer_mul_hi_ushort",
        "sample_test.integer_mul_hi_ushort2",
        "sample_test.integer_mul_hi_ushort3",
        "sample_test.integer_mul_hi_ushort4",
        "sample_test.integer_mul_hi_ushort8",
        "sample_test.integer_mul_hi_ushort16",
        "sample_test.integer_mul_hi_int",
        "sample_test.integer_mul_hi_int2",
        "sample_test.integer_mul_hi_int3",
        "sample_test.integer_mul_hi_int4",
        "sample_test.integer_mul_hi_int8",
        "sample_test.integer_mul_hi_int16",
        "sample_test.integer_mul_hi_uint",
        "sample_test.integer_mul_hi_uint2",
        "sample_test.integer_mul_hi_uint3",
        "sample_test.integer_mul_hi_uint4",
        "sample_test.integer_mul_hi_uint8",
        "sample_test.integer_mul_hi_uint16",
        "sample_test.integer_mul_hi_long",
        "sample_test.integer_mul_hi_long2",
        "sample_test.integer_mul_hi_long3",
        "sample_test.integer_mul_hi_long4",
        "sample_test.integer_mul_hi_long8",
        "sample_test.integer_mul_hi_long16",
        "sample_test.integer_mul_hi_ulong",
        "sample_test.integer_mul_hi_ulong2",
        "sample_test.integer_mul_hi_ulong3",
        "sample_test.integer_mul_hi_ulong4",
        "sample_test.integer_mul_hi_ulong8",
        "sample_test.integer_mul_hi_ulong16",
        "sample_test.integer_rotate_char",
        "sample_test.integer_rotate_char2",
        "sample_test.integer_rotate_char3",
        "sample_test.integer_rotate_char4",
        "sample_test.integer_rotate_char8",
        "sample_test.integer_rotate_char16",
        "sample_test.integer_rotate_uchar",
        "sample_test.integer_rotate_uchar2",
        "sample_test.integer_rotate_uchar3",
        "sample_test.integer_rotate_uchar4",
        "sample_test.integer_rotate_uchar8",
        "sample_test.integer_rotate_uchar16",
        "sample_test.integer_rotate_short",
        "sample_test.integer_rotate_short2",
        "sample_test.integer_rotate_short3",
        "sample_test.integer_rotate_short4",
        "sample_test.integer_rotate_short8",
        "sample_test.integer_rotate_short16",
        "sample_test.integer_rotate_ushort",
        "sample_test.integer_rotate_ushort2",
        "sample_test.integer_rotate_ushort3",
        "sample_test.integer_rotate_ushort4",
        "sample_test.integer_rotate_ushort8",
        "sample_test.integer_rotate_ushort16",
        "sample_test.integer_rotate_int",
        "sample_test.integer_rotate_int2",
        "sample_test.integer_rotate_int3",
        "sample_test.integer_rotate_int4",
        "sample_test.integer_rotate_int8",
        "sample_test.integer_rotate_int16",
        "sample_test.integer_rotate_uint",
        "sample_test.integer_rotate_uint2",
        "sample_test.integer_rotate_uint3",
        "sample_test.integer_rotate_uint4",
        "sample_test.integer_rotate_uint8",
        "sample_test.integer_rotate_uint16",
        "sample_test.integer_rotate_long",
        "sample_test.integer_rotate_long2",
        "sample_test.integer_rotate_long3",
        "sample_test.integer_rotate_long4",
        "sample_test.integer_rotate_long8",
        "sample_test.integer_rotate_long16",
        "sample_test.integer_rotate_ulong",
        "sample_test.integer_rotate_ulong2",
        "sample_test.integer_rotate_ulong3",
        "sample_test.integer_rotate_ulong4",
        "sample_test.integer_rotate_ulong8",
        "sample_test.integer_rotate_ulong16",
        "sample_test.integer_clamp_char",
        "sample_test.integer_clamp_char2",
        "sample_test.integer_clamp_char3",
        "sample_test.integer_clamp_char4",
        "sample_test.integer_clamp_char8",
        "sample_test.integer_clamp_char16",
        "sample_test.integer_clamp_uchar",
        "sample_test.integer_clamp_uchar2",
        "sample_test.integer_clamp_uchar3",
        "sample_test.integer_clamp_uchar4",
        "sample_test.integer_clamp_uchar8",
        "sample_test.integer_clamp_uchar16",
        "sample_test.integer_clamp_short",
        "sample_test.integer_clamp_short2",
        "sample_test.integer_clamp_short3",
        "sample_test.integer_clamp_short4",
        "sample_test.integer_clamp_short8",
        "sample_test.integer_clamp_short16",
        "sample_test.integer_clamp_ushort",
        "sample_test.integer_clamp_ushort2",
        "sample_test.integer_clamp_ushort3",
        "sample_test.integer_clamp_ushort4",
        "sample_test.integer_clamp_ushort8",
        "sample_test.integer_clamp_ushort16",
        "sample_test.integer_clamp_int",
        "sample_test.integer_clamp_int2",
        "sample_test.integer_clamp_int3",
        "sample_test.integer_clamp_int4",
        "sample_test.integer_clamp_int8",
        "sample_test.integer_clamp_int16",
        "sample_test.integer_clamp_uint",
        "sample_test.integer_clamp_uint2",
        "sample_test.integer_clamp_uint3",
        "sample_test.integer_clamp_uint4",
        "sample_test.integer_clamp_uint8",
        "sample_test.integer_clamp_uint16",
        "sample_test.integer_clamp_long",
        "sample_test.integer_clamp_long2",
        "sample_test.integer_clamp_long3",
        "sample_test.integer_clamp_long4",
        "sample_test.integer_clamp_long8",
        "sample_test.integer_clamp_long16",
        "sample_test.integer_clamp_ulong",
        "sample_test.integer_clamp_ulong2",
        "sample_test.integer_clamp_ulong3",
        "sample_test.integer_clamp_ulong4",
        "sample_test.integer_clamp_ulong8",
        "sample_test.integer_clamp_ulong16",
        "sample_test.integer_mad_sat_char",
        "sample_test.integer_mad_sat_char2",
        "sample_test.integer_mad_sat_char3",
        "sample_test.integer_mad_sat_char4",
        "sample_test.integer_mad_sat_char8",
        "sample_test.integer_mad_sat_char16",
        "sample_test.integer_mad_sat_uchar",
        "sample_test.integer_mad_sat_uchar2",
        "sample_test.integer_mad_sat_uchar3",
        "sample_test.integer_mad_sat_uchar4",
        "sample_test.integer_mad_sat_uchar8",
        "sample_test.integer_mad_sat_uchar16",
        "sample_test.integer_mad_sat_short",
        "sample_test.integer_mad_sat_short2",
        "sample_test.integer_mad_sat_short3",
        "sample_test.integer_mad_sat_short4",
        "sample_test.integer_mad_sat_short8",
        "sample_test.integer_mad_sat_short16",
        "sample_test.integer_mad_sat_ushort",
        "sample_test.integer_mad_sat_ushort2",
        "sample_test.integer_mad_sat_ushort3",
        "sample_test.integer_mad_sat_ushort4",
        "sample_test.integer_mad_sat_ushort8",
        "sample_test.integer_mad_sat_ushort16",
        "sample_test.integer_mad_sat_int",
        "sample_test.integer_mad_sat_int2",
        "sample_test.integer_mad_sat_int3",
        "sample_test.integer_mad_sat_int4",
        "sample_test.integer_mad_sat_int8",
        "sample_test.integer_mad_sat_int16",
        "sample_test.integer_mad_sat_uint",
        "sample_test.integer_mad_sat_uint2",
        "sample_test.integer_mad_sat_uint3",
        "sample_test.integer_mad_sat_uint4",
        "sample_test.integer_mad_sat_uint8",
        "sample_test.integer_mad_sat_uint16",
        "sample_test.integer_mad_sat_long",
        "sample_test.integer_mad_sat_long2",
        "sample_test.integer_mad_sat_long3",
        "sample_test.integer_mad_sat_long4",
        "sample_test.integer_mad_sat_long8",
        "sample_test.integer_mad_sat_long16",
        "sample_test.integer_mad_sat_ulong",
        "sample_test.integer_mad_sat_ulong2",
        "sample_test.integer_mad_sat_ulong3",
        "sample_test.integer_mad_sat_ulong4",
        "sample_test.integer_mad_sat_ulong8",
        "sample_test.integer_mad_sat_ulong16",
        "sample_test.integer_mad_hi_char",
        "sample_test.integer_mad_hi_char2",
        "sample_test.integer_mad_hi_char3",
        "sample_test.integer_mad_hi_char4",
        "sample_test.integer_mad_hi_char8",
        "sample_test.integer_mad_hi_char16",
        "sample_test.integer_mad_hi_uchar",
        "sample_test.integer_mad_hi_uchar2",
        "sample_test.integer_mad_hi_uchar3",
        "sample_test.integer_mad_hi_uchar4",
        "sample_test.integer_mad_hi_uchar8",
        "sample_test.integer_mad_hi_uchar16",
        "sample_test.integer_mad_hi_short",
        "sample_test.integer_mad_hi_short2",
        "sample_test.integer_mad_hi_short3",
        "sample_test.integer_mad_hi_short4",
        "sample_test.integer_mad_hi_short8",
        "sample_test.integer_mad_hi_short16",
        "sample_test.integer_mad_hi_ushort",
        "sample_test.integer_mad_hi_ushort2",
        "sample_test.integer_mad_hi_ushort3",
        "sample_test.integer_mad_hi_ushort4",
        "sample_test.integer_mad_hi_ushort8",
        "sample_test.integer_mad_hi_ushort16",
        "sample_test.integer_mad_hi_int",
        "sample_test.integer_mad_hi_int2",
        "sample_test.integer_mad_hi_int3",
        "sample_test.integer_mad_hi_int4",
        "sample_test.integer_mad_hi_int8",
        "sample_test.integer_mad_hi_int16",
        "sample_test.integer_mad_hi_uint",
        "sample_test.integer_mad_hi_uint2",
        "sample_test.integer_mad_hi_uint3",
        "sample_test.integer_mad_hi_uint4",
        "sample_test.integer_mad_hi_uint8",
        "sample_test.integer_mad_hi_uint16",
        "sample_test.integer_mad_hi_long",
        "sample_test.integer_mad_hi_long2",
        "sample_test.integer_mad_hi_long3",
        "sample_test.integer_mad_hi_long4",
        "sample_test.integer_mad_hi_long8",
        "sample_test.integer_mad_hi_long16",
        "sample_test.integer_mad_hi_ulong",
        "sample_test.integer_mad_hi_ulong2",
        "sample_test.integer_mad_hi_ulong3",
        "sample_test.integer_mad_hi_ulong4",
        "sample_test.integer_mad_hi_ulong8",
        "sample_test.integer_mad_hi_ulong16",
        "sample_test.integer_min_char",
        "sample_test.integer_min_char2",
        "sample_test.integer_min_char3",
        "sample_test.integer_min_char4",
        "sample_test.integer_min_char8",
        "sample_test.integer_min_char16",
        "sample_test.integer_min_uchar",
        "sample_test.integer_min_uchar2",
        "sample_test.integer_min_uchar3",
        "sample_test.integer_min_uchar4",
        "sample_test.integer_min_uchar8",
        "sample_test.integer_min_uchar16",
        "sample_test.integer_min_short",
        "sample_test.integer_min_short2",
        "sample_test.integer_min_short3",
        "sample_test.integer_min_short4",
        "sample_test.integer_min_short8",
        "sample_test.integer_min_short16",
        "sample_test.integer_min_ushort",
        "sample_test.integer_min_ushort2",
        "sample_test.integer_min_ushort3",
        "sample_test.integer_min_ushort4",
        "sample_test.integer_min_ushort8",
        "sample_test.integer_min_ushort16",
        "sample_test.integer_min_int",
        "sample_test.integer_min_int2",
        "sample_test.integer_min_int3",
        "sample_test.integer_min_int4",
        "sample_test.integer_min_int8",
        "sample_test.integer_min_int16",
        "sample_test.integer_min_uint",
        "sample_test.integer_min_uint2",
        "sample_test.integer_min_uint3",
        "sample_test.integer_min_uint4",
        "sample_test.integer_min_uint8",
        "sample_test.integer_min_uint16",
        "sample_test.integer_min_long",
        "sample_test.integer_min_long2",
        "sample_test.integer_min_long3",
        "sample_test.integer_min_long4",
        "sample_test.integer_min_long8",
        "sample_test.integer_min_long16",
        "sample_test.integer_min_ulong",
        "sample_test.integer_min_ulong2",
        "sample_test.integer_min_ulong3",
        "sample_test.integer_min_ulong4",
        "sample_test.integer_min_ulong8",
        "sample_test.integer_min_ulong16",
        "sample_test.integer_max_char",
        "sample_test.integer_max_char2",
        "sample_test.integer_max_char3",
        "sample_test.integer_max_char4",
        "sample_test.integer_max_char8",
        "sample_test.integer_max_char16",
        "sample_test.integer_max_uchar",
        "sample_test.integer_max_uchar2",
        "sample_test.integer_max_uchar3",
        "sample_test.integer_max_uchar4",
        "sample_test.integer_max_uchar8",
        "sample_test.integer_max_uchar16",
        "sample_test.integer_max_short",
        "sample_test.integer_max_short2",
        "sample_test.integer_max_short3",
        "sample_test.integer_max_short4",
        "sample_test.integer_max_short8",
        "sample_test.integer_max_short16",
        "sample_test.integer_max_ushort",
        "sample_test.integer_max_ushort2",
        "sample_test.integer_max_ushort3",
        "sample_test.integer_max_ushort4",
        "sample_test.integer_max_ushort8",
        "sample_test.integer_max_ushort16",
        "sample_test.integer_max_int",
        "sample_test.integer_max_int2",
        "sample_test.integer_max_int3",
        "sample_test.integer_max_int4",
        "sample_test.integer_max_int8",
        "sample_test.integer_max_int16",
        "sample_test.integer_max_uint",
        "sample_test.integer_max_uint2",
        "sample_test.integer_max_uint3",
        "sample_test.integer_max_uint4",
        "sample_test.integer_max_uint8",
        "sample_test.integer_max_uint16",
        "sample_test.integer_max_long",
        "sample_test.integer_max_long2",
        "sample_test.integer_max_long3",
        "sample_test.integer_max_long4",
        "sample_test.integer_max_long8",
        "sample_test.integer_max_long16",
        "sample_test.integer_max_ulong",
        "sample_test.integer_max_ulong2",
        "sample_test.integer_max_ulong3",
        "sample_test.integer_max_ulong4",
        "sample_test.integer_max_ulong8",
        "sample_test.integer_max_ulong16",
        "test_upsample.integer_upsample_char",
        "test_upsample.integer_upsample_char2",
        "test_upsample.integer_upsample_char3",
        "test_upsample.integer_upsample_char4",
        "test_upsample.integer_upsample_char8",
        "test_upsample.integer_upsample_char16",
        "test_upsample.integer_upsample_uchar",
        "test_upsample.integer_upsample_uchar2",
        "test_upsample.integer_upsample_uchar3",
        "test_upsample.integer_upsample_uchar4",
        "test_upsample.integer_upsample_uchar8",
        "test_upsample.integer_upsample_uchar16",
        "test_upsample.integer_upsample_short",
        "test_upsample.integer_upsample_short2",
        "test_upsample.integer_upsample_short3",
        "test_upsample.integer_upsample_short4",
        "test_upsample.integer_upsample_short8",
        "test_upsample.integer_upsample_short16",
        "test_upsample.integer_upsample_ushort",
        "test_upsample.integer_upsample_ushort2",
        "test_upsample.integer_upsample_ushort3",
        "test_upsample.integer_upsample_ushort4",
        "test_upsample.integer_upsample_ushort8",
        "test_upsample.integer_upsample_ushort16",
        "test_upsample.integer_upsample_int",
        "test_upsample.integer_upsample_int2",
        "test_upsample.integer_upsample_int3",
        "test_upsample.integer_upsample_int4",
        "test_upsample.integer_upsample_int8",
        "test_upsample.integer_upsample_int16",
        "test_upsample.integer_upsample_uint",
        "test_upsample.integer_upsample_uint2",
        "test_upsample.integer_upsample_uint3",
        "test_upsample.integer_upsample_uint4",
        "test_upsample.integer_upsample_uint8",
        "test_upsample.integer_upsample_uint16",
        "test_abs_char",
        "test_abs_char2",
        "test_abs_char3",
        "test_abs_char4",
        "test_abs_char8",
        "test_abs_char16",
        "test_abs_short",
        "test_abs_short2",
        "test_abs_short3",
        "test_abs_short4",
        "test_abs_short8",
        "test_abs_short16",
        "test_abs_int",
        "test_abs_int2",
        "test_abs_int3",
        "test_abs_int4",
        "test_abs_int8",
        "test_abs_int16",
        "test_abs_long",
        "test_abs_long2",
        "test_abs_long3",
        "test_abs_long4",
        "test_abs_long8",
        "test_abs_long16",
        "test_abs_uchar",
        "test_abs_uchar2",
        "test_abs_uchar3",
        "test_abs_uchar4",
        "test_abs_uchar8",
        "test_abs_uchar16",
        "test_abs_ushort",
        "test_abs_ushort2",
        "test_abs_ushort3",
        "test_abs_ushort4",
        "test_abs_ushort8",
        "test_abs_ushort16",
        "test_abs_uint",
        "test_abs_uint2",
        "test_abs_uint3",
        "test_abs_uint4",
        "test_abs_uint8",
        "test_abs_uint16",
        "test_abs_ulong",
        "test_abs_ulong2",
        "test_abs_ulong3",
        "test_abs_ulong4",
        "test_abs_ulong8",
        "test_abs_ulong16",
        "test_absdiff_char",
        "test_absdiff_char2",
        "test_absdiff_char3",
        "test_absdiff_char4",
        "test_absdiff_char8",
        "test_absdiff_char16",
        "test_absdiff_uchar",
        "test_absdiff_uchar2",
        "test_absdiff_uchar3",
        "test_absdiff_uchar4",
        "test_absdiff_uchar8",
        "test_absdiff_uchar16",
        "test_absdiff_short",
        "test_absdiff_short2",
        "test_absdiff_short3",
        "test_absdiff_short4",
        "test_absdiff_short8",
        "test_absdiff_short16",
        "test_absdiff_ushort",
        "test_absdiff_ushort2",
        "test_absdiff_ushort3",
        "test_absdiff_ushort4",
        "test_absdiff_ushort8",
        "test_absdiff_ushort16",
        "test_absdiff_int",
        "test_absdiff_int2",
        "test_absdiff_int3",
        "test_absdiff_int4",
        "test_absdiff_int8",
        "test_absdiff_int16",
        "test_absdiff_uint",
        "test_absdiff_uint2",
        "test_absdiff_uint3",
        "test_absdiff_uint4",
        "test_absdiff_uint8",
        "test_absdiff_uint16",
        "test_absdiff_long",
        "test_absdiff_long2",
        "test_absdiff_long3",
        "test_absdiff_long4",
        "test_absdiff_long8",
        "test_absdiff_long16",
        "test_absdiff_ulong",
        "test_absdiff_ulong2",
        "test_absdiff_ulong3",
        "test_absdiff_ulong4",
        "test_absdiff_ulong8",
        "test_absdiff_ulong16",
        "test_add_sat_char",
        "test_add_sat_char2",
        "test_add_sat_char3",
        "test_add_sat_char4",
        "test_add_sat_char8",
        "test_add_sat_char16",
        "test_add_sat_uchar",
        "test_add_sat_uchar2",
        "test_add_sat_uchar3",
        "test_add_sat_uchar4",
        "test_add_sat_uchar8",
        "test_add_sat_uchar16",
        "test_add_sat_short",
        "test_add_sat_short2",
        "test_add_sat_short3",
        "test_add_sat_short4",
        "test_add_sat_short8",
        "test_add_sat_short16",
        "test_add_sat_ushort",
        "test_add_sat_ushort2",
        "test_add_sat_ushort3",
        "test_add_sat_ushort4",
        "test_add_sat_ushort8",
        "test_add_sat_ushort16",
        "test_add_sat_int",
        "test_add_sat_int2",
        "test_add_sat_int3",
        "test_add_sat_int4",
        "test_add_sat_int8",
        "test_add_sat_int16",
        "test_add_sat_uint",
        "test_add_sat_uint2",
        "test_add_sat_uint3",
        "test_add_sat_uint4",
        "test_add_sat_uint8",
        "test_add_sat_uint16",
        "test_add_sat_long",
        "test_add_sat_long2",
        "test_add_sat_long3",
        "test_add_sat_long4",
        "test_add_sat_long8",
        "test_add_sat_long16",
        "test_add_sat_ulong",
        "test_add_sat_ulong2",
        "test_add_sat_ulong3",
        "test_add_sat_ulong4",
        "test_add_sat_ulong8",
        "test_add_sat_ulong16",
        "test_sub_sat_char",
        "test_sub_sat_char2",
        "test_sub_sat_char3",
        "test_sub_sat_char4",
        "test_sub_sat_char8",
        "test_sub_sat_char16",
        "test_sub_sat_uchar",
        "test_sub_sat_uchar2",
        "test_sub_sat_uchar3",
        "test_sub_sat_uchar4",
        "test_sub_sat_uchar8",
        "test_sub_sat_uchar16",
        "test_sub_sat_short",
        "test_sub_sat_short2",
        "test_sub_sat_short3",
        "test_sub_sat_short4",
        "test_sub_sat_short8",
        "test_sub_sat_short16",
        "test_sub_sat_ushort",
        "test_sub_sat_ushort2",
        "test_sub_sat_ushort3",
        "test_sub_sat_ushort4",
        "test_sub_sat_ushort8",
        "test_sub_sat_ushort16",
        "test_sub_sat_int",
        "test_sub_sat_int2",
        "test_sub_sat_int3",
        "test_sub_sat_int4",
        "test_sub_sat_int8",
        "test_sub_sat_int16",
        "test_sub_sat_uint",
        "test_sub_sat_uint2",
        "test_sub_sat_uint3",
        "test_sub_sat_uint4",
        "test_sub_sat_uint8",
        "test_sub_sat_uint16",
        "test_sub_sat_long",
        "test_sub_sat_long2",
        "test_sub_sat_long3",
        "test_sub_sat_long4",
        "test_sub_sat_long8",
        "test_sub_sat_long16",
        "test_sub_sat_ulong",
        "test_sub_sat_ulong2",
        "test_sub_sat_ulong3",
        "test_sub_sat_ulong4",
        "test_sub_sat_ulong8",
        "test_sub_sat_ulong16",
        "test_int_mul24",
        "test_int2_mul24",
        "test_int3_mul24",
        "test_int4_mul24",
        "test_int8_mul24",
        "test_int16_mul24",
        "test_uint_mul24",
        "test_uint2_mul24",
        "test_uint3_mul24",
        "test_uint4_mul24",
        "test_uint8_mul24",
        "test_uint16_mul24",
        "test_int_mad24",
        "test_int2_mad24",
        "test_int3_mad24",
        "test_int4_mad24",
        "test_int8_mad24",
        "test_int16_mad24",
        "test_uint_mad24",
        "test_uint2_mad24",
        "test_uint3_mad24",
        "test_uint4_mad24",
        "test_uint8_mad24",
        "test_uint16_mad24",
        "test_popcount_char",
        "test_popcount_char2",
        "test_popcount_char3",
        "test_popcount_char4",
        "test_popcount_char8",
        "test_popcount_char16",
        "test_popcount_uchar",
        "test_popcount_uchar2",
        "test_popcount_uchar3",
        "test_popcount_uchar4",
        "test_popcount_uchar8",
        "test_popcount_uchar16",
        "test_popcount_short",
        "test_popcount_short2",
        "test_popcount_short3",
        "test_popcount_short4",
        "test_popcount_short8",
        "test_popcount_short16",
        "test_popcount_ushort",
        "test_popcount_ushort2",
        "test_popcount_ushort3",
        "test_popcount_ushort4",
        "test_popcount_ushort8",
        "test_popcount_ushort16",
        "test_popcount_int",
        "test_popcount_int2",
        "test_popcount_int3",
        "test_popcount_int4",
        "test_popcount_int8",
        "test_popcount_int16",
        "test_popcount_uint",
        "test_popcount_uint2",
        "test_popcount_uint3",
        "test_popcount_uint4",
        "test_popcount_uint8",
        "test_popcount_uint16",
        "test_popcount_long",
        "test_popcount_long2",
        "test_popcount_long3",
        "test_popcount_long4",
        "test_popcount_long8",
        "test_popcount_long16",
        "test_popcount_ulong",
        "test_popcount_ulong2",
        "test_popcount_ulong3",
        "test_popcount_ulong4",
        "test_popcount_ulong8",
        "test_popcount_ulong16",
    };

    log_info("test_integer_ops\n");
    return test_suite(device, size_t_width, folder, test_name, sizeof(test_name) / sizeof(const char *), "");
}


bool test_math_brute_force (cl_device_id device, cl_uint size_t_width, const char *folder)
{
    static const char* test_name[] = {
        "math_kernel.acos_float",
        "math_kernel3.acos_float3",
        "math_kernel16.acos_float16",
        "math_kernel2.acos_float2",
        "math_kernel4.acos_float4",
        "math_kernel8.acos_float8",
        "math_kernel16.acosh_float16",
        "math_kernel8.acosh_float8",
        "math_kernel4.acosh_float4",
        "math_kernel2.acosh_float2",
        "math_kernel3.acosh_float3",
        "math_kernel.acosh_float",
        "math_kernel16.acospi_float16",
        "math_kernel8.acospi_float8",
        "math_kernel3.acospi_float3",
        "math_kernel4.acospi_float4",
        "math_kernel2.acospi_float2",
        "math_kernel.acospi_float",
        "math_kernel16.asin_float16",
        "math_kernel8.asin_float8",
        "math_kernel4.asin_float4",
        "math_kernel3.asin_float3",
        "math_kernel2.asin_float2",
        "math_kernel.asin_float",
        "math_kernel8.asinh_float8",
        "math_kernel16.asinh_float16",
        "math_kernel4.asinh_float4",
        "math_kernel3.asinh_float3",
        "math_kernel2.asinh_float2",
        "math_kernel.asinh_float",
        "math_kernel8.asinpi_float8",
        "math_kernel16.asinpi_float16",
        "math_kernel3.asinpi_float3",
        "math_kernel4.asinpi_float4",
        "math_kernel2.asinpi_float2",
        "math_kernel.asinpi_float",
        "math_kernel16.atan_float16",
        "math_kernel8.atan_float8",
        "math_kernel4.atan_float4",
        "math_kernel2.atan_float2",
        "math_kernel3.atan_float3",
        "math_kernel.atan_float",
        "math_kernel16.atanh_float16",
        "math_kernel4.atanh_float4",
        "math_kernel8.atanh_float8",
        "math_kernel3.atanh_float3",
        "math_kernel.atanh_float",
        "math_kernel2.atanh_float2",
        "math_kernel16.atanpi_float16",
        "math_kernel8.atanpi_float8",
        "math_kernel4.atanpi_float4",
        "math_kernel3.atanpi_float3",
        "math_kernel2.atanpi_float2",
        "math_kernel.atanpi_float",
        "math_kernel8.atan2_float8",
        "math_kernel16.atan2_float16",
        "math_kernel4.atan2_float4",
        "math_kernel3.atan2_float3",
        "math_kernel2.atan2_float2",
        "math_kernel.atan2_float",
        "math_kernel16.atan2pi_float16",
        "math_kernel8.atan2pi_float8",
        "math_kernel4.atan2pi_float4",
        "math_kernel3.atan2pi_float3",
        "math_kernel.atan2pi_float",
        "math_kernel2.atan2pi_float2",
        "math_kernel16.cbrt_float16",
        "math_kernel8.cbrt_float8",
        "math_kernel4.cbrt_float4",
        "math_kernel2.cbrt_float2",
        "math_kernel3.cbrt_float3",
        "math_kernel.cbrt_float",
        "math_kernel4.ceil_float4",
        "math_kernel8.ceil_float8",
        "math_kernel3.ceil_float3",
        "math_kernel16.ceil_float16",
        "math_kernel2.ceil_float2",
        "math_kernel.ceil_float",
        "math_kernel16.copysign_float16",
        "math_kernel4.copysign_float4",
        "math_kernel2.copysign_float2",
        "math_kernel8.copysign_float8",
        "math_kernel3.copysign_float3",
        "math_kernel.copysign_float",
        "math_kernel8.cos_float8",
        "math_kernel16.cos_float16",
        "math_kernel4.cos_float4",
        "math_kernel3.cos_float3",
        "math_kernel2.cos_float2",
        "math_kernel.cos_float",
        "math_kernel8.cosh_float8",
        "math_kernel16.cosh_float16",
        "math_kernel4.cosh_float4",
        "math_kernel3.cosh_float3",
        "math_kernel2.cosh_float2",
        "math_kernel.cosh_float",
        "math_kernel16.cospi_float16",
        "math_kernel8.cospi_float8",
        "math_kernel4.cospi_float4",
        "math_kernel3.cospi_float3",
        "math_kernel2.cospi_float2",
        "math_kernel.cospi_float",
        "math_kernel4.div_float4",
        "math_kernel16.div_float16",
        "math_kernel8.div_float8",
        "math_kernel2.div_float2",
        "math_kernel3.div_float3",
        "math_kernel.div_float",
        "math_kernel4.div_cr_float4",
        "math_kernel16.div_cr_float16",
        "math_kernel8.div_cr_float8",
        "math_kernel2.div_cr_float2",
        "math_kernel3.div_cr_float3",
        "math_kernel.div_cr_float",
        "math_kernel16.exp_float16",
        "math_kernel4.exp_float4",
        "math_kernel3.exp_float3",
        "math_kernel8.exp_float8",
        "math_kernel2.exp_float2",
        "math_kernel.exp_float",
        "math_kernel8.exp2_float8",
        "math_kernel16.exp2_float16",
        "math_kernel4.exp2_float4",
        "math_kernel2.exp2_float2",
        "math_kernel3.exp2_float3",
        "math_kernel.exp2_float",
        "math_kernel16.exp10_float16",
        "math_kernel8.exp10_float8",
        "math_kernel3.exp10_float3",
        "math_kernel4.exp10_float4",
        "math_kernel2.exp10_float2",
        "math_kernel.exp10_float",
        "math_kernel8.expm1_float8",
        "math_kernel4.expm1_float4",
        "math_kernel16.expm1_float16",
        "math_kernel2.expm1_float2",
        "math_kernel3.expm1_float3",
        "math_kernel.expm1_float",
        "math_kernel16.fabs_float16",
        "math_kernel8.fabs_float8",
        "math_kernel4.fabs_float4",
        "math_kernel3.fabs_float3",
        "math_kernel.fabs_float",
        "math_kernel2.fabs_float2",
        "math_kernel16.fdim_float16",
        "math_kernel4.fdim_float4",
        "math_kernel8.fdim_float8",
        "math_kernel2.fdim_float2",
        "math_kernel.fdim_float",
        "math_kernel3.fdim_float3",
        "math_kernel8.floor_float8",
        "math_kernel16.floor_float16",
        "math_kernel4.floor_float4",
        "math_kernel3.floor_float3",
        "math_kernel2.floor_float2",
        "math_kernel.floor_float",
        "math_kernel2.fma_float2",
        "math_kernel16.fma_float16",
        "math_kernel3.fma_float3",
        "math_kernel4.fma_float4",
        "math_kernel.fma_float",
        "math_kernel8.fma_float8",
        "math_kernel8.fmax_float8",
        "math_kernel4.fmax_float4",
        "math_kernel3.fmax_float3",
        "math_kernel.fmax_float",
        "math_kernel16.fmax_float16",
        "math_kernel2.fmax_float2",
        "math_kernel16.fmin_float16",
        "math_kernel8.fmin_float8",
        "math_kernel3.fmin_float3",
        "math_kernel4.fmin_float4",
        "math_kernel2.fmin_float2",
        "math_kernel.fmin_float",
        "math_kernel16.fmod_float16",
        "math_kernel8.fmod_float8",
        "math_kernel4.fmod_float4",
        "math_kernel2.fmod_float2",
        "math_kernel3.fmod_float3",
        "math_kernel.fmod_float",
        "math_kernel16.fract_float16",
        "math_kernel4.fract_float4",
        "math_kernel2.fract_float2",
        "math_kernel3.fract_float3",
        "math_kernel.fract_float",
        "math_kernel8.fract_float8",
        "math_kernel2.frexp_float2",
        "math_kernel.frexp_float",
        "math_kernel4.frexp_float4",
        "math_kernel8.frexp_float8",
        "math_kernel3.frexp_float3",
        "math_kernel16.frexp_float16",
        "math_kernel4.hypot_float4",
        "math_kernel16.hypot_float16",
        "math_kernel8.hypot_float8",
        "math_kernel3.hypot_float3",
        "math_kernel2.hypot_float2",
        "math_kernel.hypot_float",
        "math_kernel16.ilogb_float16",
        "math_kernel3.ilogb_float3",
        "math_kernel8.ilogb_float8",
        "math_kernel2.ilogb_float2",
        "math_kernel.ilogb_float",
        "math_kernel4.ilogb_float4",
        "math_kernel.isequal_float",
        "math_kernel4.isequal_float4",
        "math_kernel8.isequal_float8",
        "math_kernel16.isequal_float16",
        "math_kernel3.isequal_float3",
        "math_kernel2.isequal_float2",
        "math_kernel2.isfinite_float2",
        "math_kernel16.isfinite_float16",
        "math_kernel8.isfinite_float8",
        "math_kernel.isfinite_float",
        "math_kernel4.isfinite_float4",
        "math_kernel3.isfinite_float3",
        "math_kernel16.isgreater_float16",
        "math_kernel8.isgreater_float8",
        "math_kernel4.isgreater_float4",
        "math_kernel3.isgreater_float3",
        "math_kernel2.isgreater_float2",
        "math_kernel.isgreater_float",
        "math_kernel8.isgreaterequal_float8",
        "math_kernel16.isgreaterequal_float16",
        "math_kernel4.isgreaterequal_float4",
        "math_kernel.isgreaterequal_float",
        "math_kernel3.isgreaterequal_float3",
        "math_kernel2.isgreaterequal_float2",
        "math_kernel4.isinf_float4",
        "math_kernel16.isinf_float16",
        "math_kernel8.isinf_float8",
        "math_kernel3.isinf_float3",
        "math_kernel2.isinf_float2",
        "math_kernel.isinf_float",
        "math_kernel16.isless_float16",
        "math_kernel8.isless_float8",
        "math_kernel4.isless_float4",
        "math_kernel3.isless_float3",
        "math_kernel2.isless_float2",
        "math_kernel.isless_float",
        "math_kernel8.islessequal_float8",
        "math_kernel16.islessequal_float16",
        "math_kernel2.islessequal_float2",
        "math_kernel3.islessequal_float3",
        "math_kernel4.islessequal_float4",
        "math_kernel.islessequal_float",
        "math_kernel8.islessgreater_float8",
        "math_kernel16.islessgreater_float16",
        "math_kernel4.islessgreater_float4",
        "math_kernel3.islessgreater_float3",
        "math_kernel2.islessgreater_float2",
        "math_kernel.islessgreater_float",
        "math_kernel4.isnan_float4",
        "math_kernel16.isnan_float16",
        "math_kernel8.isnan_float8",
        "math_kernel3.isnan_float3",
        "math_kernel2.isnan_float2",
        "math_kernel.isnan_float",
        "math_kernel16.isnormal_float16",
        "math_kernel8.isnormal_float8",
        "math_kernel4.isnormal_float4",
        "math_kernel3.isnormal_float3",
        "math_kernel2.isnormal_float2",
        "math_kernel.isnormal_float",
        "math_kernel16.isnotequal_float16",
        "math_kernel8.isnotequal_float8",
        "math_kernel4.isnotequal_float4",
        "math_kernel3.isnotequal_float3",
        "math_kernel2.isnotequal_float2",
        "math_kernel.isnotequal_float",
        "math_kernel16.isordered_float16",
        "math_kernel8.isordered_float8",
        "math_kernel3.isordered_float3",
        "math_kernel4.isordered_float4",
        "math_kernel2.isordered_float2",
        "math_kernel.isordered_float",
        "math_kernel16.isunordered_float16",
        "math_kernel8.isunordered_float8",
        "math_kernel4.isunordered_float4",
        "math_kernel2.isunordered_float2",
        "math_kernel3.isunordered_float3",
        "math_kernel.isunordered_float",
        "math_kernel8.ldexp_float8",
        "math_kernel2.ldexp_float2",
        "math_kernel3.ldexp_float3",
        "math_kernel16.ldexp_float16",
        "math_kernel4.ldexp_float4",
        "math_kernel.ldexp_float",
        "math_kernel4.lgamma_float4",
        "math_kernel16.lgamma_float16",
        "math_kernel8.lgamma_float8",
        "math_kernel2.lgamma_float2",
        "math_kernel.lgamma_float",
        "math_kernel3.lgamma_float3",
        "math_kernel16.lgamma_r_float16",
        "math_kernel8.lgamma_r_float8",
        "math_kernel4.lgamma_r_float4",
        "math_kernel3.lgamma_r_float3",
        "math_kernel.lgamma_r_float",
        "math_kernel2.lgamma_r_float2",
        "math_kernel16.log_float16",
        "math_kernel4.log_float4",
        "math_kernel8.log_float8",
        "math_kernel2.log_float2",
        "math_kernel.log_float",
        "math_kernel3.log_float3",
        "math_kernel16.log2_float16",
        "math_kernel4.log2_float4",
        "math_kernel8.log2_float8",
        "math_kernel2.log2_float2",
        "math_kernel.log2_float",
        "math_kernel3.log2_float3",
        "math_kernel8.log10_float8",
        "math_kernel4.log10_float4",
        "math_kernel16.log10_float16",
        "math_kernel2.log10_float2",
        "math_kernel.log10_float",
        "math_kernel3.log10_float3",
        "math_kernel16.log1p_float16",
        "math_kernel8.log1p_float8",
        "math_kernel4.log1p_float4",
        "math_kernel3.log1p_float3",
        "math_kernel2.log1p_float2",
        "math_kernel.log1p_float",
        "math_kernel16.logb_float16",
        "math_kernel8.logb_float8",
        "math_kernel4.logb_float4",
        "math_kernel3.logb_float3",
        "math_kernel2.logb_float2",
        "math_kernel.logb_float",
        "math_kernel16.mad_float16",
        "math_kernel8.mad_float8",
        "math_kernel4.mad_float4",
        "math_kernel2.mad_float2",
        "math_kernel3.mad_float3",
        "math_kernel.mad_float",
        "math_kernel8.maxmag_float8",
        "math_kernel16.maxmag_float16",
        "math_kernel4.maxmag_float4",
        "math_kernel3.maxmag_float3",
        "math_kernel2.maxmag_float2",
        "math_kernel.maxmag_float",
        "math_kernel16.minmag_float16",
        "math_kernel8.minmag_float8",
        "math_kernel4.minmag_float4",
        "math_kernel3.minmag_float3",
        "math_kernel2.minmag_float2",
        "math_kernel.minmag_float",
        "math_kernel16.modf_float16",
        "math_kernel8.modf_float8",
        "math_kernel3.modf_float3",
        "math_kernel4.modf_float4",
        "math_kernel2.modf_float2",
        "math_kernel.modf_float",
        "math_kernel16.nan_float16",
        "math_kernel8.nan_float8",
        "math_kernel4.nan_float4",
        "math_kernel2.nan_float2",
        "math_kernel.nan_float",
        "math_kernel3.nan_float3",
        "math_kernel8.nextafter_float8",
        "math_kernel16.nextafter_float16",
        "math_kernel4.nextafter_float4",
        "math_kernel2.nextafter_float2",
        "math_kernel3.nextafter_float3",
        "math_kernel.nextafter_float",
        "math_kernel16.pow_float16",
        "math_kernel8.pow_float8",
        "math_kernel4.pow_float4",
        "math_kernel3.pow_float3",
        "math_kernel2.pow_float2",
        "math_kernel.pow_float",
        "math_kernel4.pown_float4",
        "math_kernel8.pown_float8",
        "math_kernel16.pown_float16",
        "math_kernel3.pown_float3",
        "math_kernel2.pown_float2",
        "math_kernel.pown_float",
        "math_kernel16.powr_float16",
        "math_kernel8.powr_float8",
        "math_kernel4.powr_float4",
        "math_kernel2.powr_float2",
        "math_kernel3.powr_float3",
        "math_kernel.powr_float",
        "math_kernel4.remainder_float4",
        "math_kernel8.remainder_float8",
        "math_kernel16.remainder_float16",
        "math_kernel3.remainder_float3",
        "math_kernel2.remainder_float2",
        "math_kernel.remainder_float",
        "math_kernel8.remquo_float8",
        "math_kernel2.remquo_float2",
        "math_kernel3.remquo_float3",
        "math_kernel16.remquo_float16",
        "math_kernel4.remquo_float4",
        "math_kernel.remquo_float",
        "math_kernel8.rint_float8",
        "math_kernel16.rint_float16",
        "math_kernel4.rint_float4",
        "math_kernel3.rint_float3",
        "math_kernel.rint_float",
        "math_kernel2.rint_float2",
        "math_kernel16.rootn_float16",
        "math_kernel8.rootn_float8",
        "math_kernel3.rootn_float3",
        "math_kernel4.rootn_float4",
        "math_kernel.rootn_float",
        "math_kernel2.rootn_float2",
        "math_kernel8.round_float8",
        "math_kernel16.round_float16",
        "math_kernel4.round_float4",
        "math_kernel2.round_float2",
        "math_kernel3.round_float3",
        "math_kernel.round_float",
        "math_kernel8.rsqrt_float8",
        "math_kernel4.rsqrt_float4",
        "math_kernel16.rsqrt_float16",
        "math_kernel3.rsqrt_float3",
        "math_kernel.rsqrt_float",
        "math_kernel2.rsqrt_float2",
        "math_kernel8.signbit_float8",
        "math_kernel16.signbit_float16",
        "math_kernel4.signbit_float4",
        "math_kernel3.signbit_float3",
        "math_kernel2.signbit_float2",
        "math_kernel.signbit_float",
        "math_kernel8.sin_float8",
        "math_kernel4.sin_float4",
        "math_kernel16.sin_float16",
        "math_kernel2.sin_float2",
        "math_kernel3.sin_float3",
        "math_kernel.sin_float",
        "math_kernel8.sincos_float8",
        "math_kernel4.sincos_float4",
        "math_kernel16.sincos_float16",
        "math_kernel2.sincos_float2",
        "math_kernel3.sincos_float3",
        "math_kernel.sincos_float",
        "math_kernel8.sinh_float8",
        "math_kernel16.sinh_float16",
        "math_kernel4.sinh_float4",
        "math_kernel3.sinh_float3",
        "math_kernel2.sinh_float2",
        "math_kernel.sinh_float",
        "math_kernel16.sinpi_float16",
        "math_kernel4.sinpi_float4",
        "math_kernel3.sinpi_float3",
        "math_kernel.sinpi_float",
        "math_kernel8.sinpi_float8",
        "math_kernel2.sinpi_float2",
        "math_kernel4.sqrt_float4",
        "math_kernel16.sqrt_float16",
        "math_kernel8.sqrt_float8",
        "math_kernel2.sqrt_float2",
        "math_kernel3.sqrt_float3",
        "math_kernel.sqrt_float",
        "math_kernel4.sqrt_cr_float4",
        "math_kernel16.sqrt_cr_float16",
        "math_kernel8.sqrt_cr_float8",
        "math_kernel2.sqrt_cr_float2",
        "math_kernel3.sqrt_cr_float3",
        "math_kernel.sqrt_cr_float",
        "math_kernel8.tan_float8",
        "math_kernel16.tan_float16",
        "math_kernel4.tan_float4",
        "math_kernel.tan_float",
        "math_kernel3.tan_float3",
        "math_kernel2.tan_float2",
        "math_kernel16.tanh_float16",
        "math_kernel8.tanh_float8",
        "math_kernel4.tanh_float4",
        "math_kernel2.tanh_float2",
        "math_kernel.tanh_float",
        "math_kernel3.tanh_float3",
        "math_kernel16.tanpi_float16",
        "math_kernel8.tanpi_float8",
        "math_kernel4.tanpi_float4",
        "math_kernel3.tanpi_float3",
        "math_kernel2.tanpi_float2",
        "math_kernel.tanpi_float",
        "math_kernel8.trunc_float8",
        "math_kernel4.trunc_float4",
        "math_kernel16.trunc_float16",
        "math_kernel2.trunc_float2",
        "math_kernel3.trunc_float3",
        "math_kernel.trunc_float",
        "math_kernel16.trunc_double16",
        "math_kernel16.half_cos_float16",
        "math_kernel8.half_cos_float8",
        "math_kernel4.half_cos_float4",
        "math_kernel3.half_cos_float3",
        "math_kernel2.half_cos_float2",
        "math_kernel.half_cos_float",
        "math_kernel16.half_divide_float16",
        "math_kernel8.half_divide_float8",
        "math_kernel4.half_divide_float4",
        "math_kernel3.half_divide_float3",
        "math_kernel2.half_divide_float2",
        "math_kernel.half_divide_float",
        "math_kernel8.half_exp_float8",
        "math_kernel16.half_exp_float16",
        "math_kernel4.half_exp_float4",
        "math_kernel3.half_exp_float3",
        "math_kernel2.half_exp_float2",
        "math_kernel.half_exp_float",
        "math_kernel16.half_exp2_float16",
        "math_kernel4.half_exp2_float4",
        "math_kernel8.half_exp2_float8",
        "math_kernel.half_exp2_float",
        "math_kernel3.half_exp2_float3",
        "math_kernel2.half_exp2_float2",
        "math_kernel8.half_exp10_float8",
        "math_kernel4.half_exp10_float4",
        "math_kernel16.half_exp10_float16",
        "math_kernel2.half_exp10_float2",
        "math_kernel3.half_exp10_float3",
        "math_kernel.half_exp10_float",
        "math_kernel8.half_log_float8",
        "math_kernel16.half_log_float16",
        "math_kernel3.half_log_float3",
        "math_kernel.half_log_float",
        "math_kernel2.half_log_float2",
        "math_kernel4.half_log_float4",
        "math_kernel16.half_log2_float16",
        "math_kernel4.half_log2_float4",
        "math_kernel8.half_log2_float8",
        "math_kernel2.half_log2_float2",
        "math_kernel3.half_log2_float3",
        "math_kernel.half_log2_float",
        "math_kernel4.half_log10_float4",
        "math_kernel8.half_log10_float8",
        "math_kernel16.half_log10_float16",
        "math_kernel2.half_log10_float2",
        "math_kernel3.half_log10_float3",
        "math_kernel.half_log10_float",
        "math_kernel8.half_powr_float8",
        "math_kernel16.half_powr_float16",
        "math_kernel4.half_powr_float4",
        "math_kernel3.half_powr_float3",
        "math_kernel2.half_powr_float2",
        "math_kernel.half_powr_float",
        "math_kernel16.half_recip_float16",
        "math_kernel8.half_recip_float8",
        "math_kernel4.half_recip_float4",
        "math_kernel3.half_recip_float3",
        "math_kernel2.half_recip_float2",
        "math_kernel.half_recip_float",
        "math_kernel16.half_rsqrt_float16",
        "math_kernel8.half_rsqrt_float8",
        "math_kernel4.half_rsqrt_float4",
        "math_kernel3.half_rsqrt_float3",
        "math_kernel2.half_rsqrt_float2",
        "math_kernel.half_rsqrt_float",
        "math_kernel16.half_sin_float16",
        "math_kernel8.half_sin_float8",
        "math_kernel4.half_sin_float4",
        "math_kernel3.half_sin_float3",
        "math_kernel2.half_sin_float2",
        "math_kernel.half_sin_float",
        "math_kernel8.half_sqrt_float8",
        "math_kernel4.half_sqrt_float4",
        "math_kernel3.half_sqrt_float3",
        "math_kernel16.half_sqrt_float16",
        "math_kernel2.half_sqrt_float2",
        "math_kernel.half_sqrt_float",
        "math_kernel16.half_tan_float16",
        "math_kernel8.half_tan_float8",
        "math_kernel4.half_tan_float4",
        "math_kernel3.half_tan_float3",
        "math_kernel2.half_tan_float2",
        "math_kernel.half_tan_float",
    };

    log_info("test_math_brute_force\n");
    return test_suite(device, size_t_width, folder, test_name, sizeof(test_name) / sizeof(const char *), "");
}


bool test_math_brute_force_double (cl_device_id device, cl_uint size_t_width, const char *folder)
{
    static const char* test_name[] = {
        "math_kernel8.acos_double8",
        "math_kernel4.acos_double4",
        "math_kernel16.acos_double16",
        "math_kernel2.acos_double2",
        "math_kernel3.acos_double3",
        "math_kernel.acos_double",
        "math_kernel16.acosh_double16",
        "math_kernel8.acosh_double8",
        "math_kernel4.acosh_double4",
        "math_kernel2.acosh_double2",
        "math_kernel3.acosh_double3",
        "math_kernel.acosh_double",
        "math_kernel8.acospi_double8",
        "math_kernel16.acospi_double16",
        "math_kernel4.acospi_double4",
        "math_kernel3.acospi_double3",
        "math_kernel2.acospi_double2",
        "math_kernel.acospi_double",
        "math_kernel16.asin_double16",
        "math_kernel8.asin_double8",
        "math_kernel4.asin_double4",
        "math_kernel3.asin_double3",
        "math_kernel.asin_double",
        "math_kernel2.asin_double2",
        "math_kernel16.asinh_double16",
        "math_kernel8.asinh_double8",
        "math_kernel4.asinh_double4",
        "math_kernel2.asinh_double2",
        "math_kernel3.asinh_double3",
        "math_kernel.asinh_double",
        "math_kernel4.asinpi_double4",
        "math_kernel8.asinpi_double8",
        "math_kernel16.asinpi_double16",
        "math_kernel2.asinpi_double2",
        "math_kernel3.asinpi_double3",
        "math_kernel.asinpi_double",
        "math_kernel16.atan_double16",
        "math_kernel8.atan_double8",
        "math_kernel4.atan_double4",
        "math_kernel2.atan_double2",
        "math_kernel3.atan_double3",
        "math_kernel.atan_double",
        "math_kernel16.atanh_double16",
        "math_kernel8.atanh_double8",
        "math_kernel4.atanh_double4",
        "math_kernel3.atanh_double3",
        "math_kernel2.atanh_double2",
        "math_kernel.atanh_double",
         "math_kernel8.atanpi_double8",
        "math_kernel16.atanpi_double16",
        "math_kernel3.atanpi_double3",
        "math_kernel4.atanpi_double4",
        "math_kernel2.atanpi_double2",
        "math_kernel.atanpi_double",
        "math_kernel16.atan2_double16",
        "math_kernel8.atan2_double8",
        "math_kernel4.atan2_double4",
        "math_kernel2.atan2_double2",
        "math_kernel3.atan2_double3",
        "math_kernel.atan2_double",
        "math_kernel8.atan2pi_double8",
        "math_kernel4.atan2pi_double4",
        "math_kernel16.atan2pi_double16",
        "math_kernel3.atan2pi_double3",
        "math_kernel2.atan2pi_double2",
        "math_kernel.atan2pi_double",
        "math_kernel4.cbrt_double4",
        "math_kernel8.cbrt_double8",
        "math_kernel3.cbrt_double3",
        "math_kernel16.cbrt_double16",
        "math_kernel2.cbrt_double2",
        "math_kernel.cbrt_double",
        "math_kernel16.ceil_double16",
        "math_kernel4.ceil_double4",
        "math_kernel2.ceil_double2",
        "math_kernel8.ceil_double8",
        "math_kernel3.ceil_double3",
        "math_kernel.ceil_double",
        "math_kernel16.copysign_double16",
        "math_kernel8.copysign_double8",
        "math_kernel4.copysign_double4",
        "math_kernel2.copysign_double2",
        "math_kernel3.copysign_double3",
        "math_kernel.copysign_double",
        "math_kernel8.cos_double8",
        "math_kernel16.cos_double16",
        "math_kernel4.cos_double4",
        "math_kernel3.cos_double3",
        "math_kernel2.cos_double2",
        "math_kernel.cos_double",
        "math_kernel16.cosh_double16",
        "math_kernel8.cosh_double8",
        "math_kernel4.cosh_double4",
        "math_kernel3.cosh_double3",
        "math_kernel2.cosh_double2",
        "math_kernel.cosh_double",
        "math_kernel4.cospi_double4",
        "math_kernel16.cospi_double16",
        "math_kernel8.cospi_double8",
        "math_kernel3.cospi_double3",
        "math_kernel.cospi_double",
        "math_kernel2.cospi_double2",
        "math_kernel16.exp_double16",
        "math_kernel8.exp_double8",
        "math_kernel4.exp_double4",
        "math_kernel2.exp_double2",
        "math_kernel3.exp_double3",
        "math_kernel.exp_double",
        "math_kernel8.exp2_double8",
        "math_kernel16.exp2_double16",
        "math_kernel4.exp2_double4",
        "math_kernel3.exp2_double3",
        "math_kernel2.exp2_double2",
        "math_kernel.exp2_double",
        "math_kernel8.exp10_double8",
        "math_kernel4.exp10_double4",
        "math_kernel16.exp10_double16",
        "math_kernel3.exp10_double3",
        "math_kernel.exp10_double",
        "math_kernel2.exp10_double2",
        "math_kernel16.expm1_double16",
        "math_kernel8.expm1_double8",
        "math_kernel2.expm1_double2",
        "math_kernel4.expm1_double4",
        "math_kernel3.expm1_double3",
        "math_kernel.expm1_double",
        "math_kernel16.fabs_double16",
        "math_kernel8.fabs_double8",
        "math_kernel4.fabs_double4",
        "math_kernel3.fabs_double3",
        "math_kernel2.fabs_double2",
        "math_kernel.fabs_double",
        "math_kernel8.fdim_double8",
        "math_kernel16.fdim_double16",
        "math_kernel4.fdim_double4",
        "math_kernel3.fdim_double3",
        "math_kernel2.fdim_double2",
        "math_kernel.fdim_double",
        "math_kernel4.floor_double4",
        "math_kernel16.floor_double16",
        "math_kernel8.floor_double8",
        "math_kernel3.floor_double3",
        "math_kernel2.floor_double2",
        "math_kernel.floor_double",
        "math_kernel4.fma_double4",
        "math_kernel16.fma_double16",
        "math_kernel8.fma_double8",
        "math_kernel2.fma_double2",
        "math_kernel3.fma_double3",
        "math_kernel.fma_double",
        "math_kernel8.fmax_float8",
        "math_kernel4.fmax_float4",
        "math_kernel3.fmax_float3",
        "math_kernel.fmax_float",
        "math_kernel16.fmax_float16",
        "math_kernel2.fmax_float2",
        "math_kernel8.fmax_double8",
        "math_kernel16.fmax_double16",
        "math_kernel2.fmax_double2",
        "math_kernel4.fmax_double4",
        "math_kernel3.fmax_double3",
        "math_kernel.fmax_double",
        "math_kernel16.fmin_double16",
        "math_kernel8.fmin_double8",
        "math_kernel4.fmin_double4",
        "math_kernel3.fmin_double3",
        "math_kernel2.fmin_double2",
        "math_kernel.fmin_double",
        "math_kernel8.fmod_double8",
        "math_kernel16.fmod_double16",
        "math_kernel3.fmod_double3",
        "math_kernel4.fmod_double4",
        "math_kernel2.fmod_double2",
        "math_kernel.fmod_double",
        "math_kernel16.fract_double16",
        "math_kernel8.fract_double8",
        "math_kernel4.fract_double4",
        "math_kernel2.fract_double2",
        "math_kernel3.fract_double3",
        "math_kernel.fract_double",
        "math_kernel4.frexp_double4",
        "math_kernel8.frexp_double8",
        "math_kernel2.frexp_double2",
        "math_kernel3.frexp_double3",
        "math_kernel16.frexp_double16",
        "math_kernel.frexp_double",
        "math_kernel4.hypot_double4",
        "math_kernel8.hypot_double8",
        "math_kernel16.hypot_double16",
        "math_kernel2.hypot_double2",
        "math_kernel3.hypot_double3",
        "math_kernel.hypot_double",
        "math_kernel16.ilogb_double16",
        "math_kernel8.ilogb_double8",
        "math_kernel4.ilogb_double4",
        "math_kernel3.ilogb_double3",
        "math_kernel.ilogb_double",
        "math_kernel2.ilogb_double2",
        "math_kernel16.isequal_double16",
        "math_kernel8.isequal_double8",
        "math_kernel4.isequal_double4",
        "math_kernel3.isequal_double3",
        "math_kernel.isequal_double",
        "math_kernel2.isequal_double2",
        "math_kernel16.isfinite_double16",
        "math_kernel8.isfinite_double8",
        "math_kernel4.isfinite_double4",
        "math_kernel3.isfinite_double3",
        "math_kernel2.isfinite_double2",
        "math_kernel.isfinite_double",
        "math_kernel16.isgreater_double16",
        "math_kernel8.isgreater_double8",
        "math_kernel4.isgreater_double4",
        "math_kernel3.isgreater_double3",
        "math_kernel.isgreater_double",
        "math_kernel2.isgreater_double2",
        "math_kernel16.isgreaterequal_double16",
        "math_kernel8.isgreaterequal_double8",
        "math_kernel4.isgreaterequal_double4",
        "math_kernel3.isgreaterequal_double3",
        "math_kernel2.isgreaterequal_double2",
        "math_kernel.isgreaterequal_double",
        "math_kernel8.isinf_double8",
        "math_kernel16.isinf_double16",
        "math_kernel3.isinf_double3",
        "math_kernel4.isinf_double4",
        "math_kernel2.isinf_double2",
        "math_kernel.isinf_double",
        "math_kernel8.isless_double8",
        "math_kernel4.isless_double4",
        "math_kernel16.isless_double16",
        "math_kernel2.isless_double2",
        "math_kernel3.isless_double3",
        "math_kernel.isless_double",
        "math_kernel16.islessequal_double16",
        "math_kernel8.islessequal_double8",
        "math_kernel4.islessequal_double4",
        "math_kernel2.islessequal_double2",
        "math_kernel3.islessequal_double3",
        "math_kernel.islessequal_double",
        "math_kernel16.islessgreater_double16",
        "math_kernel3.islessgreater_double3",
        "math_kernel8.islessgreater_double8",
        "math_kernel4.islessgreater_double4",
        "math_kernel2.islessgreater_double2",
        "math_kernel.islessgreater_double",
        "math_kernel8.isnan_double8",
        "math_kernel4.isnan_double4",
        "math_kernel16.isnan_double16",
        "math_kernel3.isnan_double3",
        "math_kernel2.isnan_double2",
        "math_kernel.isnan_double",
        "math_kernel16.isnormal_double16",
        "math_kernel8.isnormal_double8",
        "math_kernel4.isnormal_double4",
        "math_kernel2.isnormal_double2",
        "math_kernel3.isnormal_double3",
        "math_kernel.isnormal_double",
        "math_kernel16.isnotequal_double16",
        "math_kernel4.isnotequal_double4",
        "math_kernel8.isnotequal_double8",
        "math_kernel3.isnotequal_double3",
        "math_kernel2.isnotequal_double2",
        "math_kernel.isnotequal_double",
        "math_kernel16.isordered_double16",
        "math_kernel3.isordered_double3",
        "math_kernel4.isordered_double4",
        "math_kernel8.isordered_double8",
        "math_kernel2.isordered_double2",
        "math_kernel.isordered_double",
        "math_kernel8.isunordered_double8",
        "math_kernel16.isunordered_double16",
        "math_kernel4.isunordered_double4",
        "math_kernel3.isunordered_double3",
        "math_kernel2.isunordered_double2",
        "math_kernel.isunordered_double",
        "math_kernel16.ldexp_double16",
        "math_kernel4.ldexp_double4",
        "math_kernel8.ldexp_double8",
        "math_kernel2.ldexp_double2",
        "math_kernel.ldexp_double",
        "math_kernel3.ldexp_double3",
        "math_kernel8.lgamma_double8",
        "math_kernel16.lgamma_double16",
        "math_kernel4.lgamma_double4",
        "math_kernel2.lgamma_double2",
        "math_kernel.lgamma_double",
        "math_kernel3.lgamma_double3",
        "math_kernel16.lgamma_r_double16",
        "math_kernel8.lgamma_r_double8",
        "math_kernel3.lgamma_r_double3",
        "math_kernel4.lgamma_r_double4",
        "math_kernel.lgamma_r_double",
        "math_kernel2.lgamma_r_double2",
        "math_kernel8.log_double8",
        "math_kernel16.log_double16",
        "math_kernel4.log_double4",
        "math_kernel3.log_double3",
        "math_kernel2.log_double2",
        "math_kernel.log_double",
        "math_kernel8.log2_double8",
        "math_kernel16.log2_double16",
        "math_kernel4.log2_double4",
        "math_kernel3.log2_double3",
        "math_kernel.log2_double",
        "math_kernel2.log2_double2",
        "math_kernel16.log10_double16",
        "math_kernel4.log10_double4",
        "math_kernel8.log10_double8",
        "math_kernel3.log10_double3",
        "math_kernel2.log10_double2",
        "math_kernel.log10_double",
        "math_kernel16.log1p_double16",
        "math_kernel4.log1p_double4",
        "math_kernel8.log1p_double8",
        "math_kernel2.log1p_double2",
        "math_kernel3.log1p_double3",
        "math_kernel.log1p_double",
        "math_kernel16.logb_double16",
        "math_kernel8.logb_double8",
        "math_kernel4.logb_double4",
        "math_kernel2.logb_double2",
        "math_kernel3.logb_double3",
        "math_kernel.logb_double",
        "math_kernel8.mad_double8",
        "math_kernel16.mad_double16",
        "math_kernel4.mad_double4",
        "math_kernel3.mad_double3",
        "math_kernel2.mad_double2",
        "math_kernel.mad_double",
        "math_kernel8.maxmag_double8",
        "math_kernel16.maxmag_double16",
        "math_kernel4.maxmag_double4",
        "math_kernel3.maxmag_double3",
        "math_kernel2.maxmag_double2",
        "math_kernel.maxmag_double",
        "math_kernel16.minmag_double16",
        "math_kernel8.minmag_double8",
        "math_kernel4.minmag_double4",
        "math_kernel3.minmag_double3",
        "math_kernel2.minmag_double2",
        "math_kernel.minmag_double",
        "math_kernel16.modf_double16",
        "math_kernel8.modf_double8",
        "math_kernel4.modf_double4",
        "math_kernel2.modf_double2",
        "math_kernel3.modf_double3",
        "math_kernel.modf_double",
        "math_kernel8.nan_double8",
        "math_kernel16.nan_double16",
        "math_kernel4.nan_double4",
        "math_kernel3.nan_double3",
        "math_kernel2.nan_double2",
        "math_kernel.nan_double",
        "math_kernel8.nextafter_double8",
        "math_kernel4.nextafter_double4",
        "math_kernel16.nextafter_double16",
        "math_kernel3.nextafter_double3",
        "math_kernel2.nextafter_double2",
        "math_kernel.nextafter_double",
        "math_kernel4.pow_double4",
        "math_kernel8.pow_double8",
        "math_kernel16.pow_double16",
        "math_kernel3.pow_double3",
        "math_kernel2.pow_double2",
        "math_kernel.pow_double",
        "math_kernel4.pown_double4",
        "math_kernel8.pown_double8",
        "math_kernel2.pown_double2",
        "math_kernel3.pown_double3",
        "math_kernel.pown_double",
        "math_kernel16.pown_double16",
        "math_kernel16.powr_double16",
        "math_kernel8.powr_double8",
        "math_kernel4.powr_double4",
        "math_kernel3.powr_double3",
        "math_kernel2.powr_double2",
        "math_kernel.powr_double",
        "math_kernel4.remainder_double4",
        "math_kernel8.remainder_double8",
        "math_kernel16.remainder_double16",
        "math_kernel2.remainder_double2",
        "math_kernel3.remainder_double3",
        "math_kernel.remainder_double",
        "math_kernel8.remquo_double8",
        "math_kernel16.remquo_double16",
        "math_kernel3.remquo_double3",
        "math_kernel4.remquo_double4",
        "math_kernel2.remquo_double2",
        "math_kernel.remquo_double",
        "math_kernel8.rint_double8",
        "math_kernel4.rint_double4",
        "math_kernel16.rint_double16",
        "math_kernel3.rint_double3",
        "math_kernel2.rint_double2",
        "math_kernel.rint_double",
        "math_kernel16.rootn_double16",
        "math_kernel8.rootn_double8",
        "math_kernel4.rootn_double4",
        "math_kernel3.rootn_double3",
        "math_kernel2.rootn_double2",
        "math_kernel.rootn_double",
        "math_kernel16.round_double16",
        "math_kernel8.round_double8",
        "math_kernel4.round_double4",
        "math_kernel3.round_double3",
        "math_kernel2.round_double2",
        "math_kernel.round_double",
        "math_kernel8.rsqrt_double8",
        "math_kernel4.rsqrt_double4",
        "math_kernel16.rsqrt_double16",
        "math_kernel3.rsqrt_double3",
        "math_kernel.rsqrt_double",
        "math_kernel2.rsqrt_double2",
        "math_kernel8.signbit_double8",
        "math_kernel4.signbit_double4",
        "math_kernel16.signbit_double16",
        "math_kernel2.signbit_double2",
        "math_kernel3.signbit_double3",
        "math_kernel.signbit_double",
        "math_kernel16.sin_double16",
        "math_kernel4.sin_double4",
        "math_kernel8.sin_double8",
        "math_kernel2.sin_double2",
        "math_kernel3.sin_double3",
        "math_kernel.sin_double",
        "math_kernel16.sincos_double16",
        "math_kernel8.sincos_double8",
        "math_kernel4.sincos_double4",
        "math_kernel3.sincos_double3",
        "math_kernel2.sincos_double2",
        "math_kernel.sincos_double",
        "math_kernel16.sinh_double16",
        "math_kernel4.sinh_double4",
        "math_kernel2.sinh_double2",
        "math_kernel8.sinh_double8",
        "math_kernel3.sinh_double3",
        "math_kernel.sinh_double",
        "math_kernel16.sinpi_double16",
        "math_kernel8.sinpi_double8",
        "math_kernel3.sinpi_double3",
        "math_kernel4.sinpi_double4",
        "math_kernel2.sinpi_double2",
        "math_kernel.sinpi_double",
        "math_kernel16.sqrt_double16",
        "math_kernel8.sqrt_double8",
        "math_kernel4.sqrt_double4",
        "math_kernel2.sqrt_double2",
        "math_kernel3.sqrt_double3",
        "math_kernel.sqrt_double",
        "math_kernel8.tan_double8",
        "math_kernel16.tan_double16",
        "math_kernel.tan_double",
        "math_kernel3.tan_double3",
        "math_kernel4.tan_double4",
        "math_kernel2.tan_double2",
        "math_kernel4.tanh_double4",
        "math_kernel8.tanh_double8",
        "math_kernel2.tanh_double2",
        "math_kernel16.tanh_double16",
        "math_kernel3.tanh_double3",
        "math_kernel.tanh_double",
        "math_kernel16.tanpi_double16",
        "math_kernel4.tanpi_double4",
        "math_kernel8.tanpi_double8",
        "math_kernel3.tanpi_double3",
        "math_kernel.tanpi_double",
        "math_kernel2.tanpi_double2",
        "math_kernel16.trunc_double16",
        "math_kernel8.trunc_double8",
        "math_kernel4.trunc_double4",
        "math_kernel3.trunc_double3",
        "math_kernel2.trunc_double2",
        "math_kernel.trunc_double",
    };

    log_info("test_math_brute_force_double\n");
    return test_suite(device, size_t_width, folder, test_name, sizeof(test_name) / sizeof(const char *), "cl_khr_fp64");
}


bool test_printf (cl_device_id device, cl_uint size_t_width, const char *folder)
{
    static const char* test_name[] = {
        "test0.testCaseInt",
        "test1.testCaseFloat",
        "test5.testCaseChar",
        "test6.testCaseString",
        "test7.testCaseVector_float",
        "test7.testCaseVector_long",
        "test7.testCaseVector_uchar",
        "test7.testCaseVector_uint",
        "test8.testCaseAddrSpace_constant",
        "test8.testCaseAddrSpace_global",
        "test8.testCaseAddrSpace_local",
        "test8.testCaseAddrSpace_private",
    };

    log_info("test_printf\n");
    return test_suite(device, size_t_width, folder, test_name, sizeof(test_name) / sizeof(const char *), "");
}


bool test_profiling (cl_device_id device, cl_uint size_t_width, const char *folder)
{
    static const char* test_name[] = {
        "testReadf",
        "image_filter",
    };

    log_info("test_profiling\n");
    return test_suite(device, size_t_width, folder, test_name, sizeof(test_name) / sizeof(const char *), "");
}


bool test_relationals (cl_device_id device, cl_uint size_t_width, const char *folder)
{
    static const char* test_name[] = {
        "sample_test.relational_any_char",
        "sample_test.relational_any_char2",
        "sample_test.relational_any_char3",
        "sample_test.relational_any_char4",
        "sample_test.relational_any_char8",
        "sample_test.relational_any_char16",
        "sample_test.relational_any_short",
        "sample_test.relational_any_short2",
        "sample_test.relational_any_short3",
        "sample_test.relational_any_short4",
        "sample_test.relational_any_short8",
        "sample_test.relational_any_short16",
        "sample_test.relational_any_int",
        "sample_test.relational_any_int2",
        "sample_test.relational_any_int3",
        "sample_test.relational_any_int4",
        "sample_test.relational_any_int8",
        "sample_test.relational_any_int16",
        "sample_test.relational_any_long",
        "sample_test.relational_any_long2",
        "sample_test.relational_any_long3",
        "sample_test.relational_any_long4",
        "sample_test.relational_any_long8",
        "sample_test.relational_any_long16",
        "sample_test.relational_all_char",
        "sample_test.relational_all_char2",
        "sample_test.relational_all_char3",
        "sample_test.relational_all_char4",
        "sample_test.relational_all_char8",
        "sample_test.relational_all_char16",
        "sample_test.relational_all_short",
        "sample_test.relational_all_short2",
        "sample_test.relational_all_short3",
        "sample_test.relational_all_short4",
        "sample_test.relational_all_short8",
        "sample_test.relational_all_short16",
        "sample_test.relational_all_int",
        "sample_test.relational_all_int2",
        "sample_test.relational_all_int3",
        "sample_test.relational_all_int4",
        "sample_test.relational_all_int8",
        "sample_test.relational_all_int16",
        "sample_test.relational_all_long",
        "sample_test.relational_all_long2",
        "sample_test.relational_all_long3",
        "sample_test.relational_all_long4",
        "sample_test.relational_all_long8",
        "sample_test.relational_all_long16",
        "sample_test.relational_bitselect_char",
        "sample_test.relational_bitselect_char2",
        "sample_test.relational_bitselect_char3",
        "sample_test.relational_bitselect_char4",
        "sample_test.relational_bitselect_char8",
        "sample_test.relational_bitselect_char16",
        "sample_test.relational_bitselect_uchar",
        "sample_test.relational_bitselect_uchar2",
        "sample_test.relational_bitselect_uchar3",
        "sample_test.relational_bitselect_uchar4",
        "sample_test.relational_bitselect_uchar8",
        "sample_test.relational_bitselect_uchar16",
        "sample_test.relational_bitselect_short",
        "sample_test.relational_bitselect_short2",
        "sample_test.relational_bitselect_short3",
        "sample_test.relational_bitselect_short4",
        "sample_test.relational_bitselect_short8",
        "sample_test.relational_bitselect_short16",
        "sample_test.relational_bitselect_ushort",
        "sample_test.relational_bitselect_ushort2",
        "sample_test.relational_bitselect_ushort3",
        "sample_test.relational_bitselect_ushort4",
        "sample_test.relational_bitselect_ushort8",
        "sample_test.relational_bitselect_ushort16",
        "sample_test.relational_bitselect_int",
        "sample_test.relational_bitselect_int2",
        "sample_test.relational_bitselect_int3",
        "sample_test.relational_bitselect_int4",
        "sample_test.relational_bitselect_int8",
        "sample_test.relational_bitselect_int16",
        "sample_test.relational_bitselect_uint",
        "sample_test.relational_bitselect_uint2",
        "sample_test.relational_bitselect_uint3",
        "sample_test.relational_bitselect_uint4",
        "sample_test.relational_bitselect_uint8",
        "sample_test.relational_bitselect_uint16",
        "sample_test.relational_bitselect_long",
        "sample_test.relational_bitselect_long2",
        "sample_test.relational_bitselect_long3",
        "sample_test.relational_bitselect_long4",
        "sample_test.relational_bitselect_long8",
        "sample_test.relational_bitselect_long16",
        "sample_test.relational_bitselect_ulong",
        "sample_test.relational_bitselect_ulong2",
        "sample_test.relational_bitselect_ulong3",
        "sample_test.relational_bitselect_ulong4",
        "sample_test.relational_bitselect_ulong8",
        "sample_test.relational_bitselect_ulong16",
        "sample_test.relational_bitselect_float",
        "sample_test.relational_bitselect_float2",
        "sample_test.relational_bitselect_float3",
        "sample_test.relational_bitselect_float4",
        "sample_test.relational_bitselect_float8",
        "sample_test.relational_bitselect_float16",
        "sample_test.relational_select_signed_char",
        "sample_test.relational_select_signed_char2",
        "sample_test.relational_select_signed_char4",
        "sample_test.relational_select_signed_char8",
        "sample_test.relational_select_signed_char16",
        "sample_test.relational_select_signed_short",
        "sample_test.relational_select_signed_short2",
        "sample_test.relational_select_signed_short4",
        "sample_test.relational_select_signed_short8",
        "sample_test.relational_select_signed_short16",
        "sample_test.relational_select_signed_int",
        "sample_test.relational_select_signed_int2",
        "sample_test.relational_select_signed_int4",
        "sample_test.relational_select_signed_int8",
        "sample_test.relational_select_signed_int16",
        "sample_test.relational_select_signed_long",
        "sample_test.relational_select_signed_long2",
        "sample_test.relational_select_signed_long4",
        "sample_test.relational_select_signed_long8",
        "sample_test.relational_select_signed_long16",
        "sample_test.relational_select_unsigned_uchar",
        "sample_test.relational_select_unsigned_uchar2",
        "sample_test.relational_select_unsigned_uchar4",
        "sample_test.relational_select_unsigned_uchar8",
        "sample_test.relational_select_unsigned_uchar16",
        "sample_test.relational_select_unsigned_ushort",
        "sample_test.relational_select_unsigned_ushort2",
        "sample_test.relational_select_unsigned_ushort4",
        "sample_test.relational_select_unsigned_ushort8",
        "sample_test.relational_select_unsigned_ushort16",
        "sample_test.relational_select_unsigned_uint",
        "sample_test.relational_select_unsigned_uint2",
        "sample_test.relational_select_unsigned_uint4",
        "sample_test.relational_select_unsigned_uint8",
        "sample_test.relational_select_unsigned_uint16",
        "sample_test.relational_select_unsigned_ulong",
        "sample_test.relational_select_unsigned_ulong2",
        "sample_test.relational_select_unsigned_ulong4",
        "sample_test.relational_select_unsigned_ulong8",
        "sample_test.relational_select_unsigned_ulong16",
        "sample_test.relational_isequal_float",
        "sample_test.relational_isequal_float2",
        "sample_test.relational_isequal_float3",
        "sample_test.relational_isequal_float4",
        "sample_test.relational_isequal_float8",
        "sample_test.relational_isequal_float16",
        "sample_test.relational_isnotequal_float",
        "sample_test.relational_isnotequal_float2",
        "sample_test.relational_isnotequal_float3",
        "sample_test.relational_isnotequal_float4",
        "sample_test.relational_isnotequal_float8",
        "sample_test.relational_isnotequal_float16",
        "sample_test.relational_isgreater_float",
        "sample_test.relational_isgreater_float2",
        "sample_test.relational_isgreater_float3",
        "sample_test.relational_isgreater_float4",
        "sample_test.relational_isgreater_float8",
        "sample_test.relational_isgreater_float16",
        "sample_test.relational_isgreaterequal_float",
        "sample_test.relational_isgreaterequal_float2",
        "sample_test.relational_isgreaterequal_float3",
        "sample_test.relational_isgreaterequal_float4",
        "sample_test.relational_isgreaterequal_float8",
        "sample_test.relational_isgreaterequal_float16",
        "sample_test.relational_isless_float",
        "sample_test.relational_isless_float2",
        "sample_test.relational_isless_float3",
        "sample_test.relational_isless_float4",
        "sample_test.relational_isless_float8",
        "sample_test.relational_isless_float16",
        "sample_test.relational_islessequal_float",
        "sample_test.relational_islessequal_float2",
        "sample_test.relational_islessequal_float3",
        "sample_test.relational_islessequal_float4",
        "sample_test.relational_islessequal_float8",
        "sample_test.relational_islessequal_float16",
        "sample_test.relational_islessgreater_float",
        "sample_test.relational_islessgreater_float2",
        "sample_test.relational_islessgreater_float3",
        "sample_test.relational_islessgreater_float4",
        "sample_test.relational_islessgreater_float8",
        "sample_test.relational_islessgreater_float16",
        "sample_test.shuffle_built_in_char2_char2",
        "sample_test.shuffle_built_in_char2_char4",
        "sample_test.shuffle_built_in_char2_char8",
        "sample_test.shuffle_built_in_char2_char16",
        "sample_test.shuffle_built_in_char4_char2",
        "sample_test.shuffle_built_in_char4_char4",
        "sample_test.shuffle_built_in_char4_char8",
        "sample_test.shuffle_built_in_char4_char16",
        "sample_test.shuffle_built_in_char8_char2",
        "sample_test.shuffle_built_in_char8_char4",
        "sample_test.shuffle_built_in_char8_char8",
        "sample_test.shuffle_built_in_char8_char16",
        "sample_test.shuffle_built_in_char16_char2",
        "sample_test.shuffle_built_in_char16_char4",
        "sample_test.shuffle_built_in_char16_char8",
        "sample_test.shuffle_built_in_char16_char16",
        "sample_test.shuffle_built_in_uchar2_uchar2",
        "sample_test.shuffle_built_in_uchar2_uchar4",
        "sample_test.shuffle_built_in_uchar2_uchar8",
        "sample_test.shuffle_built_in_uchar2_uchar16",
        "sample_test.shuffle_built_in_uchar4_uchar2",
        "sample_test.shuffle_built_in_uchar4_uchar4",
        "sample_test.shuffle_built_in_uchar4_uchar8",
        "sample_test.shuffle_built_in_uchar4_uchar16",
        "sample_test.shuffle_built_in_uchar8_uchar2",
        "sample_test.shuffle_built_in_uchar8_uchar4",
        "sample_test.shuffle_built_in_uchar8_uchar8",
        "sample_test.shuffle_built_in_uchar8_uchar16",
        "sample_test.shuffle_built_in_uchar16_uchar2",
        "sample_test.shuffle_built_in_uchar16_uchar4",
        "sample_test.shuffle_built_in_uchar16_uchar8",
        "sample_test.shuffle_built_in_uchar16_uchar16",
        "sample_test.shuffle_built_in_short2_short2",
        "sample_test.shuffle_built_in_short2_short4",
        "sample_test.shuffle_built_in_short2_short8",
        "sample_test.shuffle_built_in_short2_short16",
        "sample_test.shuffle_built_in_short4_short2",
        "sample_test.shuffle_built_in_short4_short4",
        "sample_test.shuffle_built_in_short4_short8",
        "sample_test.shuffle_built_in_short4_short16",
        "sample_test.shuffle_built_in_short8_short2",
        "sample_test.shuffle_built_in_short8_short4",
        "sample_test.shuffle_built_in_short8_short8",
        "sample_test.shuffle_built_in_short8_short16",
        "sample_test.shuffle_built_in_short16_short2",
        "sample_test.shuffle_built_in_short16_short4",
        "sample_test.shuffle_built_in_short16_short8",
        "sample_test.shuffle_built_in_short16_short16",
        "sample_test.shuffle_built_in_ushort2_ushort2",
        "sample_test.shuffle_built_in_ushort2_ushort4",
        "sample_test.shuffle_built_in_ushort2_ushort8",
        "sample_test.shuffle_built_in_ushort2_ushort16",
        "sample_test.shuffle_built_in_ushort4_ushort2",
        "sample_test.shuffle_built_in_ushort4_ushort4",
        "sample_test.shuffle_built_in_ushort4_ushort8",
        "sample_test.shuffle_built_in_ushort4_ushort16",
        "sample_test.shuffle_built_in_ushort8_ushort2",
        "sample_test.shuffle_built_in_ushort8_ushort4",
        "sample_test.shuffle_built_in_ushort8_ushort8",
        "sample_test.shuffle_built_in_ushort8_ushort16",
        "sample_test.shuffle_built_in_ushort16_ushort2",
        "sample_test.shuffle_built_in_ushort16_ushort4",
        "sample_test.shuffle_built_in_ushort16_ushort8",
        "sample_test.shuffle_built_in_ushort16_ushort16",
        "sample_test.shuffle_built_in_int2_int2",
        "sample_test.shuffle_built_in_int2_int4",
        "sample_test.shuffle_built_in_int2_int8",
        "sample_test.shuffle_built_in_int2_int16",
        "sample_test.shuffle_built_in_int4_int2",
        "sample_test.shuffle_built_in_int4_int4",
        "sample_test.shuffle_built_in_int4_int8",
        "sample_test.shuffle_built_in_int4_int16",
        "sample_test.shuffle_built_in_int8_int2",
        "sample_test.shuffle_built_in_int8_int4",
        "sample_test.shuffle_built_in_int8_int8",
        "sample_test.shuffle_built_in_int8_int16",
        "sample_test.shuffle_built_in_int16_int2",
        "sample_test.shuffle_built_in_int16_int4",
        "sample_test.shuffle_built_in_int16_int8",
        "sample_test.shuffle_built_in_int16_int16",
        "sample_test.shuffle_built_in_uint2_uint2",
        "sample_test.shuffle_built_in_uint2_uint4",
        "sample_test.shuffle_built_in_uint2_uint8",
        "sample_test.shuffle_built_in_uint2_uint16",
        "sample_test.shuffle_built_in_uint4_uint2",
        "sample_test.shuffle_built_in_uint4_uint4",
        "sample_test.shuffle_built_in_uint4_uint8",
        "sample_test.shuffle_built_in_uint4_uint16",
        "sample_test.shuffle_built_in_uint8_uint2",
        "sample_test.shuffle_built_in_uint8_uint4",
        "sample_test.shuffle_built_in_uint8_uint8",
        "sample_test.shuffle_built_in_uint8_uint16",
        "sample_test.shuffle_built_in_uint16_uint2",
        "sample_test.shuffle_built_in_uint16_uint4",
        "sample_test.shuffle_built_in_uint16_uint8",
        "sample_test.shuffle_built_in_uint16_uint16",
        "sample_test.shuffle_built_in_long2_long2",
        "sample_test.shuffle_built_in_long2_long4",
        "sample_test.shuffle_built_in_long2_long8",
        "sample_test.shuffle_built_in_long2_long16",
        "sample_test.shuffle_built_in_long4_long2",
        "sample_test.shuffle_built_in_long4_long4",
        "sample_test.shuffle_built_in_long4_long8",
        "sample_test.shuffle_built_in_long4_long16",
        "sample_test.shuffle_built_in_long8_long2",
        "sample_test.shuffle_built_in_long8_long4",
        "sample_test.shuffle_built_in_long8_long8",
        "sample_test.shuffle_built_in_long8_long16",
        "sample_test.shuffle_built_in_long16_long2",
        "sample_test.shuffle_built_in_long16_long4",
        "sample_test.shuffle_built_in_long16_long8",
        "sample_test.shuffle_built_in_long16_long16",
        "sample_test.shuffle_built_in_ulong2_ulong2",
        "sample_test.shuffle_built_in_ulong2_ulong4",
        "sample_test.shuffle_built_in_ulong2_ulong8",
        "sample_test.shuffle_built_in_ulong2_ulong16",
        "sample_test.shuffle_built_in_ulong4_ulong2",
        "sample_test.shuffle_built_in_ulong4_ulong4",
        "sample_test.shuffle_built_in_ulong4_ulong8",
        "sample_test.shuffle_built_in_ulong4_ulong16",
        "sample_test.shuffle_built_in_ulong8_ulong2",
        "sample_test.shuffle_built_in_ulong8_ulong4",
        "sample_test.shuffle_built_in_ulong8_ulong8",
        "sample_test.shuffle_built_in_ulong8_ulong16",
        "sample_test.shuffle_built_in_ulong16_ulong2",
        "sample_test.shuffle_built_in_ulong16_ulong4",
        "sample_test.shuffle_built_in_ulong16_ulong8",
        "sample_test.shuffle_built_in_ulong16_ulong16",
        "sample_test.shuffle_built_in_float2_float2",
        "sample_test.shuffle_built_in_float2_float4",
        "sample_test.shuffle_built_in_float2_float8",
        "sample_test.shuffle_built_in_float2_float16",
        "sample_test.shuffle_built_in_float4_float2",
        "sample_test.shuffle_built_in_float4_float4",
        "sample_test.shuffle_built_in_float4_float8",
        "sample_test.shuffle_built_in_float4_float16",
        "sample_test.shuffle_built_in_float8_float2",
        "sample_test.shuffle_built_in_float8_float4",
        "sample_test.shuffle_built_in_float8_float8",
        "sample_test.shuffle_built_in_float8_float16",
        "sample_test.shuffle_built_in_float16_float2",
        "sample_test.shuffle_built_in_float16_float4",
        "sample_test.shuffle_built_in_float16_float8",
        "sample_test.shuffle_built_in_float16_float16",
        "sample_test.shuffle_built_in_dual_input_char2_char2",
        "sample_test.shuffle_built_in_dual_input_char2_char4",
        "sample_test.shuffle_built_in_dual_input_char2_char8",
        "sample_test.shuffle_built_in_dual_input_char2_char16",
        "sample_test.shuffle_built_in_dual_input_char4_char2",
        "sample_test.shuffle_built_in_dual_input_char4_char4",
        "sample_test.shuffle_built_in_dual_input_char4_char8",
        "sample_test.shuffle_built_in_dual_input_char4_char16",
        "sample_test.shuffle_built_in_dual_input_char8_char2",
        "sample_test.shuffle_built_in_dual_input_char8_char4",
        "sample_test.shuffle_built_in_dual_input_char8_char8",
        "sample_test.shuffle_built_in_dual_input_char8_char16",
        "sample_test.shuffle_built_in_dual_input_char16_char2",
        "sample_test.shuffle_built_in_dual_input_char16_char4",
        "sample_test.shuffle_built_in_dual_input_char16_char8",
        "sample_test.shuffle_built_in_dual_input_char16_char16",
        "sample_test.shuffle_built_in_dual_input_uchar2_uchar2",
        "sample_test.shuffle_built_in_dual_input_uchar2_uchar4",
        "sample_test.shuffle_built_in_dual_input_uchar2_uchar8",
        "sample_test.shuffle_built_in_dual_input_uchar2_uchar16",
        "sample_test.shuffle_built_in_dual_input_uchar4_uchar2",
        "sample_test.shuffle_built_in_dual_input_uchar4_uchar4",
        "sample_test.shuffle_built_in_dual_input_uchar4_uchar8",
        "sample_test.shuffle_built_in_dual_input_uchar4_uchar16",
        "sample_test.shuffle_built_in_dual_input_uchar8_uchar2",
        "sample_test.shuffle_built_in_dual_input_uchar8_uchar4",
        "sample_test.shuffle_built_in_dual_input_uchar8_uchar8",
        "sample_test.shuffle_built_in_dual_input_uchar8_uchar16",
        "sample_test.shuffle_built_in_dual_input_uchar16_uchar2",
        "sample_test.shuffle_built_in_dual_input_uchar16_uchar4",
        "sample_test.shuffle_built_in_dual_input_uchar16_uchar8",
        "sample_test.shuffle_built_in_dual_input_uchar16_uchar16",
        "sample_test.shuffle_built_in_dual_input_short2_short2",
        "sample_test.shuffle_built_in_dual_input_short2_short4",
        "sample_test.shuffle_built_in_dual_input_short2_short8",
        "sample_test.shuffle_built_in_dual_input_short2_short16",
        "sample_test.shuffle_built_in_dual_input_short4_short2",
        "sample_test.shuffle_built_in_dual_input_short4_short4",
        "sample_test.shuffle_built_in_dual_input_short4_short8",
        "sample_test.shuffle_built_in_dual_input_short4_short16",
        "sample_test.shuffle_built_in_dual_input_short8_short2",
        "sample_test.shuffle_built_in_dual_input_short8_short4",
        "sample_test.shuffle_built_in_dual_input_short8_short8",
        "sample_test.shuffle_built_in_dual_input_short8_short16",
        "sample_test.shuffle_built_in_dual_input_short16_short2",
        "sample_test.shuffle_built_in_dual_input_short16_short4",
        "sample_test.shuffle_built_in_dual_input_short16_short8",
        "sample_test.shuffle_built_in_dual_input_short16_short16",
        "sample_test.shuffle_built_in_dual_input_ushort2_ushort2",
        "sample_test.shuffle_built_in_dual_input_ushort2_ushort4",
        "sample_test.shuffle_built_in_dual_input_ushort2_ushort8",
        "sample_test.shuffle_built_in_dual_input_ushort2_ushort16",
        "sample_test.shuffle_built_in_dual_input_ushort4_ushort2",
        "sample_test.shuffle_built_in_dual_input_ushort4_ushort4",
        "sample_test.shuffle_built_in_dual_input_ushort4_ushort8",
        "sample_test.shuffle_built_in_dual_input_ushort4_ushort16",
        "sample_test.shuffle_built_in_dual_input_ushort8_ushort2",
        "sample_test.shuffle_built_in_dual_input_ushort8_ushort4",
        "sample_test.shuffle_built_in_dual_input_ushort8_ushort8",
        "sample_test.shuffle_built_in_dual_input_ushort8_ushort16",
        "sample_test.shuffle_built_in_dual_input_ushort16_ushort2",
        "sample_test.shuffle_built_in_dual_input_ushort16_ushort4",
        "sample_test.shuffle_built_in_dual_input_ushort16_ushort8",
        "sample_test.shuffle_built_in_dual_input_ushort16_ushort16",
        "sample_test.shuffle_built_in_dual_input_int2_int2",
        "sample_test.shuffle_built_in_dual_input_int2_int4",
        "sample_test.shuffle_built_in_dual_input_int2_int8",
        "sample_test.shuffle_built_in_dual_input_int2_int16",
        "sample_test.shuffle_built_in_dual_input_int4_int2",
        "sample_test.shuffle_built_in_dual_input_int4_int4",
        "sample_test.shuffle_built_in_dual_input_int4_int8",
        "sample_test.shuffle_built_in_dual_input_int4_int16",
        "sample_test.shuffle_built_in_dual_input_int8_int2",
        "sample_test.shuffle_built_in_dual_input_int8_int4",
        "sample_test.shuffle_built_in_dual_input_int8_int8",
        "sample_test.shuffle_built_in_dual_input_int8_int16",
        "sample_test.shuffle_built_in_dual_input_int16_int2",
        "sample_test.shuffle_built_in_dual_input_int16_int4",
        "sample_test.shuffle_built_in_dual_input_int16_int8",
        "sample_test.shuffle_built_in_dual_input_int16_int16",
        "sample_test.shuffle_built_in_dual_input_uint2_uint2",
        "sample_test.shuffle_built_in_dual_input_uint2_uint4",
        "sample_test.shuffle_built_in_dual_input_uint2_uint8",
        "sample_test.shuffle_built_in_dual_input_uint2_uint16",
        "sample_test.shuffle_built_in_dual_input_uint4_uint2",
        "sample_test.shuffle_built_in_dual_input_uint4_uint4",
        "sample_test.shuffle_built_in_dual_input_uint4_uint8",
        "sample_test.shuffle_built_in_dual_input_uint4_uint16",
        "sample_test.shuffle_built_in_dual_input_uint8_uint2",
        "sample_test.shuffle_built_in_dual_input_uint8_uint4",
        "sample_test.shuffle_built_in_dual_input_uint8_uint8",
        "sample_test.shuffle_built_in_dual_input_uint8_uint16",
        "sample_test.shuffle_built_in_dual_input_uint16_uint2",
        "sample_test.shuffle_built_in_dual_input_uint16_uint4",
        "sample_test.shuffle_built_in_dual_input_uint16_uint8",
        "sample_test.shuffle_built_in_dual_input_uint16_uint16",
        "sample_test.shuffle_built_in_dual_input_long2_long2",
        "sample_test.shuffle_built_in_dual_input_long2_long4",
        "sample_test.shuffle_built_in_dual_input_long2_long8",
        "sample_test.shuffle_built_in_dual_input_long2_long16",
        "sample_test.shuffle_built_in_dual_input_long4_long2",
        "sample_test.shuffle_built_in_dual_input_long4_long4",
        "sample_test.shuffle_built_in_dual_input_long4_long8",
        "sample_test.shuffle_built_in_dual_input_long4_long16",
        "sample_test.shuffle_built_in_dual_input_long8_long2",
        "sample_test.shuffle_built_in_dual_input_long8_long4",
        "sample_test.shuffle_built_in_dual_input_long8_long8",
        "sample_test.shuffle_built_in_dual_input_long8_long16",
        "sample_test.shuffle_built_in_dual_input_long16_long2",
        "sample_test.shuffle_built_in_dual_input_long16_long4",
        "sample_test.shuffle_built_in_dual_input_long16_long8",
        "sample_test.shuffle_built_in_dual_input_long16_long16",
        "sample_test.shuffle_built_in_dual_input_ulong2_ulong2",
        "sample_test.shuffle_built_in_dual_input_ulong2_ulong4",
        "sample_test.shuffle_built_in_dual_input_ulong2_ulong8",
        "sample_test.shuffle_built_in_dual_input_ulong2_ulong16",
        "sample_test.shuffle_built_in_dual_input_ulong4_ulong2",
        "sample_test.shuffle_built_in_dual_input_ulong4_ulong4",
        "sample_test.shuffle_built_in_dual_input_ulong4_ulong8",
        "sample_test.shuffle_built_in_dual_input_ulong4_ulong16",
        "sample_test.shuffle_built_in_dual_input_ulong8_ulong2",
        "sample_test.shuffle_built_in_dual_input_ulong8_ulong4",
        "sample_test.shuffle_built_in_dual_input_ulong8_ulong8",
        "sample_test.shuffle_built_in_dual_input_ulong8_ulong16",
        "sample_test.shuffle_built_in_dual_input_ulong16_ulong2",
        "sample_test.shuffle_built_in_dual_input_ulong16_ulong4",
        "sample_test.shuffle_built_in_dual_input_ulong16_ulong8",
        "sample_test.shuffle_built_in_dual_input_ulong16_ulong16",
        "sample_test.shuffle_built_in_dual_input_float2_float2",
        "sample_test.shuffle_built_in_dual_input_float2_float4",
        "sample_test.shuffle_built_in_dual_input_float2_float8",
        "sample_test.shuffle_built_in_dual_input_float2_float16",
        "sample_test.shuffle_built_in_dual_input_float4_float2",
        "sample_test.shuffle_built_in_dual_input_float4_float4",
        "sample_test.shuffle_built_in_dual_input_float4_float8",
        "sample_test.shuffle_built_in_dual_input_float4_float16",
        "sample_test.shuffle_built_in_dual_input_float8_float2",
        "sample_test.shuffle_built_in_dual_input_float8_float4",
        "sample_test.shuffle_built_in_dual_input_float8_float8",
        "sample_test.shuffle_built_in_dual_input_float8_float16",
        "sample_test.shuffle_built_in_dual_input_float16_float2",
        "sample_test.shuffle_built_in_dual_input_float16_float4",
        "sample_test.shuffle_built_in_dual_input_float16_float8",
        "sample_test.shuffle_built_in_dual_input_float16_float16",
    };

    log_info("test_relationals\n");
    return test_suite(device, size_t_width, folder, test_name, sizeof(test_name) / sizeof(const char *), "");
}


bool test_relationals_double (cl_device_id device, cl_uint size_t_width, const char *folder)
{
    static const char* test_name[] = {
        "sample_test.relational_bitselect_double",
        "sample_test.relational_bitselect_double2",
        "sample_test.relational_bitselect_double3",
        "sample_test.relational_bitselect_double4",
        "sample_test.relational_bitselect_double8",
        "sample_test.relational_bitselect_double16",
        "sample_test.relational_isequal_double",
        "sample_test.relational_isequal_double2",
        "sample_test.relational_isequal_double3",
        "sample_test.relational_isequal_double4",
        "sample_test.relational_isequal_double8",
        "sample_test.relational_isequal_double16",
        "sample_test.relational_isnotequal_double",
        "sample_test.relational_isnotequal_double2",
        "sample_test.relational_isnotequal_double3",
        "sample_test.relational_isnotequal_double4",
        "sample_test.relational_isnotequal_double8",
        "sample_test.relational_isnotequal_double16",
        "sample_test.relational_isgreater_double",
        "sample_test.relational_isgreater_double2",
        "sample_test.relational_isgreater_double3",
        "sample_test.relational_isgreater_double4",
        "sample_test.relational_isgreater_double8",
        "sample_test.relational_isgreater_double16",
        "sample_test.relational_isgreaterequal_double",
        "sample_test.relational_isgreaterequal_double2",
        "sample_test.relational_isgreaterequal_double3",
        "sample_test.relational_isgreaterequal_double4",
        "sample_test.relational_isgreaterequal_double8",
        "sample_test.relational_isgreaterequal_double16",
        "sample_test.relational_isless_double",
        "sample_test.relational_isless_double2",
        "sample_test.relational_isless_double3",
        "sample_test.relational_isless_double4",
        "sample_test.relational_isless_double8",
        "sample_test.relational_isless_double16",
        "sample_test.relational_islessequal_double",
        "sample_test.relational_islessequal_double2",
        "sample_test.relational_islessequal_double3",
        "sample_test.relational_islessequal_double4",
        "sample_test.relational_islessequal_double8",
        "sample_test.relational_islessequal_double16",
        "sample_test.relational_islessgreater_double",
        "sample_test.relational_islessgreater_double2",
        "sample_test.relational_islessgreater_double3",
        "sample_test.relational_islessgreater_double4",
        "sample_test.relational_islessgreater_double8",
        "sample_test.relational_islessgreater_double16",
        "sample_test.shuffle_built_in_double2_double2",
        "sample_test.shuffle_built_in_double2_double4",
        "sample_test.shuffle_built_in_double2_double8",
        "sample_test.shuffle_built_in_double2_double16",
        "sample_test.shuffle_built_in_double4_double2",
        "sample_test.shuffle_built_in_double4_double4",
        "sample_test.shuffle_built_in_double4_double8",
        "sample_test.shuffle_built_in_double4_double16",
        "sample_test.shuffle_built_in_double8_double2",
        "sample_test.shuffle_built_in_double8_double4",
        "sample_test.shuffle_built_in_double8_double8",
        "sample_test.shuffle_built_in_double8_double16",
        "sample_test.shuffle_built_in_double16_double2",
        "sample_test.shuffle_built_in_double16_double4",
        "sample_test.shuffle_built_in_double16_double8",
        "sample_test.shuffle_built_in_double16_double16",
        "sample_test.shuffle_built_in_dual_input_double2_double2",
        "sample_test.shuffle_built_in_dual_input_double2_double4",
        "sample_test.shuffle_built_in_dual_input_double2_double8",
        "sample_test.shuffle_built_in_dual_input_double2_double16",
        "sample_test.shuffle_built_in_dual_input_double4_double2",
        "sample_test.shuffle_built_in_dual_input_double4_double4",
        "sample_test.shuffle_built_in_dual_input_double4_double8",
        "sample_test.shuffle_built_in_dual_input_double4_double16",
        "sample_test.shuffle_built_in_dual_input_double8_double2",
        "sample_test.shuffle_built_in_dual_input_double8_double4",
        "sample_test.shuffle_built_in_dual_input_double8_double8",
        "sample_test.shuffle_built_in_dual_input_double8_double16",
        "sample_test.shuffle_built_in_dual_input_double16_double2",
        "sample_test.shuffle_built_in_dual_input_double16_double4",
        "sample_test.shuffle_built_in_dual_input_double16_double8",
        "sample_test.shuffle_built_in_dual_input_double16_double16",
    };

    log_info("test_relationals_double\n");
    return test_suite(device, size_t_width, folder, test_name, sizeof(test_name) / sizeof(const char *), "cl_khr_fp64");
}


bool test_select (cl_device_id device, cl_uint size_t_width, const char *folder)
{
    static const char* test_name[] = {
        "select_uchar_uchar",
        "select_uchar2_uchar2",
        "select_uchar3_uchar3",
        "select_uchar4_uchar4",
        "select_uchar8_uchar8",
        "select_uchar16_uchar16",
        "select_uchar_char",
        "select_uchar2_char2",
        "select_uchar3_char3",
        "select_uchar4_char4",
        "select_uchar8_char8",
        "select_uchar16_char16",
        "select_char_uchar",
        "select_char2_uchar2",
        "select_char3_uchar3",
        "select_char4_uchar4",
        "select_char8_uchar8",
        "select_char16_uchar16",
        "select_char_char",
        "select_char2_char2",
        "select_char3_char3",
        "select_char4_char4",
        "select_char8_char8",
        "select_char16_char16",
        "select_ushort_ushort",
        "select_ushort2_ushort2",
        "select_ushort3_ushort3",
        "select_ushort4_ushort4",
        "select_ushort8_ushort8",
        "select_ushort16_ushort16",
        "select_ushort_short",
        "select_ushort2_short2",
        "select_ushort3_short3",
        "select_ushort4_short4",
        "select_ushort8_short8",
        "select_ushort16_short16",
        "select_short_ushort",
        "select_short2_ushort2",
        "select_short3_ushort3",
        "select_short4_ushort4",
        "select_short8_ushort8",
        "select_short16_ushort16",
        "select_short_short",
        "select_short2_short2",
        "select_short3_short3",
        "select_short4_short4",
        "select_short8_short8",
        "select_short16_short16",
        "select_uint_uint",
        "select_uint2_uint2",
        "select_uint3_uint3",
        "select_uint4_uint4",
        "select_uint8_uint8",
        "select_uint16_uint16",
        "select_uint_int",
        "select_uint2_int2",
        "select_uint3_int3",
        "select_uint4_int4",
        "select_uint8_int8",
        "select_uint16_int16",
        "select_int_uint",
        "select_int2_uint2",
        "select_int3_uint3",
        "select_int4_uint4",
        "select_int8_uint8",
        "select_int16_uint16",
        "select_int_int",
        "select_int2_int2",
        "select_int3_int3",
        "select_int4_int4",
        "select_int8_int8",
        "select_int16_int16",
        "select_float_uint",
        "select_float2_uint2",
        "select_float3_uint3",
        "select_float4_uint4",
        "select_float8_uint8",
        "select_float16_uint16",
        "select_float_int",
        "select_float2_int2",
        "select_float3_int3",
        "select_float4_int4",
        "select_float8_int8",
        "select_float16_int16",
        "select_ulong_ulong",
        "select_ulong2_ulong2",
        "select_ulong3_ulong3",
        "select_ulong4_ulong4",
        "select_ulong8_ulong8",
        "select_ulong16_ulong16",
        "select_ulong_long",
        "select_ulong2_long2",
        "select_ulong3_long3",
        "select_ulong4_long4",
        "select_ulong8_long8",
        "select_ulong16_long16",
        "select_long_ulong",
        "select_long2_ulong2",
        "select_long3_ulong3",
        "select_long4_ulong4",
        "select_long8_ulong8",
        "select_long16_ulong16",
        "select_long_long",
        "select_long2_long2",
        "select_long3_long3",
        "select_long4_long4",
        "select_long8_long8",
        "select_long16_long16",
    };

    log_info("test_select\n");
    return test_suite(device, size_t_width, folder, test_name, sizeof(test_name) / sizeof(const char *), "");
}


bool test_select_double (cl_device_id device, cl_uint size_t_width, const char *folder)
{
    static const char* test_name[] = {
        "select_double_ulong",
        "select_double2_ulong2",
        "select_double3_ulong3",
        "select_double4_ulong4",
        "select_double8_ulong8",
        "select_double16_ulong16",
        "select_double_long",
        "select_double2_long2",
        "select_double3_long3",
        "select_double4_long4",
        "select_double8_long8",
        "select_double16_long16",
    };

    log_info("test_select_double\n");
    return test_suite(device, size_t_width, folder, test_name, sizeof(test_name) / sizeof(const char *), "cl_khr_fp64");
}

bool test_vec_align (cl_device_id device, cl_uint size_t_width, const char *folder)
{
    static const char* test_name[] = {
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_char2",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_char3",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_char4",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_char8",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_char16",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_uchar2",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_uchar3",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_uchar4",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_uchar8",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_uchar16",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_short2",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_short3",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_short4",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_short8",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_short16",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_ushort2",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_ushort3",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_ushort4",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_ushort8",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_ushort16",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_int2",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_int3",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_int4",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_int8",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_int16",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_uint2",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_uint3",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_uint4",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_uint8",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_uint16",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_long2",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_long3",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_long4",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_long8",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_long16",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_ulong2",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_ulong3",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_ulong4",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_ulong8",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_ulong16",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_float2",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_float3",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_float4",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_float8",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_float16",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_char",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_charp",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_ucharp",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_shortp",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_ushortp",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_intp",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_uintp",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_longp",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_ulongp",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_floatp",
    };

    log_info("vec_align\n");
    return test_suite(device, size_t_width, folder, test_name, sizeof(test_name) / sizeof(const char *), "");
}


bool test_vec_align_double (cl_device_id device, cl_uint size_t_width, const char *folder)
{
    static const char* test_name[] = {
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_double2",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_double3",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_double4",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_double8",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_double16",
        "test_vec_align_packed_struct_arr.vec_align_packed_struct_arr_doublep",
    };

    log_info("vec_align_double\n");
    return test_suite(device, size_t_width, folder, test_name, sizeof(test_name) / sizeof(const char *), "cl_khr_fp64");
}


bool test_vec_step (cl_device_id device, cl_uint size_t_width, const char *folder)
{
    static const char* test_name[] = {
        "test_step_var.step_var_char",
        "test_step_var.step_var_char2",
        "test_step_var.step_var_char3",
        "test_step_var.step_var_char4",
        "test_step_var.step_var_char8",
        "test_step_var.step_var_char16",
        "test_step_var.step_var_uchar",
        "test_step_var.step_var_uchar2",
        "test_step_var.step_var_uchar3",
        "test_step_var.step_var_uchar4",
        "test_step_var.step_var_uchar8",
        "test_step_var.step_var_uchar16",
        "test_step_var.step_var_short",
        "test_step_var.step_var_short2",
        "test_step_var.step_var_short3",
        "test_step_var.step_var_short4",
        "test_step_var.step_var_short8",
        "test_step_var.step_var_short16",
        "test_step_var.step_var_ushort",
        "test_step_var.step_var_ushort2",
        "test_step_var.step_var_ushort3",
        "test_step_var.step_var_ushort4",
        "test_step_var.step_var_ushort8",
        "test_step_var.step_var_ushort16",
        "test_step_var.step_var_int",
        "test_step_var.step_var_int2",
        "test_step_var.step_var_int3",
        "test_step_var.step_var_int4",
        "test_step_var.step_var_int8",
        "test_step_var.step_var_int16",
        "test_step_var.step_var_uint",
        "test_step_var.step_var_uint2",
        "test_step_var.step_var_uint3",
        "test_step_var.step_var_uint4",
        "test_step_var.step_var_uint8",
        "test_step_var.step_var_uint16",
        "test_step_var.step_var_long",
        "test_step_var.step_var_long2",
        "test_step_var.step_var_long3",
        "test_step_var.step_var_long4",
        "test_step_var.step_var_long8",
        "test_step_var.step_var_long16",
        "test_step_var.step_var_ulong",
        "test_step_var.step_var_ulong2",
        "test_step_var.step_var_ulong3",
        "test_step_var.step_var_ulong4",
        "test_step_var.step_var_ulong8",
        "test_step_var.step_var_ulong16",
        "test_step_var.step_var_float",
        "test_step_var.step_var_float2",
        "test_step_var.step_var_float3",
        "test_step_var.step_var_float4",
        "test_step_var.step_var_float8",
        "test_step_var.step_var_float16",
    };

    log_info("vec_step\n");
    return test_suite(device, size_t_width, folder, test_name, sizeof(test_name) / sizeof(const char *), "");
}

bool test_vec_step_double (cl_device_id device, cl_uint size_t_width, const char *folder)
{
    static const char* test_name[] = {
        "test_step_var.step_var_double",
        "test_step_var.step_var_double2",
        "test_step_var.step_var_double3",
        "test_step_var.step_var_double4",
        "test_step_var.step_var_double8",
        "test_step_var.step_var_double16",
    };

    log_info("vec_step_double\n");
    return test_suite(device, size_t_width, folder, test_name, sizeof(test_name) / sizeof(const char *), "cl_khr_fp64");
}

template<typename T>
void getT(const TestResult& res, unsigned arg, T& out)
{
    out = *(T*)(res.kernelArgs().getArg(arg)->getBuffer());
}

class LinkageTestService {
    std::vector<const char*> m_moduleNames;
    const char* m_kernelName;
    int m_expectedResult;
    const char *m_name;

public:
    LinkageTestService(const char **moduleNames, int numModules,
                       const char *kernelName) :
    m_moduleNames(numModules),
    m_kernelName(kernelName),
    m_expectedResult(-1),
    m_name(NULL) {
        std::copy(moduleNames, moduleNames+numModules, m_moduleNames.begin());
    }

    void setExpectedResult(int expectedRes) {
        m_expectedResult = expectedRes;
    }

    bool compareResult(cl_device_id dev, cl_uint width) {
        clContextWrapper context;
        clCommandQueueWrapper queue;
        size_t num_modules = m_moduleNames.size();
        std::vector<cl_program> programs(num_modules);
        create_context_and_queue(dev, &context, &queue);

        for (size_t i=0; i<num_modules; i++)
        {
            std::string filepath;
            get_bc_file_path("compile_and_link", m_moduleNames[i], filepath, width);
            programs[i] = create_program_from_bc(context, filepath);
        }
        // Linking to the modules together.
        LinkTask linkTask(&programs[0], num_modules, context, dev);
        if (!linkTask.execute()) {
            std::cerr << "Failed due to the following link error: "
                      << linkTask.getErrorLog() << std::endl;
            return false;
        }

        // Running the Kernel.
        cl_program exec = linkTask.getExecutable();
        clKernelWrapper kernel = create_kernel_helper(exec, m_kernelName);
        TestResult res;
        WorkSizeInfo ws;
        generate_kernel_data(context, kernel, ws, res);
        run_kernel(kernel, queue, ws, res);

        // Checking the result.
        res.readToHost(queue);
        int actual_value;
        getT(res, 0, actual_value);
        return (m_expectedResult == actual_value);
    }

    void setName(const char* name)
    {
        m_name = name;
    }

    const char* getName()const
    {
        return m_name;
    }
};

bool test_compile_and_link (cl_device_id device, cl_uint width, const char *folder)
{
    try_extract(folder);
    std::cout << "Running tests:" << std::endl;

    // Each array represents a testcast in compile and link. The first element
    // is the name of the 'main' module, as the second is the module being
    // linked.
    const char* private_files[]  = {"private_link", "private"};
    const char* internal_files[] = {"internal_linkage", "internal_linkage.mod"};
    const char* external_files[] = {"external_linkage", "external_linkage.mod"};
    const char* available_externally_files[] = {"available_externally", "global"};

    std::vector<LinkageTestService*> linkageTests;
    linkageTests.push_back(new LinkageTestService(private_files, 2, "k"));
    linkageTests.push_back(new LinkageTestService(internal_files, 2, "internal_linkage"));
    linkageTests.push_back(new LinkageTestService(external_files, 2, "external_linkage"));
    linkageTests.push_back(new LinkageTestService(available_externally_files, 2, "k"));
    // Set tests Names.
    linkageTests[0]->setName("private_linkage");
    linkageTests[1]->setName("internal_linkage");
    linkageTests[2]->setName("external_linkage");
    linkageTests[3]->setName("available_externally");
    // Set expected results.
    linkageTests[0]->setExpectedResult(std::string("spir_conformance").size());
    linkageTests[1]->setExpectedResult(1);
    linkageTests[2]->setExpectedResult(42);
    linkageTests[3]->setExpectedResult(42);

    unsigned int tests_passed = 0;
    CounterEventHandler SuccE(tests_passed, linkageTests.size());
    std::list<std::string> ErrList;

    for (size_t i=0; i<linkageTests.size(); i++)
    {
        AccumulatorEventHandler FailE(ErrList, linkageTests[i]->getName());
        std::cout << linkageTests[i]->getName() << "..." << std::endl;
        if(linkageTests[i]->compareResult(device, width))
        {
            (SuccE)(linkageTests[i]->getName(), "");
            std::cout << linkageTests[i]->getName() << " passed." << std::endl;
        }
        else
        {

            (FailE)(linkageTests[i]->getName(), "");
            std::cout << linkageTests[i]->getName() << " FAILED" << std::endl;
        }
    }

    std::cout << std::endl;
    std::cout << "PASSED " << tests_passed << " of " << SuccE.TN << " tests.\n" << std::endl;
    // Deallocating.
    std::for_each(linkageTests.begin(), linkageTests.end(), dealloc<LinkageTestService>);
    return tests_passed == SuccE.TN;
}

static bool test_sampler_enumeration(cl_device_id device, cl_uint width, const char *folder)
{
  static const char* test_name[] = {
      "sampler_NormF_AddrC_FilterL",
      "sampler_NormF_AddrC_FilterN",
      "sampler_NormF_AddrE_FilterL",
      "sampler_NormF_AddrE_FilterN",
      // "sampler_NormF_AddrM_FilterL" - Invalid combination
      // "sampler_NormF_AddrM_FilterN" - Invalid combination
      "sampler_NormF_AddrN_FilterL",
      "sampler_NormF_AddrN_FilterN",
      // "sampler_NormF_AddrR_FilterL" - Invalid combination
      // "sampler_NormF_AddrR_FilterN" - Invalid combination
      "sampler_NormT_AddrC_FilterL",
      "sampler_NormT_AddrC_FilterN",
      "sampler_NormT_AddrE_FilterL",
      "sampler_NormT_AddrE_FilterN",
      "sampler_NormT_AddrM_FilterL",
      "sampler_NormT_AddrM_FilterN",
      "sampler_NormT_AddrN_FilterL",
      "sampler_NormT_AddrN_FilterN",
      "sampler_NormT_AddrR_FilterL",
      "sampler_NormT_AddrR_FilterN"
  };

  log_info("test_sampler_enum_values\n");
  return test_suite(device, width, folder, test_name, sizeof(test_name) / sizeof(const char *), "");
}

const char* HOSTVAL_SAMPLER    = "hostval_sampler";
const char* HOSTVAL_IMAGE_DESC = "hostval_image_desc";
const char* HOSTVAL_IMAGE_DESC_3D = "hostval_image_desc_3d";

static bool test_image_enumeration(cl_context context, cl_command_queue queue,
                                   cl_program prog, cl_device_id device,
                                   CounterEventHandler &SuccE, std::list<std::string> &ErrList)
{
    // Creating image descriptor value generator.
    ImageValuesGenerator imgVals;
    bool success = true;

    for(ImageValuesGenerator::iterator it = imgVals.begin(), e = imgVals.end(); it != e; ++it)
    {
        bool currentSuccess = true;
        AccumulatorEventHandler FailE(ErrList, it.toString());

        std::string kernelName(HOSTVAL_IMAGE_DESC);
        kernelName.append("_");
        kernelName.append(it.getImageTypeName());

        if (it.getImageTypeName() == "image3d")
        {
            // If the type is a 3D image we continue to the next one
            continue;
        }

        // Saving the original image generator, for later restoration.
        std::string baseGenName = it.getBaseImageGeneratorName();
        KernelArgInfo baseInfo;
        baseInfo.setTypeName(baseGenName.c_str());
        DataGenerator *pDataGen = DataGenerator::getInstance();
        KernelArgGenerator* pOrig = pDataGen->getArgGenerator(baseInfo);

        try
        {
            // Creating the kernel for this specific enumeration.
            WorkSizeInfo ws;
            clKernelWrapper kernel = create_kernel_helper(prog, kernelName);

            // Acquiring a reference to the image generator we need for this image
            // type.
            KernelArgInfo typedInfo;
            const std::string tyName = it.getImageGeneratorName();
            typedInfo.setTypeName(tyName.c_str());
            KernelArgGeneratorImage* pImgGen = (KernelArgGeneratorImage*)pDataGen->getArgGenerator(typedInfo);

            // If the channel order is not valid for the current image type, we
            // continue to the next one.
            if (!pImgGen->isValidChannelOrder(context, it.getOpenCLChannelOrder()))
                continue;

            // Due to unknown number of types at the beggining count them on the fly
            SuccE.TN++;

            // Configuring the image generator so it will produce the correct image
            // descriptor.
            pImgGen->setChannelOrder(it.getOpenCLChannelOrder());
            pDataGen->setArgGenerator(baseInfo, pImgGen);

            // Generate the arguments and run the kernel.
            TestResult res;
            generate_kernel_data(context, kernel, ws, res);
            run_kernel(kernel, queue, ws, res);

            // Informing the result.
            std::cout << "enum_" << it.toString() << "..." << std::endl;
            int actualOrder = 0, actualTy = 0;
            getT<int>(res, 1U, actualOrder), getT<int>(res, 2U, actualTy);
            if (actualOrder != it.getSPIRChannelOrder())
            {
                std::cout << " expected channel order: " << it.getSPIRChannelOrder()
                          << " but received " << actualOrder << "." << std::endl;
                success = currentSuccess = false;
            }

            if (actualTy != it.getDataType())
            {
                std::cout << " expected data type: " << it.getDataType()
                          << " but received " << actualTy << "." << std::endl;
                success = currentSuccess = false;
            }

            if (currentSuccess)
            {
                (SuccE)(it.toString(), kernelName);
                std::cout << "enum_" << it.toString() << " passed." << std::endl;
            }
            else
            {
                (FailE)(it.toString(), kernelName);
                std::cout << "enum_" << it.toString() << " FAILED" << std::endl;
            }
        } catch(std::exception e)
        {
            (FailE)(it.toString(), kernelName);
            print_error(1, e.what());
            success = currentSuccess = false;
        }

        // Restore the base image generator to its original value.
        pDataGen->setArgGenerator(baseInfo, pOrig);
    }

    return success;
}

static bool test_image_enumeration_3d(cl_context context, cl_command_queue queue,
                                   cl_program prog, cl_device_id device,
                                   CounterEventHandler &SuccE, std::list<std::string> &ErrList)
{
    // Creating image descriptor value generator.
    ImageValuesGenerator imgVals;
    bool success = true;

    for(ImageValuesGenerator::iterator it = imgVals.begin(), e = imgVals.end(); it != e; ++it)
    {
        bool currentSuccess = true;
        AccumulatorEventHandler FailE(ErrList, it.toString());

        std::string kernelName(HOSTVAL_IMAGE_DESC);
        kernelName.append("_");
        kernelName.append(it.getImageTypeName());

        if (it.getImageTypeName() != "image3d")
        {
            // If the type is not a 3D image we continue to the next one
            continue;
        }

        // Saving the original image generator, for later restoration.
        std::string baseGenName = it.getBaseImageGeneratorName();
        KernelArgInfo baseInfo;
        baseInfo.setTypeName(baseGenName.c_str());
        DataGenerator *pDataGen = DataGenerator::getInstance();
        KernelArgGenerator* pOrig = pDataGen->getArgGenerator(baseInfo);

        try
        {
            // Creating the kernel for this specific enumeration.
            WorkSizeInfo ws;
            clKernelWrapper kernel = create_kernel_helper(prog, kernelName);

            // Acquiring a reference to the image generator we need for this image
            // type.
            KernelArgInfo typedInfo;
            const std::string tyName = it.getImageGeneratorName();
            typedInfo.setTypeName(tyName.c_str());
            KernelArgGeneratorImage* pImgGen = (KernelArgGeneratorImage*)pDataGen->getArgGenerator(typedInfo);

            // If the channel order is not valid for the current image type, we
            // continue to the next one.
            if (!pImgGen->isValidChannelOrder(context, it.getOpenCLChannelOrder()))
                continue;

            // Due to unknown number of types at the beggining count them on the fly
            SuccE.TN++;

            // Configuring the image generator so it will produce the correct image
            // descriptor.
            pImgGen->setChannelOrder(it.getOpenCLChannelOrder());
            pDataGen->setArgGenerator(baseInfo, pImgGen);

            // Generate the arguments and run the kernel.
            TestResult res;
            generate_kernel_data(context, kernel, ws, res);
            run_kernel(kernel, queue, ws, res);

            // Informing the result.
            std::cout << "enum_" << it.toString() << "..." << std::endl;
            int actualOrder = 0, actualTy = 0;
            getT<int>(res, 1U, actualOrder), getT<int>(res, 2U, actualTy);
            if (actualOrder != it.getSPIRChannelOrder())
            {
                std::cout << " expected channel order: " << it.getSPIRChannelOrder()
                          << " but received " << actualOrder << "." << std::endl;
                success = currentSuccess = false;
            }

            if (actualTy != it.getDataType())
            {
                std::cout << " expected data type: " << it.getDataType()
                          << " but received " << actualTy << "." << std::endl;
                success = currentSuccess = false;
            }

            if (currentSuccess)
            {
                (SuccE)(it.toString(), kernelName);
                std::cout << "enum_" << it.toString() << " passed." << std::endl;
            }
            else
            {
                (FailE)(it.toString(), kernelName);
                std::cout << "enum_" << it.toString() << " FAILED" << std::endl;
            }
        } catch(std::exception e)
        {
            (FailE)(it.toString(), kernelName);
            print_error(1, e.what());
            success = currentSuccess = false;
        }

        // Restore the base image generator to its original value.
        pDataGen->setArgGenerator(baseInfo, pOrig);
    }

    return success;
}

static bool test_enum_values(cl_device_id device, cl_uint width, const char *folder)
{
    try_extract(folder);
    std::cout << "Running tests:" << std::endl;
    bool success = true;
    typedef bool (*EnumTest)(cl_context, cl_command_queue, cl_program, cl_device_id, CounterEventHandler &SuccE, std::list<std::string> &ErrList);
    EnumTest test_functions[] = { test_image_enumeration, test_image_enumeration_3d };
    const char *enum_tests[] = { HOSTVAL_IMAGE_DESC, HOSTVAL_IMAGE_DESC_3D };
    const size_t TEST_NUM = sizeof(enum_tests)/sizeof(char*);

    unsigned int tests_passed = 0;
    CounterEventHandler SuccE(tests_passed, 0);
    std::list<std::string> ErrList;

    // Composing the name of the CSV file.
    char* dir = get_exe_dir();
    std::string csvName(dir);
    csvName.append(dir_sep());
    csvName.append("khr.csv");
    free(dir);

    // Figure out whether the test can run on the device. If not, we skip it.
    const KhrSupport& khrDb = *KhrSupport::get(csvName);

    for (size_t i=0; i<TEST_NUM; i++)
    {
        const char *cur_test = enum_tests[i];
        cl_bool images = khrDb.isImagesRequired(folder, cur_test);
        cl_bool images3D = khrDb.isImages3DRequired(folder, cur_test);
        if(images == CL_TRUE && checkForImageSupport(device) != 0)
        {
            std::cout << cur_test << " Skipped. (Cannot run on device due to Images is not supported)." << std::endl;
            continue;
        }

        if(images3D == CL_TRUE && checkFor3DImageSupport(device) != 0)
        {
            std::cout << cur_test << " Skipped. (Cannot run on device as 3D images are not supported)." << std::endl;
            continue;
        }

        std::string bc_file_path;
        get_bc_file_path(folder, cur_test, bc_file_path, width);
        clContextWrapper context;
        clCommandQueueWrapper queue;
        create_context_and_queue(device, &context, &queue);
        clProgramWrapper bcprog = create_program_from_bc(context, bc_file_path);

        // Build the kernel.
        SpirBuildTask build_task(bcprog, device, "-x spir -spir-std=1.2 -cl-kernel-arg-info");
        if (!build_task.execute())
        {
            std::cerr << "Cannot run enum_values suite due to the "
                      << "following build error: "
                      << build_task.getErrorLog()
                      << std::endl;
            return false;
        }

        success &= test_functions[i](context, queue, bcprog, device, SuccE, ErrList);
    }

    std::cout << std::endl;
    std::cout << "PASSED " << tests_passed << " of " << SuccE.TN << " tests.\n" << std::endl;

    if (!ErrList.empty())
    {
        std::cout << "Failed tests:" << std::endl;
        std::for_each(ErrList.begin(), ErrList.end(), printError);
    }
    std::cout << std::endl;
    return success;
}

std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems)
{
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}


static bool
test_kernel_attributes(cl_device_id device, cl_uint width, const char *folder)
{
    try_extract(folder);
    std::cout << "Running tests:" << std::endl;
    bool success = true;
    clContextWrapper context;
    std::string bc_file_path;
    clCommandQueueWrapper queue;
    clKernelWrapper kernel;
    char attributes[256] = {0};
    size_t i, res_size = 0;

    unsigned int tests_passed = 0;
    CounterEventHandler SuccE(tests_passed, 1);
    std::list<std::string> ErrList;
    std::string test_name("kernel_attributes");

    log_info("kernel_attributes...\n");
    AccumulatorEventHandler FailE(ErrList, test_name);

    try
    {
        create_context_and_queue(device, &context, &queue);
        get_bc_file_path(folder, "kernel_attributes", bc_file_path, width);
        clProgramWrapper bcprog = create_program_from_bc(context, bc_file_path);

        // Building the program, so we could create the kernel.
        SpirBuildTask build_task(bcprog, device, "-x spir -spir-std=1.2 -cl-kernel-arg-info");
        if (!build_task.execute())
        {
            std::cerr << "Cannot run kernel_attributes suite due to the following build error: "
                      << build_task.getErrorLog()
                      << std::endl;
            throw std::exception();
        }

        // Querying the kernel for its attributes.
        kernel = create_kernel_helper(bcprog, "test");
        cl_int err_code = clGetKernelInfo(kernel, CL_KERNEL_ATTRIBUTES, sizeof(attributes), attributes, &res_size);
        if (err_code != CL_SUCCESS)
        {
            std::cerr << "clGetKernelInfo unable retrieve kernel attributes (error code: " << err_code << " )\n";
            throw std::exception();
        }

        // Building the expected attributes vector.
        std::vector<std::string> expected;
        expected.push_back(std::string("work_group_size_hint(64,1,1)"));
        expected.push_back(std::string("vec_type_hint(float4)"));

        std::vector<std::string> actual;
        split(attributes, ' ', actual);

        for(i = 0; i < expected.size(); ++i)
        {
            if(std::find(actual.begin(), actual.end(), expected[i]) == actual.end())
            {
                // Attribute not found
                std::cout << "Extracted from kernel: " << attributes << std::endl;
                std::cerr << "expected " << expected[i] << " attribute not found" << std::endl;
                throw std::exception();
            }
        }
        (SuccE)(test_name, "");
        log_info("kernel_attributes passed.\n");
    } catch(std::exception e)
    {
        (FailE)(test_name, "");
        log_info("kernel_attributes FAILED\n");
        success = false;
    }

    std::cout << std::endl;
    std::cout << "PASSED " << tests_passed << " of " << 1 << " tests.\n" << std::endl;

    if (!ErrList.empty())
    {
        std::cout << "Failed tests:" << std::endl;
        std::for_each(ErrList.begin(), ErrList.end(), printError);
    }
    std::cout << std::endl;
    return success;
}

static bool test_binary_type(cl_device_id device, cl_uint width, const char *folder)
{
    std::string bc_file_path;
    clContextWrapper context;
    clCommandQueueWrapper queue;

    // Extract the suite if needed.
    try_extract(folder);
    std::cout << "Running tests:" << std::endl;
    bool success = true;
    unsigned int tests_passed = 0;
    CounterEventHandler SuccE(tests_passed, 1);
    std::list<std::string> ErrList;
    std::string test_name("binary_type");

    log_info("binary_type...\n");
    AccumulatorEventHandler FailE(ErrList, test_name);

    try
    {
        // Creating the program object.
        get_bc_file_path(folder, "simple", bc_file_path, width);
        create_context_and_queue(device, &context, &queue);
        clProgramWrapper clprog = create_program_from_bc(context, bc_file_path);

        // Checking the attribute matches the requierment in Section 9.15.2 of the
        // extensions SPEC.
        cl_int binary_type = 0;
        size_t ret_size = 0;
        if (cl_int err_code = clGetProgramBuildInfo(clprog, device, CL_PROGRAM_BINARY_TYPE, sizeof(cl_int), &binary_type, &ret_size))
        {
            std::cerr << "Cannot run test_binary_type suite due to the "
                      << "following build error: "
                      << err_code << std::endl;
            throw std::exception();
        }

        assert(ret_size == sizeof(cl_int) && "Return size doesn't match.");
        if (binary_type != CL_PROGRAM_BINARY_TYPE_INTERMEDIATE)
        {
            std::cerr << "binary type is " << binary_type
                      << " as opposed to " << CL_PROGRAM_BINARY_TYPE_INTERMEDIATE
                      << " which is the expected value." << std::endl;
            throw std::exception();
        }
        (SuccE)(test_name, "");
        log_info("binary_type passed.\n");
    } catch(std::exception e)
    {
        (FailE)(test_name, "");
        log_info("binary_type FAILED\n");
        success = false;
    }


    std::cout << std::endl;
    std::cout << "PASSED " << tests_passed << " of " << 1 << " tests.\n" << std::endl;

    if (!ErrList.empty())
    {
        std::cout << "Failed tests:" << std::endl;
        std::for_each(ErrList.begin(), ErrList.end(), printError);
    }
    std::cout << std::endl;
    return success;
}

struct sub_suite
{
    const char *name;
    const char *folder;
    const testfn test_function;
};

static const sub_suite spir_suites[] = {
    { "api", "api", test_api },
    { "api_double", "api", test_api_double },
    { "atomics", "atomics", test_atomics },
    { "basic", "basic", test_basic },
    { "basic_double", "basic", test_basic_double },
    { "commonfns", "commonfns", test_commonfns },
    { "commonfns_double", "commonfns", test_commonfns_double },
    { "conversions", "conversions", test_conversions },
    { "conversions_double", "conversions", test_conversions_double },
    { "geometrics", "geometrics", test_geometrics },
    { "geometrics_double", "geometrics", test_geometrics_double },
    { "half", "half", test_half },
    { "half_double", "half", test_half_double },
    { "kernel_image_methods", "kernel_image_methods",
      test_kernel_image_methods },
    { "images_kernel_read_write", "images_kernel_read_write",
      test_images_kernel_read_write },
    { "images_samplerlessRead", "images_samplerlessRead",
      test_images_samplerless_read },
    { "integer_ops", "integer_ops", test_integer_ops },
    { "math_brute_force", "math_brute_force", test_math_brute_force },
    { "math_brute_force_double", "math_brute_force",
      test_math_brute_force_double },
    { "printf", "printf", test_printf },
    { "profiling", "profiling", test_profiling },
    { "relationals", "relationals", test_relationals },
    { "relationals_double", "relationals", test_relationals_double },
    { "select", "select", test_select },
    { "select_double", "select", test_select_double },
    { "vec_align", "vec_align", test_vec_align },
    { "vec_align_double", "vec_align", test_vec_align_double },
    { "vec_step", "vec_step", test_vec_step },
    { "vec_step_double", "vec_step", test_vec_step_double },
    { "compile_and_link", "compile_and_link", test_compile_and_link },
    { "sampler_enumeration", "sampler_enumeration", test_sampler_enumeration },
    { "enum_values", "enum_values", test_enum_values },
    // {"kernel_attributes",           "kernel_attributes",
    // test_kernel_attributes}, // disabling temporarily, see GitHub #1284
    { "binary_type", "binary_type", test_binary_type },
};


/**
Utility function using to find a specific sub-suite name in the SPIR tests.
Called in case the user asked for running a specific sub-suite or specific tests.
 */
static int find_suite_name (std::string suite_name)
{
    for (unsigned int i = 0; i < sizeof(spir_suites) / sizeof(sub_suite); ++i)
    {
        if (0 == suite_name.compare(spir_suites[i].name))
        {
            return i;
        }
    }
    return -1;
}


/**
Look for the first device from the first platform .
 */
cl_device_id get_platform_device (cl_device_type device_type, cl_uint choosen_device_index, cl_uint choosen_platform_index)
{
    int error = CL_SUCCESS;
    cl_uint num_platforms = 0;
    cl_platform_id *platforms;
    cl_uint num_devices = 0;
    cl_device_id *devices = NULL;

    /* Get the platform */
    error = clGetPlatformIDs(0, NULL, &num_platforms);
    if ( error != CL_SUCCESS )
    {
        throw std::runtime_error("clGetPlatformIDs failed: " + std::string(IGetErrorString(error)));
    }
    if ( choosen_platform_index >= num_platforms )
    {
        throw std::runtime_error("platform index out of range");
    }

    platforms = (cl_platform_id *) malloc( num_platforms * sizeof( cl_platform_id ) );
    if ( !platforms )
    {
        throw std::runtime_error("platform malloc failed");
    }
    BufferOwningPtr<cl_platform_id> platformsBuf(platforms);

    error = clGetPlatformIDs(num_platforms, platforms, NULL);
    if ( error != CL_SUCCESS )
    {
        throw std::runtime_error("clGetPlatformIDs failed: " + std::string(IGetErrorString(error)));
    }

    /* Get the number of requested devices */
    error = clGetDeviceIDs(platforms[choosen_platform_index],  device_type, 0, NULL, &num_devices );
    if ( error != CL_SUCCESS )
    {
        throw std::runtime_error("clGetDeviceIDs failed: " + std::string(IGetErrorString(error)));
    }
    if ( choosen_device_index >= num_devices )
    {
        throw std::runtime_error("device index out of rangen");
    }

    devices = (cl_device_id *) malloc( num_devices * sizeof( cl_device_id ) );
    if ( !devices )
    {
        throw std::runtime_error("device malloc failed");
    }
    BufferOwningPtr<cl_device_id> devicesBuf(devices);

    /* Get the requested device */
    error = clGetDeviceIDs(platforms[choosen_platform_index],  device_type, num_devices, devices, NULL );
    if ( error != CL_SUCCESS )
    {
        throw std::runtime_error("clGetDeviceIDs failed: " + std::string(IGetErrorString(error)));
    }

    return devices[choosen_device_index];
}


/**
 Parses the command line parameters and set the
 appropriate global variables accordingly
 The valid options are:
    a) none - run all SPIR tests
    b) one argument (tests-suite name) - run one SPIR tests-suite
    c) two arguments (tests-suite name and test name) - run one SPIR test
 */
static int ParseCommandLine (int argc, const char *argv[],
    std::string& suite_name, std::string& test_name, cl_device_type *device_type, cl_uint *device_index, cl_uint *platform_index, cl_uint *size_t_width)
{
    int based_on_env_var = 0;

    /* Check for environment variable to set device type */
    char *env_mode = getenv( "CL_DEVICE_TYPE" );
    if( env_mode != NULL )
    {
        based_on_env_var = 1;
        if( strcmp( env_mode, "gpu" ) == 0 || strcmp( env_mode, "CL_DEVICE_TYPE_GPU" ) == 0 )
            *device_type = CL_DEVICE_TYPE_GPU;
        else if( strcmp( env_mode, "cpu" ) == 0 || strcmp( env_mode, "CL_DEVICE_TYPE_CPU" ) == 0 )
            *device_type = CL_DEVICE_TYPE_CPU;
        else if( strcmp( env_mode, "accelerator" ) == 0 || strcmp( env_mode, "CL_DEVICE_TYPE_ACCELERATOR" ) == 0 )
            *device_type = CL_DEVICE_TYPE_ACCELERATOR;
        else if( strcmp( env_mode, "default" ) == 0 || strcmp( env_mode, "CL_DEVICE_TYPE_DEFAULT" ) == 0 )
            *device_type = CL_DEVICE_TYPE_DEFAULT;
        else
        {
            throw Exceptions::CmdLineError( "Unknown CL_DEVICE_TYPE env variable setting\n");
        }
    }

    env_mode = getenv( "CL_DEVICE_INDEX" );
    if( env_mode != NULL )
    {
        *device_index = atoi(env_mode);
    }

    env_mode = getenv( "CL_PLATFORM_INDEX" );
    if( env_mode != NULL )
    {
        *platform_index = atoi(env_mode);
    }

        /* Process the command line arguments */

    /* Special case: just list the tests */
    if( ( argc > 1 ) && (!strcmp( argv[ 1 ], "-list" ) || !strcmp( argv[ 1 ], "-h" ) || !strcmp( argv[ 1 ], "--help" )))
    {
        log_info( "Usage: %s [<suite name>] [pid<num>] [id<num>] [<device type>] [w32] [no-unzip]\n", argv[0] );
        log_info( "\t<suite name>\tOne or more of: (default all)\n");
        log_info( "\tpid<num>\t\tIndicates platform at index <num> should be used (default 0).\n" );
        log_info( "\tid<num>\t\tIndicates device at index <num> should be used (default 0).\n" );
        log_info( "\t<device_type>\tcpu|gpu|accelerator|<CL_DEVICE_TYPE_*> (default CL_DEVICE_TYPE_DEFAULT)\n" );
        log_info( "\tw32\t\tIndicates device address bits is 32.\n" );
        log_info( "\tno-unzip\t\tDo not extract test files from Zip; use existing.\n" );

        for( unsigned int i = 0; i < (sizeof(spir_suites) / sizeof(sub_suite)); i++ )
        {
            log_info( "\t\t%s\n", spir_suites[i].name );
        }
        return 0;
    }

    /* Do we have a CPU/GPU specification? */
    while( argc > 1 )
    {
        if( strcmp( argv[ argc - 1 ], "gpu" ) == 0 || strcmp( argv[ argc - 1 ], "CL_DEVICE_TYPE_GPU" ) == 0 )
        {
            *device_type = CL_DEVICE_TYPE_GPU;
            argc--;
        }
        else if( strcmp( argv[ argc - 1 ], "cpu" ) == 0 || strcmp( argv[ argc - 1 ], "CL_DEVICE_TYPE_CPU" ) == 0 )
        {
            *device_type = CL_DEVICE_TYPE_CPU;
            argc--;
        }
        else if( strcmp( argv[ argc - 1 ], "accelerator" ) == 0 || strcmp( argv[ argc - 1 ], "CL_DEVICE_TYPE_ACCELERATOR" ) == 0 )
        {
            *device_type = CL_DEVICE_TYPE_ACCELERATOR;
            argc--;
        }
        else if( strcmp( argv[ argc - 1 ], "CL_DEVICE_TYPE_DEFAULT" ) == 0 )
        {
            *device_type = CL_DEVICE_TYPE_DEFAULT;
            argc--;
        }
        else if( strcmp( argv[ argc - 1 ], "w32" ) == 0 )
        {
            *size_t_width = 32;
            argc--;
        }
        else if( strcmp( argv[ argc - 1 ], "no-unzip" ) == 0 )
        {
            no_unzip = 1;
            argc--;
        }
        else break;
    }

    /* Did we choose a specific device index? */
    if( argc > 1 )
    {
        if( strlen( argv[ argc - 1 ] ) >= 3 && argv[ argc - 1 ][0] == 'i' && argv[ argc - 1 ][1] == 'd' )
        {
            *device_index = atoi( &(argv[ argc - 1 ][2]) );
            argc--;
        }
    }

    /* Did we choose a specific platform index? */
    if( argc > 1 )
    {
        if( strlen( argv[ argc - 1 ] ) >= 3 && argv[ argc - 1 ][0] == 'p' && argv[ argc - 1 ][1] == 'i' && argv[ argc - 1 ][2] == 'd')
        {
            *platform_index = atoi( &(argv[ argc - 1 ][3]) );
            argc--;
        }
    }

    switch( *device_type )
    {
        case CL_DEVICE_TYPE_GPU:
            log_info( "Requesting GPU device " );
            break;
        case CL_DEVICE_TYPE_CPU:
            log_info( "Requesting CPU device " );
            break;
        case CL_DEVICE_TYPE_ACCELERATOR:
            log_info( "Requesting Accelerator device " );
            break;
        case CL_DEVICE_TYPE_DEFAULT:
            log_info( "Requesting Default device " );
            break;
        default:
            throw Exceptions::CmdLineError( "Requesting unknown device ");
            break;
    }
    log_info( based_on_env_var ? "based on environment variable " : "based on command line " );
    log_info( "for platform index %d and device index %d\n", *platform_index, *device_index);

    if (argc > 3)
    {
        throw Exceptions::CmdLineError("Command line error. Unrecognized token\n");
    }
    else {
        if (argc > 1)
        {
            suite_name.assign(argv[1]);
        }
        if (argc == 3)
        {
            test_name.assign(argv[2]);
        }
    }

    return 1;
}

struct WLMsg: EventHandler
{
    const char* Msg;

    WLMsg(const char* M): Msg(M){}

    void operator()(const std::string& T, const std::string& K)
    {
        std::cout << "Test " << T << " Kernel " << K << "\t" << Msg << std::endl;
    }
};


int main (int argc, const char* argv[])
{
    std::string test_suite_name;                       // name of the selected tests-suite (NULL for all)
    std::string test_file_name;                        // name of the .selected test (NULL for all)
    cl_device_type device_type = CL_DEVICE_TYPE_DEFAULT;
    cl_uint choosen_device_index = 0;
    cl_uint choosen_platform_index = 0;
    cl_uint size_t_width = 0;                            // device address bits (32 or 64).
    cl_int err;
    int failed = 0;
    int ntests = 0;
    custom_cout atf_info;
    custom_cerr atf_error;
    override_buff atf_cout(std::cout, atf_info);
    override_buff atf_err(std::cerr, atf_error);

    WLMsg Success("\t\tPassed"), Failure("\t\tFailure");
    try
    {
        if (ParseCommandLine(argc, argv, test_suite_name, test_file_name, &device_type, &choosen_device_index, &choosen_platform_index, &size_t_width) == 0)
            return 0;

        cl_device_id device = get_platform_device(device_type, choosen_device_index, choosen_platform_index);
        printDeviceHeader(device);

        std::vector<Version> versions;
        get_spir_version(device, versions);

        if (!is_extension_available(device, "cl_khr_spir")
            || (std::find(versions.begin(), versions.end(), Version{ 1, 2 })
                == versions.end()))
        {
            log_info("Spir extension version 1.2 is not supported by the device\n");
            return 0;
        }

        // size_t_width <> 0 - device address bits is forced by command line argument
        if ((0 == size_t_width) && ((err = clGetDeviceInfo(device, CL_DEVICE_ADDRESS_BITS, sizeof(cl_uint), &size_t_width, NULL))))
        {
            print_error( err, "Unable to obtain device address bits" );
            return -1;
        }

        if (! test_suite_name.empty())
        {
            // command line is not empty - do not run all the tests
            int tsn = find_suite_name(test_suite_name);
            ntests = 1;
            if (tsn < 0)
            {
                throw Exceptions::CmdLineError("Command line error. Error in SPIR sub-suite name\n");
            }
            else if (test_file_name.empty())
            {
                if (!spir_suites[tsn].test_function(device, size_t_width, spir_suites[tsn].folder))
                    failed++;
            }
            else
            {
                OclExtensions devExt = OclExtensions::getDeviceCapabilities(device);
                TestRunner runner(&Success, &Failure, devExt);
                std::string folder = getTestFolder(test_suite_name.c_str());
                try_extract(folder.c_str());
                if (!runner.runBuildTest(device, folder.c_str(), test_file_name.c_str(), size_t_width))
                    failed++;
            }
        }
        else
        {
            // Run all the tests
            ntests = (sizeof(spir_suites) / sizeof(spir_suites[0]));
            for (unsigned int i = 0; i < ntests; ++i)
            {
                if (!spir_suites[i].test_function(device, size_t_width, spir_suites[i].folder))
                    failed++;
            }
        }
        if (failed)
            std::cout << "FAILED " << failed << " of " << ntests << " test suites.\n" << std::endl;
        else
            std::cout << "PASSED " << ntests << " of " << ntests << " test suites.\n" << std::endl;
        return failed;
    }
    catch(const Exceptions::CmdLineError& e)
    {
        print_error(1, e.what());
        return 1;
    }
    catch(const std::runtime_error& e)
    {
        print_error(2, e.what());
        return 2;
    }
    catch(const std::exception& e)
    {
        print_error(3, e.what());
        return 3;
    }
}

