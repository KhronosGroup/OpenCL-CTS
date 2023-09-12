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
#ifndef TESTNONUNIFORMWORKGROUP_H
#define TESTNONUNIFORMWORKGROUP_H

#include "procs.h"
#include <vector>
#include "tools.h"
#include <algorithm>

#define MAX_SIZE_OF_ALLOCATED_MEMORY (400*1024*1024)

#define NUMBER_OF_REGIONS 8

#define MAX_DIMS 3

// This structure reflects data received from kernel.
typedef struct _DataContainerAttrib
{
  cl_ulong get_global_size[MAX_DIMS];
  cl_ulong get_global_offset[MAX_DIMS];
  cl_ulong get_local_size[MAX_DIMS];
  cl_ulong get_enqueued_local_size[MAX_DIMS];
  cl_ulong get_global_id[MAX_DIMS];
  cl_ulong get_local_id[MAX_DIMS];
  cl_ulong get_group_id[MAX_DIMS];
  cl_ulong get_num_groups[MAX_DIMS];
  cl_ulong get_work_dim;
  cl_ushort test_local_barrier_result_bool;
  cl_ushort test_global_barrier_result_bool;
  cl_ushort test_local_atomic_result_value;
}DataContainerAttrib;

// Describes range of testing.
namespace Range {
  enum RangeEnum {
    BASIC = (1 << 0),
    BARRIERS = (1 << 1),
    ATOMICS = (1 << 2),

    ALL = Range::BASIC | Range::BARRIERS | Range::ATOMICS
  };
}

std::string showArray (const size_t *arr, cl_uint dims);

// Main class responsible for testing
class TestNonUniformWorkGroup {
public:
    TestNonUniformWorkGroup(const cl_device_id &device,
                            const cl_context &context,
                            const cl_command_queue &queue, const cl_uint dims,
                            size_t *globalSize, const size_t *localSize,
                            const size_t *buffersSize,
                            const size_t *globalWorkOffset,
                            const size_t *reqdWorkGroupSize = NULL);

    ~TestNonUniformWorkGroup();

    static size_t getMaxLocalWorkgroupSize(const cl_device_id &device);
    static void setMaxLocalWorkgroupSize(size_t workGroupSize)
    {
        TestNonUniformWorkGroup::_maxLocalWorkgroupSize = workGroupSize;
    }
  static void enableStrictMode (bool state);

  void setTestRange (int range) {_testRange = range;}
  int prepareDevice ();
  int verifyResults ();
  int runKernel ();

private:
  size_t _globalSize[MAX_DIMS];
  size_t _localSize[MAX_DIMS];
  size_t _globalWorkOffset[MAX_DIMS];
  bool _globalWorkOffset_IsNull;
  size_t _enqueuedLocalSize[MAX_DIMS];
  bool _localSize_IsNull;
  size_t _reqdWorkGroupSize[MAX_DIMS];
  static size_t _maxLocalWorkgroupSize;
  size_t _maxWorkItemSizes[MAX_DIMS];
  size_t _numOfGlobalWorkItems; // in global work group
  const cl_device_id _device;
  const cl_context _context;
  const cl_command_queue _queue;
  const cl_uint _dims;

  int _testRange;

  std::vector<DataContainerAttrib> _resultsRegionArray;
  std::vector<DataContainerAttrib> _referenceRegionArray;
  cl_uint _globalAtomicTestValue;

  clProgramWrapper _program;
  clKernelWrapper _testKernel;

  Error::ErrorClass _err;

  TestNonUniformWorkGroup ();

  static bool _strictMode;
  void setLocalWorkgroupSize (const size_t *globalSize, const size_t *localSize);
  void setGlobalWorkgroupSize (const size_t *globalSize);
  void verifyData (DataContainerAttrib * reference, DataContainerAttrib * results, short regionNumber);
  void calculateExpectedValues ();
  void showTestInfo ();
  size_t adjustLocalArraySize(size_t localArraySize);
  size_t adjustGlobalBufferSize(size_t globalBufferSize);
};

// Class responsible for running subtest scenarios in test function
class SubTestExecutor {
public:
  SubTestExecutor(const cl_device_id &device, const cl_context &context, const cl_command_queue &queue)
    : _device (device), _context (context), _queue (queue), _failCounter (0), _overallCounter (0) {}

  void runTestNonUniformWorkGroup(const cl_uint dims, size_t *globalSize,
                                  const size_t *localSize, int range);

  void runTestNonUniformWorkGroup(const cl_uint dims, size_t *globalSize,
                                  const size_t *localSize,
                                  const size_t *globalWorkOffset,
                                  const size_t *reqdWorkGroupSize, int range);

  int calculateWorkGroupSize(size_t &maxWgSize, int testRange);
  int status();

private:
  SubTestExecutor();
  const cl_device_id _device;
  const cl_context _context;
  const cl_command_queue _queue;
  unsigned int _failCounter;
  unsigned int _overallCounter;
};

#endif // TESTNONUNIFORMWORKGROUP_H
