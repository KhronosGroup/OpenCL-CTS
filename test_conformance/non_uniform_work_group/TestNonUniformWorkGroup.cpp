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
#include "TestNonUniformWorkGroup.h"
#include <vector>
#include <sstream>
#define NL "\n"

size_t TestNonUniformWorkGroup::_maxLocalWorkgroupSize = 0;
bool TestNonUniformWorkGroup::_strictMode = false;

// Main Kernel source code
static const char *KERNEL_FUNCTION =
  NL "#define MAX_DIMS 3"
  NL "typedef struct _DataContainerAttrib"
  NL "{"
  NL "    unsigned long get_global_size[MAX_DIMS];"
  NL "    unsigned long get_global_offset[MAX_DIMS];"
  NL "    unsigned long get_local_size[MAX_DIMS];"
  NL "    unsigned long get_enqueued_local_size[MAX_DIMS];"
  NL "    unsigned long get_global_id[MAX_DIMS];"
  NL "    unsigned long get_local_id[MAX_DIMS];"
  NL "    unsigned long get_group_id[MAX_DIMS];"
  NL "    unsigned long get_num_groups[MAX_DIMS];"
  NL "    unsigned long get_work_dim;"
  NL "    unsigned short test_local_barrier_result_bool;"
  NL "    unsigned short test_global_barrier_result_bool;"
  NL "    unsigned short test_local_atomic_result_value;"
  NL "}DataContainerAttrib;"

  NL "enum Error{"
  NL "  ERR_GLOBAL_SIZE=0,"
  NL "  ERR_GLOBAL_WORK_OFFSET,"
  NL "  ERR_LOCAL_SIZE,"
  NL "  ERR_GLOBAL_ID,"
  NL "  ERR_LOCAL_ID,"
  NL "  ERR_ENQUEUED_LOCAL_SIZE,"
  NL "  ERR_NUM_GROUPS,"
  NL "  ERR_GROUP_ID,"
  NL "  ERR_WORK_DIM,"
  NL "  ERR_GLOBAL_BARRIER,"
  NL "  ERR_LOCAL_BARRIER,"
  NL "  ERR_GLOBAL_ATOMIC,"
  NL "  ERR_LOCAL_ATOMIC,"
  NL "  ERR_STRICT_MODE,"
  NL "  ERR_BUILD_STATUS,"
  NL "  ERR_UNKNOWN,"
  NL "  ERR_DIFFERENT,"
  NL "  _LAST_ELEM"
  NL "};"

  NL "uint getGlobalIndex (uint gid2, uint gid1, uint gid0) {"
  NL "    return gid2*get_global_size(0)*get_global_size(1) + gid1*get_global_size(0) + gid0;"
  NL "}"

  NL "int getRegionIndex () {"
  NL "    uint gid0 = get_global_id(0) - get_global_offset(0);"
  NL "    uint gid1 = get_global_id(1) - get_global_offset(1);"
  NL "    uint gid2 = get_global_id(2) - get_global_offset(2);"
  NL "    if (gid0 == 0 && gid1 == 0 && gid2 == 0) {"
  NL "      return 0;"
  NL "    } else if (gid0 == get_global_size(0) - 1 && gid1 == 0 && gid2 == 0) {"
  NL "      return 1;"
  NL "    } else if (gid0 == 0 && gid1 == get_global_size(1) - 1 && gid2 == 0) {"
  NL "      return 2;"
  NL "    } else if (gid0 == get_global_size(0) - 1 && gid1 == get_global_size(1) - 1 && gid2 == 0) {"
  NL "      return 3;"
  NL "    } else if (gid0 == 0 && gid1 == 0 && gid2 == get_global_size(2) - 1) {"
  NL "      return 4;"
  NL "    } else if (gid0 == get_global_size(0) - 1 && gid1 == 0 && gid2 == get_global_size(2) - 1) {"
  NL "      return 5;"
  NL "    } else if (gid0 == 0 && gid1 == get_global_size(1) - 1 && gid2 == get_global_size(2) - 1) {"
  NL "      return 6;"
  NL "    } else if (gid0 == get_global_size(0) - 1 && gid1 == get_global_size(1) - 1 && gid2 == get_global_size(2) - 1) {"
  NL "      return 7;"
  NL "    }"
  NL "    return -1;"
  NL "}"

  NL "void getLocalSize(__global DataContainerAttrib *results) {"
  NL "  for (unsigned short i = 0; i < MAX_DIMS; i++) {"
  NL "    results->get_local_size[i] = get_local_size(i);"
  NL "  }"
  NL "}"

  NL "#ifdef TESTBASIC"
  // values set by this function will be checked on the host side
  NL "void testBasicHost(__global DataContainerAttrib *results) {"
  NL "    for (unsigned short i = 0; i < MAX_DIMS; i++) {"
  NL "      results->get_global_size[i] = get_global_size(i);"
  NL "      results->get_global_offset[i] = get_global_offset(i);"
  NL "      results->get_enqueued_local_size[i] = get_enqueued_local_size(i);"
  NL "      results->get_global_id[i] = get_global_id(i);"
  NL "      results->get_local_id[i] = get_local_id(i);"
  NL "      results->get_group_id[i] = get_group_id(i);"
  NL "      results->get_num_groups[i] = get_num_groups(i);"
  NL "    }"
  NL "    results->get_work_dim = get_work_dim();"
  NL "}"
  // values set by this function are checked on the kernel side
  NL "void testBasicKernel(__global unsigned int *errorCounterBuffer, __local DataContainerAttrib *resultsForThread0) {"
  NL "  uint lid0 = get_local_id(0);"
  NL "  uint lid1 = get_local_id(1);"
  NL "  uint lid2 = get_local_id(2);"
  NL "  if (lid0 == 0 && lid1 == 0 && lid2 == 0) {"
  NL "    for (unsigned short i = 0; i < MAX_DIMS; i++) {"
  NL "      resultsForThread0->get_global_size[i] = get_global_size(i);"
  NL "      resultsForThread0->get_global_offset[i] = get_global_offset(i);"
  NL "      resultsForThread0->get_enqueued_local_size[i] = get_enqueued_local_size(i);"
  NL "      resultsForThread0->get_group_id[i] = get_group_id(i);"
  NL "      resultsForThread0->get_num_groups[i] = get_num_groups(i);"
  NL "    }"
  NL "    resultsForThread0->get_work_dim = get_work_dim();"
  NL "  }"
  NL "    barrier(CLK_LOCAL_MEM_FENCE);"
  // verifies built in functions on the kernel side
  NL "  if (lid0 != 0 || lid1 != 0 || lid2 != 0) {"
  NL "    for (unsigned short i = 0; i < MAX_DIMS; i++) {"
  NL "      if (resultsForThread0->get_global_size[i] != get_global_size(i)) {"
  NL "        atomic_inc(&errorCounterBuffer[ERR_GLOBAL_SIZE]);"
  NL "      }"
  NL "      if (resultsForThread0->get_global_offset[i] != get_global_offset(i)) {"
  NL "        atomic_inc(&errorCounterBuffer[ERR_GLOBAL_WORK_OFFSET]);"
  NL "      }"
  NL "      if (resultsForThread0->get_enqueued_local_size[i] != get_enqueued_local_size(i)) {"
  NL "        atomic_inc(&errorCounterBuffer[ERR_ENQUEUED_LOCAL_SIZE]);"
  NL "      }"
  NL "      if (resultsForThread0->get_group_id[i] != get_group_id(i)) {"
  NL "        atomic_inc(&errorCounterBuffer[ERR_GROUP_ID]);"
  NL "      }"
  NL "      if (resultsForThread0->get_num_groups[i] != get_num_groups(i)) {"
  NL "        atomic_inc(&errorCounterBuffer[ERR_NUM_GROUPS]);"
  NL "      }"
  NL "    }"
  NL "    if (resultsForThread0->get_work_dim != get_work_dim()) {"
  NL "      atomic_inc(&errorCounterBuffer[ERR_WORK_DIM]);"
  NL "    }"
  NL "  }"
  NL "}"
  NL "#endif"

  NL "#ifdef TESTBARRIERS"
  NL "void testBarriers(__global unsigned int *errorCounterBuffer, __local unsigned int *testLocalBuffer, __global unsigned int *testGlobalBuffer) {"
  NL "    uint gid0 = get_global_id(0);"
  NL "    uint gid1 = get_global_id(1);"
  NL "    uint gid2 = get_global_id(2);"
  NL "    uint lid0 = get_local_id(0);"
  NL "    uint lid1 = get_local_id(1);"
  NL "    uint lid2 = get_local_id(2);"
  NL
  NL "    uint globalIndex = getGlobalIndex(gid2-get_global_offset(2), gid1-get_global_offset(1), gid0-get_global_offset(0));"
  NL "    uint localIndex = lid2*get_local_size(0)*get_local_size(1) + lid1*get_local_size(0) + lid0;"
  NL "    testLocalBuffer[localIndex] = 0;"
  NL "    testGlobalBuffer[globalIndex] = 0;"
  NL "    uint maxLocalIndex = get_local_size(0)*get_local_size(1)*get_local_size(2)-1;"
  NL "    uint nextLocalIndex = (localIndex>=maxLocalIndex)?0:(localIndex+1);"
  NL "    uint next_lid0 = (lid0+1>=get_local_size(0))?0:lid0+1;"
  NL "    uint next_lid1 = (lid1+1>=get_local_size(1))?0:lid1+1;"
  NL "    uint next_lid2 = (lid2+1>=get_local_size(2))?0:lid2+1;"
  NL "    uint nextGlobalIndexInLocalWorkGroup = getGlobalIndex (get_group_id(2)*get_enqueued_local_size(2)+next_lid2, get_group_id(1)*get_enqueued_local_size(1)+next_lid1, get_group_id(0)*get_enqueued_local_size(0)+next_lid0);"
  // testing local barriers
  NL "    testLocalBuffer[localIndex] = localIndex;"
  NL "    barrier(CLK_LOCAL_MEM_FENCE);"
  NL "    uint temp = testLocalBuffer[nextLocalIndex];"
  NL "    if (temp != nextLocalIndex) {"
  NL "      atomic_inc(&errorCounterBuffer[ERR_LOCAL_BARRIER]);"
  NL "    }"
  // testing global barriers
  NL "    testGlobalBuffer[globalIndex] = globalIndex;"
  NL "    barrier(CLK_GLOBAL_MEM_FENCE);"
  NL "    uint temp2 = testGlobalBuffer[nextGlobalIndexInLocalWorkGroup];"
  NL "    if (temp2 != nextGlobalIndexInLocalWorkGroup) {"
  NL "      atomic_inc(&errorCounterBuffer[ERR_GLOBAL_BARRIER]);"
  NL "    }"
  NL "}"
  NL "#endif"

  NL "#ifdef TESTATOMICS"
  NL "void testAtomics(__global unsigned int *globalAtomicTestVariable, __local unsigned int *localAtomicTestVariable) {"
  NL "    uint gid0 = get_global_id(0);"
  NL "    uint gid1 = get_global_id(1);"
  NL "    uint gid2 = get_global_id(2);"
  NL
  NL "    uint globalIndex = getGlobalIndex(gid2-get_global_offset(2), gid1-get_global_offset(1), gid0-get_global_offset(0));"
  // testing atomic function on local memory
  NL "    atomic_inc(localAtomicTestVariable);"
  NL "    barrier(CLK_LOCAL_MEM_FENCE);"
  // testing atomic function on global memory
  NL "    atomic_inc(globalAtomicTestVariable);"
  NL "}"
  NL "#endif"

  NL "#ifdef RWGSX"
  NL "#ifdef RWGSY"
  NL "#ifdef RWGSZ"
  NL "__attribute__((reqd_work_group_size(RWGSX, RWGSY, RWGSZ)))"
  NL "#endif"
  NL "#endif"
  NL "#endif"
  NL "__kernel void testKernel(__global DataContainerAttrib *results, __local unsigned int *testLocalBuffer,"
  NL "      __global unsigned int *testGlobalBuffer, __global unsigned int *globalAtomicTestVariable, __global unsigned int *errorCounterBuffer) {"
  NL "    uint gid0 = get_global_id(0);"
  NL "    uint gid1 = get_global_id(1);"
  NL "    uint gid2 = get_global_id(2);"
  NL
  NL "    uint globalIndex = getGlobalIndex(gid2-get_global_offset(2), gid1-get_global_offset(1), gid0-get_global_offset(0));"
  NL "    int regionIndex = getRegionIndex();"
  NL "    if (regionIndex >= 0) {"
  NL "      getLocalSize(&results[regionIndex]);"
  NL "    }"
  NL "#ifdef TESTBASIC"
  NL "    if (regionIndex >= 0) {"
  NL "      testBasicHost(&results[regionIndex]);"
  NL "    }"
  NL "    __local DataContainerAttrib resultsForThread0;"
  NL "    testBasicKernel(errorCounterBuffer, &resultsForThread0);"
  NL "#endif"
  NL "#ifdef TESTBARRIERS"
  NL "    testBarriers(errorCounterBuffer, testLocalBuffer, testGlobalBuffer);"
  NL "#endif"
  NL "#ifdef TESTATOMICS"
  NL "    __local unsigned int localAtomicTestVariable;"
  NL "    localAtomicTestVariable = 0;"
  NL "    barrier(CLK_LOCAL_MEM_FENCE);"
  NL "    testAtomics(globalAtomicTestVariable, &localAtomicTestVariable);"
  NL "    barrier(CLK_LOCAL_MEM_FENCE);"
  NL "    if (localAtomicTestVariable != get_local_size(0) * get_local_size(1) * get_local_size(2)) {"
  NL "      atomic_inc(&errorCounterBuffer[ERR_LOCAL_ATOMIC]);"
  NL "    }"
  NL "#endif"
  NL "}"
  NL ;

TestNonUniformWorkGroup::TestNonUniformWorkGroup(
    const cl_device_id &device, const cl_context &context,
    const cl_command_queue &queue, const cl_uint dims, size_t *globalSize,
    const size_t *localSize, const size_t *buffersSize,
    const size_t *globalWorkOffset, const size_t *reqdWorkGroupSize)
    : _device(device), _context(context), _queue(queue), _dims(dims)
{

    if (globalSize == NULL || dims < 1 || dims > 3)
    {
        // throw std::invalid_argument("globalSize is NULL value.");
        // This is method of informing that parameters are wrong.
        // It would be checked by prepareDevice() function.
        // This is used because of lack of exception support.
        _globalSize[0] = 0;
        return;
    }

    // For OpenCL-3.0 support for non-uniform workgroups is optional, it's still
    // useful to run these tests since we can verify the behavior of the
    // get_enqueued_local_size() builtin for uniform workgroups, so we round up
    // the global size to insure uniform workgroups on those 3.0 devices.
    // We only need to do this when localSize is non-null, otherwise the driver
    // will select a value for localSize which will be uniform on devices that
    // don't support non-uniform work-groups.
    if (nullptr != localSize && get_device_cl_version(device) >= Version(3, 0))
    {
        // Query for the non-uniform work-group support.
        cl_bool are_non_uniform_sub_groups_supported{ CL_FALSE };
        auto error =
            clGetDeviceInfo(device, CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT,
                            sizeof(are_non_uniform_sub_groups_supported),
                            &are_non_uniform_sub_groups_supported, nullptr);
        if (error)
        {
            print_error(error,
                        "clGetDeviceInfo failed for "
                        "CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT");
            // This signals an error to the caller (see above).
            _globalSize[0] = 0;
            return;
        }

        // If non-uniform work-groups are not supported round up the global
        // sizes so workgroups are uniform and we have at least one.
        if (CL_FALSE == are_non_uniform_sub_groups_supported)
        {
            log_info(
                "WARNING: Non-uniform work-groups are not supported on this "
                "device.\n Running test with uniform work-groups.\n");
            for (unsigned dim = 0; dim < dims; ++dim)
            {
                auto global_size_before = globalSize[dim];
                auto global_size_rounded = global_size_before
                    + (localSize[dim] - global_size_before % localSize[dim]);
                globalSize[dim] = global_size_rounded;
                log_info("Rounding globalSize[%d] = %zu -> %zu\n", dim,
                         global_size_before, global_size_rounded);
            }
        }
    }

    cl_uint i;
    _globalWorkOffset_IsNull = true;
    _localSize_IsNull = true;

    setGlobalWorkgroupSize(globalSize);
    setLocalWorkgroupSize(globalSize, localSize);
    for (i = _dims; i < MAX_DIMS; i++)
    {
        _globalSize[i] = 1;
    }

    for (i = 0; i < MAX_DIMS; i++)
    {
        _globalWorkOffset[i] = 0;
    }

    if (globalWorkOffset)
    {
        _globalWorkOffset_IsNull = false;
        for (i = 0; i < _dims; i++)
        {
            _globalWorkOffset[i] = globalWorkOffset[i];
        }
    }

    for (i = 0; i < MAX_DIMS; i++)
    {
        _enqueuedLocalSize[i] = 1;
    }

    if (localSize)
    {
        _localSize_IsNull = false;
        for (i = 0; i < _dims; i++)
        {
            _enqueuedLocalSize[i] = _localSize[i];
        }
    }

    if (reqdWorkGroupSize)
    {
        for (i = 0; i < _dims; i++)
        {
            _reqdWorkGroupSize[i] = reqdWorkGroupSize[i];
        }
        for (i = _dims; i < MAX_DIMS; i++)
        {
            _reqdWorkGroupSize[i] = 1;
        }
    }
    else
    {
        _reqdWorkGroupSize[0] = 0;
        _reqdWorkGroupSize[1] = 0;
        _reqdWorkGroupSize[2] = 0;
    }

    _testRange = Range::ALL;

    _numOfGlobalWorkItems = _globalSize[0] * _globalSize[1] * _globalSize[2];

    DataContainerAttrib temp = { { 0, 0, 0 } };

    // array with results from each region
    _resultsRegionArray.resize(NUMBER_OF_REGIONS, temp);
    _referenceRegionArray.resize(NUMBER_OF_REGIONS, temp);
}

TestNonUniformWorkGroup::~TestNonUniformWorkGroup () {
  if (_err.checkError()) {
    _err.showStats();
  }
}

void TestNonUniformWorkGroup::setLocalWorkgroupSize (const size_t *globalSize, const size_t *localSize)
{
   cl_uint i;
   // Enforce localSize should not exceed globalSize
   if (localSize) {
       for (i = 0; i < _dims; i++) {
           if ((globalSize[i] < localSize[i])) {
               _localSize[i] = globalSize[i];
           }else{
               _localSize[i] = localSize[i];
           }
      }
   }
}

void TestNonUniformWorkGroup::setGlobalWorkgroupSize (const size_t *globalSize)
{
   cl_uint i;
   for (i = 0; i < _dims; i++) {
       _globalSize[i] = globalSize[i];
   }
}

void TestNonUniformWorkGroup::verifyData (DataContainerAttrib * reference, DataContainerAttrib * results, short regionNumber) {

  std::ostringstream tmp;
  std::string errorLocation;

  if (_testRange & Range::BASIC) {
    for (unsigned short i = 0; i < MAX_DIMS; i++) {
      tmp.str("");
      tmp.clear();
      tmp << "region number: " << regionNumber << " for dim: " << i;
      errorLocation = tmp.str();

      if (results->get_global_size[i] != reference->get_global_size[i]) {
        _err.show(Error::ERR_GLOBAL_SIZE, errorLocation, results->get_global_size[i], reference->get_global_size[i]);
      }

      if (results->get_global_offset[i] != reference->get_global_offset[i]) {
        _err.show(Error::ERR_GLOBAL_WORK_OFFSET, errorLocation, results->get_global_offset[i], reference->get_global_offset[i]);
      }

      if (results->get_local_size[i] != reference->get_local_size[i] || results->get_local_size[i] > _maxWorkItemSizes[i]) {
        _err.show(Error::ERR_LOCAL_SIZE, errorLocation, results->get_local_size[i], reference->get_local_size[i]);
      }

      if (results->get_enqueued_local_size[i] != reference->get_enqueued_local_size[i] || results->get_enqueued_local_size[i] > _maxWorkItemSizes[i]) {
        _err.show(Error::ERR_ENQUEUED_LOCAL_SIZE, errorLocation, results->get_enqueued_local_size[i], reference->get_enqueued_local_size[i]);
      }

      if (results->get_num_groups[i] != reference->get_num_groups[i]) {
        _err.show(Error::ERR_NUM_GROUPS, errorLocation, results->get_num_groups[i], reference->get_num_groups[i]);
      }
    }
  }

  tmp.str("");
  tmp.clear();
  tmp << "region number: " << regionNumber;
  errorLocation = tmp.str();
  if (_testRange & Range::BASIC) {
    if (results->get_work_dim != reference->get_work_dim) {
      _err.show(Error::ERR_WORK_DIM, errorLocation, results->get_work_dim, reference->get_work_dim);
    }
  }
}

void TestNonUniformWorkGroup::calculateExpectedValues () {
  size_t numberOfPossibleRegions[MAX_DIMS];

  numberOfPossibleRegions[0] = (_globalSize[0]>1)?2:1;
  numberOfPossibleRegions[1] = (_globalSize[1]>1)?2:1;
  numberOfPossibleRegions[2] = (_globalSize[2]>1)?2:1;

  for (cl_ushort i = 0; i < NUMBER_OF_REGIONS; ++i) {

    if (i & 0x01 && numberOfPossibleRegions[0] == 1) {
      continue;
    }

    if (i & 0x02 && numberOfPossibleRegions[1] == 1) {
      continue;
    }

    if (i & 0x04 && numberOfPossibleRegions[2] == 1) {
      continue;
    }

    for (cl_ushort dim = 0; dim < MAX_DIMS; ++dim) {
      _referenceRegionArray[i].get_global_size[dim] = static_cast<unsigned long>(_globalSize[dim]);
      _referenceRegionArray[i].get_global_offset[dim] = static_cast<unsigned long>(_globalWorkOffset[dim]);
      _referenceRegionArray[i].get_enqueued_local_size[dim] = static_cast<unsigned long>(_enqueuedLocalSize[dim]);
      _referenceRegionArray[i].get_local_size[dim] = static_cast<unsigned long>(_enqueuedLocalSize[dim]);
      _referenceRegionArray[i].get_num_groups[dim] = static_cast<unsigned long>(ceil(static_cast<float>(_globalSize[dim]) / _enqueuedLocalSize[dim]));
    }
    _referenceRegionArray[i].get_work_dim = _dims;

    if (i & 0x01) {
      _referenceRegionArray[i].get_local_size[0] = static_cast<unsigned long>((_globalSize[0] - 1) % _enqueuedLocalSize[0] + 1);
    }

    if (i & 0x02) {
      _referenceRegionArray[i].get_local_size[1] = static_cast<unsigned long>((_globalSize[1] - 1) % _enqueuedLocalSize[1] + 1);
    }

    if (i & 0x04) {
      _referenceRegionArray[i].get_local_size[2] = static_cast<unsigned long>((_globalSize[2] - 1) % _enqueuedLocalSize[2] + 1);
    }
  }
}

size_t TestNonUniformWorkGroup::getMaxLocalWorkgroupSize (const cl_device_id &device) {
  int err;

  if (TestNonUniformWorkGroup::_maxLocalWorkgroupSize == 0) {
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
      sizeof(TestNonUniformWorkGroup::_maxLocalWorkgroupSize), &TestNonUniformWorkGroup::_maxLocalWorkgroupSize, NULL);
    if (err)
    {
        log_error("clGetDeviceInfo failed\n");
        return 0;
    }
  }

  return TestNonUniformWorkGroup::_maxLocalWorkgroupSize;
}

void TestNonUniformWorkGroup::enableStrictMode(bool state) {
  TestNonUniformWorkGroup::_strictMode = state;
}

int TestNonUniformWorkGroup::prepareDevice () {
  int err;
  cl_uint device_max_dimensions;
  cl_uint i;

  if (_globalSize[0] == 0)
  {
    log_error("Some arguments passed into constructor were wrong.\n");
    return -1;
  }

  err = clGetDeviceInfo(_device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
    sizeof(device_max_dimensions), &device_max_dimensions, NULL);
  test_error(err, "clGetDeviceInfo failed");

  err = clGetDeviceInfo(_device, CL_DEVICE_MAX_WORK_ITEM_SIZES,
    sizeof(_maxWorkItemSizes), _maxWorkItemSizes, NULL);

  test_error(err, "clGetDeviceInfo failed");

  // Trim the local size to the limitations of what the device supports in each dimension.
  for (i = 0; i < _dims; i++) {
    if(_enqueuedLocalSize[i] > _maxWorkItemSizes[i]) {
      _enqueuedLocalSize[i] = _maxWorkItemSizes[i];
    }
  }

  if(_localSize_IsNull == false)
    calculateExpectedValues();

  std::string buildOptions{};
  if(_reqdWorkGroupSize[0] != 0 && _reqdWorkGroupSize[1] != 0 && _reqdWorkGroupSize[2] != 0) {
    std::ostringstream tmp(" ");
    tmp << " -D RWGSX=" << _reqdWorkGroupSize[0]
      << " -D RWGSY=" << _reqdWorkGroupSize[1]
      << " -D RWGSZ=" << _reqdWorkGroupSize[2] << " ";
      buildOptions += tmp.str();
  }

  if (_testRange & Range::BASIC)
    buildOptions += " -D TESTBASIC";
  if (_testRange & Range::ATOMICS)
    buildOptions += " -D TESTATOMICS";
  if (_testRange & Range::BARRIERS)
    buildOptions += " -D TESTBARRIERS";

  err = create_single_kernel_helper_with_build_options (_context, &_program, &_testKernel, 1,
    &KERNEL_FUNCTION, "testKernel", buildOptions.c_str());
  if (err)
  {
    log_error("Error %d in line: %d of file %s\n", err, __LINE__, __FILE__);
    return -1;
  }

  return 0;
}

int TestNonUniformWorkGroup::verifyResults () {
  if (_localSize_IsNull) {
    // for global work groups where local work group size is not defined (set to NULL in clEnqueueNDRangeKernel)
    // we need to check what optimal size was chosen by device
    // we assumed that local size value for work item 0 is right for the rest work items
    _enqueuedLocalSize[0] = static_cast<size_t>(_resultsRegionArray[0].get_local_size[0]);
    _enqueuedLocalSize[1] = static_cast<size_t>(_resultsRegionArray[0].get_local_size[1]);
    _enqueuedLocalSize[2] = static_cast<size_t>(_resultsRegionArray[0].get_local_size[2]);
    calculateExpectedValues();

    // strict mode verification
    if(_strictMode) {
      size_t localWorkGroupSize = _enqueuedLocalSize[0]*_enqueuedLocalSize[1]*_enqueuedLocalSize[2];
      if (localWorkGroupSize != TestNonUniformWorkGroup::getMaxLocalWorkgroupSize(_device))
          _err.show(Error::ERR_STRICT_MODE, "",localWorkGroupSize, TestNonUniformWorkGroup::getMaxLocalWorkgroupSize(_device));
    }

    log_info ("Local work group size calculated by driver: %s\n", showArray(_enqueuedLocalSize, _dims).c_str());
 }

  for (cl_ushort i = 0; i < NUMBER_OF_REGIONS; ++i) {
    verifyData(&_referenceRegionArray[i], &_resultsRegionArray[i], i);
  }

  if (_testRange & Range::ATOMICS) {
    if (_globalAtomicTestValue != _numOfGlobalWorkItems) {
      _err.show(Error::ERR_GLOBAL_ATOMIC);
    }
  }

  if (_err.checkError())
    return -1;

  return 0;
}

std::string showArray (const size_t *arr, cl_uint dims) {
  std::ostringstream tmpStringStream ("");

  tmpStringStream << "{";
  for (cl_uint i=0; i < dims; i++) {
    tmpStringStream << arr[i];
    if (i+1 < dims)
      tmpStringStream << ", ";
  }
  tmpStringStream << "}";

  return tmpStringStream.str();
}

void TestNonUniformWorkGroup::showTestInfo () {
  std::string tmpString;
  log_info ("T E S T  P A R A M E T E R S :\n");
  log_info ("\tNumber of dimensions:\t%d\n", _dims);

  tmpString = showArray(_globalSize, _dims);

  log_info("\tGlobal work group size:\t%s\n", tmpString.c_str());

  if (!_localSize_IsNull) {
    tmpString = showArray(_enqueuedLocalSize, _dims);
  } else {
    tmpString = "NULL";
  }
  log_info("\tLocal work group size:\t%s\n", tmpString.c_str());

  if (!_globalWorkOffset_IsNull) {
    tmpString = showArray(_globalWorkOffset, _dims);
  } else {
    tmpString = "NULL";
  }
  log_info("\tGlobal work group offset:\t%s\n", tmpString.c_str());

  if (_reqdWorkGroupSize[0] != 0 && _reqdWorkGroupSize[1] != 0 && _reqdWorkGroupSize[2] != 0) {
    tmpString = showArray(_reqdWorkGroupSize, _dims);
  } else {
    tmpString = "attribute disabled";
  }
  log_info ("\treqd_work_group_size attribute:\t%s\n", tmpString.c_str());

  tmpString = "";
  if(_testRange & Range::BASIC)
     tmpString += "basic";
  if(_testRange & Range::ATOMICS) {
    if(tmpString != "") tmpString += ", ";
    tmpString += "atomics";
  }
  if(_testRange & Range::BARRIERS) {
    if(tmpString != "") tmpString += ", ";
    tmpString += "barriers";
  }
  log_info ("\tTest range:\t%s\n", tmpString.c_str());
  if(_strictMode) {
    log_info ("\tStrict mode:\tON\n");
    if (!_localSize_IsNull) {
      log_info ("\tATTENTION: strict mode applies only NULL local work group size\n");
    } else {
        log_info("\t\tExpected value of local work group size is %zu.\n",
                 TestNonUniformWorkGroup::getMaxLocalWorkgroupSize(_device));
    }

  }
}

size_t TestNonUniformWorkGroup::adjustLocalArraySize (size_t localArraySize) {
  // In case if localArraySize is too big, sometimes we can not run kernel because of lack
  // of resources due to kernel itself requires some local memory to run
  int err;

  cl_ulong kernelLocalMemSize = 0;
  err = clGetKernelWorkGroupInfo(_testKernel, _device, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(kernelLocalMemSize), &kernelLocalMemSize, NULL);
  test_error(err, "clGetKernelWorkGroupInfo failed");

  cl_ulong deviceLocalMemSize = 0;
  err = clGetDeviceInfo(_device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(deviceLocalMemSize), &deviceLocalMemSize, NULL);
  test_error(err, "clGetDeviceInfo failed");

  if (kernelLocalMemSize + localArraySize > deviceLocalMemSize) {
    size_t adjustedLocalArraySize = deviceLocalMemSize - kernelLocalMemSize;
    log_info("localArraySize was adjusted from %zu to %zu\n", localArraySize,
             adjustedLocalArraySize);
    localArraySize = adjustedLocalArraySize;
  }

  return localArraySize;
}

size_t TestNonUniformWorkGroup::adjustGlobalBufferSize(size_t globalBufferSize) {
  // In case if global buffer size is too big, sometimes we can not run kernel because of lack
  // of resources due to kernel itself requires some global memory to run
  int err;

  cl_ulong deviceMaxAllocObjSize = 0;
  err = clGetDeviceInfo(_device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(deviceMaxAllocObjSize), &deviceMaxAllocObjSize, NULL);
  test_error(err, "clGetDeviceInfo failed");

  size_t adjustedGlobalBufferSize = globalBufferSize;
  if (deviceMaxAllocObjSize < globalBufferSize) {
    adjustedGlobalBufferSize = deviceMaxAllocObjSize;
    log_info("globalBufferSize was adjusted from %zu to %zu\n",
             globalBufferSize, adjustedGlobalBufferSize);
  }

  return adjustedGlobalBufferSize;
}

int TestNonUniformWorkGroup::runKernel () {
  int err;

  // TEST INFO
  showTestInfo();

  size_t localArraySize = (_localSize_IsNull)?TestNonUniformWorkGroup::getMaxLocalWorkgroupSize(_device):(_enqueuedLocalSize[0]*_enqueuedLocalSize[1]*_enqueuedLocalSize[2]);
  clMemWrapper resultsRegionArray = clCreateBuffer(_context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, _resultsRegionArray.size() * sizeof(DataContainerAttrib), &_resultsRegionArray.front(), &err);
  test_error(err, "clCreateBuffer failed");

  size_t *localSizePtr = (_localSize_IsNull)?NULL:_enqueuedLocalSize;
  size_t *globalWorkOffsetPtr = (_globalWorkOffset_IsNull)?NULL:_globalWorkOffset;

  err = clSetKernelArg(_testKernel, 0, sizeof(resultsRegionArray), &resultsRegionArray);
  test_error(err, "clSetKernelArg failed");

  //creating local buffer
  localArraySize = adjustLocalArraySize(localArraySize*sizeof(unsigned int));
  err = clSetKernelArg(_testKernel, 1, localArraySize, NULL);
  test_error(err, "clSetKernelArg failed");

  size_t globalBufferSize = adjustGlobalBufferSize(_numOfGlobalWorkItems*sizeof(cl_uint));
  clMemWrapper testGlobalArray = clCreateBuffer(_context, CL_MEM_READ_WRITE, globalBufferSize, NULL, &err);
  test_error(err, "clCreateBuffer failed");

  err = clSetKernelArg(_testKernel, 2, sizeof(testGlobalArray), &testGlobalArray);
  test_error(err, "clSetKernelArg failed");

  _globalAtomicTestValue = 0;
  clMemWrapper globalAtomicTestVariable = clCreateBuffer(_context, (CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR), sizeof(_globalAtomicTestValue), &_globalAtomicTestValue, &err);
  test_error(err, "clCreateBuffer failed");

  err = clSetKernelArg(_testKernel, 3, sizeof(globalAtomicTestVariable), &globalAtomicTestVariable);
  test_error(err, "clSetKernelArg failed");

  clMemWrapper errorArray = clCreateBuffer(_context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, _err.errorArrayCounterSize(), _err.errorArrayCounter(), &err);
  test_error(err, "clCreateBuffer failed");

  err = clSetKernelArg(_testKernel, 4, sizeof(errorArray), &errorArray);
  test_error(err, "clSetKernelArg failed");

  err = clEnqueueNDRangeKernel(_queue, _testKernel, _dims, globalWorkOffsetPtr, _globalSize,
    localSizePtr, 0, NULL, NULL);
  test_error(err, "clEnqueueNDRangeKernel failed");


  err = clFinish(_queue);
  test_error(err, "clFinish failed");

  err = clEnqueueReadBuffer(_queue, globalAtomicTestVariable, CL_TRUE, 0, sizeof(unsigned int), &_globalAtomicTestValue, 0, NULL, NULL);
  test_error(err, "clEnqueueReadBuffer failed");

  if (_err.checkError()) {
    return -1;
  }

  // synchronization of main buffer
  err = clEnqueueReadBuffer(_queue, resultsRegionArray, CL_TRUE, 0, _resultsRegionArray.size() * sizeof(DataContainerAttrib), &_resultsRegionArray.front(), 0, NULL, NULL);
  test_error(err, "clEnqueueReadBuffer failed");

  err = clEnqueueReadBuffer(_queue, errorArray, CL_TRUE, 0, _err.errorArrayCounterSize(), _err.errorArrayCounter(), 0, NULL, NULL);
  test_error(err, "clEnqueueReadBuffer failed");
  // Synchronization of errors occurred in kernel into general error stats
  _err.synchronizeStatsMap();

  return 0;
}

void SubTestExecutor::runTestNonUniformWorkGroup(const cl_uint dims,
                                                 size_t *globalSize,
                                                 const size_t *localSize,
                                                 int range)
{
    runTestNonUniformWorkGroup(dims, globalSize, localSize, NULL, NULL, range);
}

void SubTestExecutor::runTestNonUniformWorkGroup(
    const cl_uint dims, size_t *globalSize, const size_t *localSize,
    const size_t *globalWorkOffset, const size_t *reqdWorkGroupSize, int range)
{


    int err;
    ++_overallCounter;
    TestNonUniformWorkGroup test(_device, _context, _queue, dims, globalSize,
                                 localSize, NULL, globalWorkOffset,
                                 reqdWorkGroupSize);

    test.setTestRange(range);
    err = test.prepareDevice();
    if (err)
    {
        log_error("Error: prepare device\n");
        ++_failCounter;
        return;
    }

    err = test.runKernel();
    if (err)
    {
        log_error("Error: run kernel\n");
        ++_failCounter;
        return;
    }

    err = test.verifyResults();
    if (err)
    {
        log_error("Error: verify results\n");
        ++_failCounter;
        return;
    }
}

int SubTestExecutor::calculateWorkGroupSize(size_t &maxWgSize, int testRange) {
  int err;

  clProgramWrapper program;
  clKernelWrapper testKernel;
  std::string buildOptions{};

  if (testRange & Range::BASIC)
    buildOptions += " -D TESTBASIC";
  if (testRange & Range::ATOMICS)
    buildOptions += " -D TESTATOMICS";
  if (testRange & Range::BARRIERS)
    buildOptions += " -D TESTBARRIERS";

  err = create_single_kernel_helper_with_build_options (_context, &program, &testKernel, 1,
    &KERNEL_FUNCTION, "testKernel", buildOptions.c_str());
  if (err)
  {
    log_error("Error %d in line: %d of file %s\n", err, __LINE__, __FILE__);
    return err;
  }

  err = clGetKernelWorkGroupInfo (testKernel, _device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(maxWgSize), &maxWgSize, NULL);
  test_error(err, "clGetKernelWorkGroupInfo failed");

  TestNonUniformWorkGroup::setMaxLocalWorkgroupSize(maxWgSize);

  return 0;
}

int SubTestExecutor::status() {

  if (_failCounter>0) {
    log_error ("%d subtest(s) (of %d) failed\n", _failCounter, _overallCounter);
    return -1;
  } else {
    log_info ("All %d subtest(s) passed\n", _overallCounter);
    return 0;
  }
}

