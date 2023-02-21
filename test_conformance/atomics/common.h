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
#ifndef _COMMON_H_
#define _COMMON_H_

#include "harness/testHarness.h"
#include "harness/typeWrappers.h"
#include "harness/ThreadPool.h"

#include "host_atomics.h"

#include <vector>
#include <sstream>

#define MAX_DEVICE_THREADS (gHost ? 0U : gMaxDeviceThreads)
#define MAX_HOST_THREADS GetThreadCount()

#define EXECUTE_TEST(error, test)                                              \
    error |= test;                                                             \
    if (error && !gContinueOnError) return error;

enum TExplicitAtomicType
{
    TYPE_ATOMIC_INT,
    TYPE_ATOMIC_UINT,
    TYPE_ATOMIC_LONG,
    TYPE_ATOMIC_ULONG,
    TYPE_ATOMIC_FLOAT,
    TYPE_ATOMIC_DOUBLE,
    TYPE_ATOMIC_INTPTR_T,
    TYPE_ATOMIC_UINTPTR_T,
    TYPE_ATOMIC_SIZE_T,
    TYPE_ATOMIC_PTRDIFF_T,
    TYPE_ATOMIC_FLAG
};

enum TExplicitMemoryScopeType
{
    MEMORY_SCOPE_EMPTY,
    MEMORY_SCOPE_WORK_GROUP,
    MEMORY_SCOPE_DEVICE,
    MEMORY_SCOPE_ALL_DEVICES, // Alias for MEMORY_SCOPE_ALL_SVM_DEVICES
    MEMORY_SCOPE_ALL_SVM_DEVICES
};

extern bool
    gHost; // temporary flag for testing native host threads (test verification)
extern bool gOldAPI; // temporary flag for testing with old API (OpenCL 1.2)
extern bool gContinueOnError; // execute all cases even when errors detected
extern bool
    gNoGlobalVariables; // disable cases with global atomics in program scope
extern bool gNoGenericAddressSpace; // disable cases with generic address space
extern bool gUseHostPtr; // use malloc/free instead of clSVMAlloc/clSVMFree
extern bool gDebug; // print OpenCL kernel code
extern int gInternalIterations; // internal test iterations for atomic
                                // operation, sufficient to verify atomicity
extern int
    gMaxDeviceThreads; // maximum number of threads executed on OCL device
extern cl_device_atomic_capabilities gAtomicMemCap,
    gAtomicFenceCap; // atomic memory and fence capabilities for this device

extern const char *
get_memory_order_type_name(TExplicitMemoryOrderType orderType);
extern const char *
get_memory_scope_type_name(TExplicitMemoryScopeType scopeType);

extern cl_int getSupportedMemoryOrdersAndScopes(
    cl_device_id device, std::vector<TExplicitMemoryOrderType> &memoryOrders,
    std::vector<TExplicitMemoryScopeType> &memoryScopes);

class AtomicTypeInfo {
public:
    TExplicitAtomicType _type;
    AtomicTypeInfo(TExplicitAtomicType type): _type(type) {}
    cl_uint Size(cl_device_id device);
    const char *AtomicTypeName();
    const char *RegularTypeName();
    const char *AddSubOperandTypeName();
    int IsSupported(cl_device_id device);
};

template <typename HostDataType>
class AtomicTypeExtendedInfo : public AtomicTypeInfo {
public:
    AtomicTypeExtendedInfo(TExplicitAtomicType type): AtomicTypeInfo(type) {}
    HostDataType MinValue();
    HostDataType MaxValue();
    HostDataType SpecialValue(cl_uchar x)
    {
        HostDataType tmp;
        cl_uchar *ptr = (cl_uchar *)&tmp;
        for (cl_uint i = 0; i < sizeof(HostDataType) / sizeof(cl_uchar); i++)
            ptr[i] = x;
        return tmp;
    }
    HostDataType SpecialValue(cl_ushort x)
    {
        HostDataType tmp;
        cl_ushort *ptr = (cl_ushort *)&tmp;
        for (cl_uint i = 0; i < sizeof(HostDataType) / sizeof(cl_ushort); i++)
            ptr[i] = x;
        return tmp;
    }
};

class CTest {
public:
    virtual int Execute(cl_device_id deviceID, cl_context context,
                        cl_command_queue queue, int num_elements) = 0;
};

template <typename HostAtomicType, typename HostDataType>
class CBasicTest : CTest {
public:
    typedef struct
    {
        CBasicTest *test;
        cl_uint tid;
        cl_uint threadCount;
        volatile HostAtomicType *destMemory;
        HostDataType *oldValues;
    } THostThreadContext;
    static cl_int HostThreadFunction(cl_uint job_id, cl_uint thread_id,
                                     void *userInfo)
    {
        THostThreadContext *threadContext =
            ((THostThreadContext *)userInfo) + job_id;
        threadContext->test->HostFunction(
            threadContext->tid, threadContext->threadCount,
            threadContext->destMemory, threadContext->oldValues);
        return 0;
    }
    CBasicTest(TExplicitAtomicType dataType, bool useSVM)
        : CTest(), _maxDeviceThreads(MAX_DEVICE_THREADS), _dataType(dataType),
          _useSVM(useSVM), _startValue(255), _localMemory(false),
          _declaredInProgram(false), _usedInFunction(false),
          _genericAddrSpace(false), _oldValueCheck(true),
          _localRefValues(false), _maxGroupSize(0), _passCount(0),
          _iterations(gInternalIterations)
    {}
    virtual ~CBasicTest()
    {
        if (_passCount)
            log_info("  %u tests executed successfully for %s\n", _passCount,
                     DataType().AtomicTypeName());
    }
    virtual cl_uint NumResults(cl_uint threadCount, cl_device_id deviceID)
    {
        return 1;
    }
    virtual cl_uint NumNonAtomicVariablesPerThread() { return 1; }
    virtual bool ExpectedValue(HostDataType &expected, cl_uint threadCount,
                               HostDataType *startRefValues,
                               cl_uint whichDestValue)
    {
        return false;
    }
    virtual bool GenerateRefs(cl_uint threadCount, HostDataType *startRefValues,
                              MTdata d)
    {
        return false;
    }
    virtual bool VerifyRefs(bool &correct, cl_uint threadCount,
                            HostDataType *refValues,
                            HostAtomicType *finalValues)
    {
        return false;
    }
    virtual std::string PragmaHeader(cl_device_id deviceID);
    virtual std::string ProgramHeader(cl_uint maxNumDestItems);
    virtual std::string FunctionCode();
    virtual std::string KernelCode(cl_uint maxNumDestItems);
    virtual std::string ProgramCore() = 0;
    virtual std::string SingleTestName()
    {
        std::string testName = LocalMemory() ? "local" : "global";
        testName += " ";
        testName += DataType().AtomicTypeName();
        if (DeclaredInProgram())
        {
            testName += " declared in program";
        }
        if (DeclaredInProgram() && UsedInFunction()) testName += ",";
        if (UsedInFunction())
        {
            testName += " used in ";
            if (GenericAddrSpace()) testName += "generic ";
            testName += "function";
        }
        return testName;
    }
    virtual int ExecuteSingleTest(cl_device_id deviceID, cl_context context,
                                  cl_command_queue queue);
    int ExecuteForEachPointerType(cl_device_id deviceID, cl_context context,
                                  cl_command_queue queue)
    {
        int error = 0;
        UsedInFunction(false);
        EXECUTE_TEST(error, ExecuteSingleTest(deviceID, context, queue));
        UsedInFunction(true);
        GenericAddrSpace(false);
        EXECUTE_TEST(error, ExecuteSingleTest(deviceID, context, queue));
        GenericAddrSpace(true);
        EXECUTE_TEST(error, ExecuteSingleTest(deviceID, context, queue));
        GenericAddrSpace(false);
        return error;
    }
    int ExecuteForEachDeclarationType(cl_device_id deviceID, cl_context context,
                                      cl_command_queue queue)
    {
        int error = 0;
        DeclaredInProgram(false);
        EXECUTE_TEST(error,
                     ExecuteForEachPointerType(deviceID, context, queue));
        if (!UseSVM())
        {
            DeclaredInProgram(true);
            EXECUTE_TEST(error,
                         ExecuteForEachPointerType(deviceID, context, queue));
        }
        return error;
    }
    virtual int ExecuteForEachParameterSet(cl_device_id deviceID,
                                           cl_context context,
                                           cl_command_queue queue)
    {
        int error = 0;
        if (_maxDeviceThreads > 0 && !UseSVM())
        {
            LocalMemory(true);
            EXECUTE_TEST(
                error, ExecuteForEachDeclarationType(deviceID, context, queue));
        }
        if (_maxDeviceThreads + MaxHostThreads() > 0)
        {
            LocalMemory(false);
            EXECUTE_TEST(
                error, ExecuteForEachDeclarationType(deviceID, context, queue));
        }
        return error;
    }
    virtual int Execute(cl_device_id deviceID, cl_context context,
                        cl_command_queue queue, int num_elements)
    {
        if (sizeof(HostAtomicType) != DataType().Size(deviceID))
        {
            log_info("Invalid test: Host atomic type size (%u) is different "
                     "than OpenCL type size (%u)\n",
                     (cl_uint)sizeof(HostAtomicType),
                     DataType().Size(deviceID));
            return -1;
        }
        if (sizeof(HostAtomicType) != sizeof(HostDataType))
        {
            log_info("Invalid test: Host atomic type size (%u) is different "
                     "than corresponding type size (%u)\n",
                     (cl_uint)sizeof(HostAtomicType),
                     (cl_uint)sizeof(HostDataType));
            return -1;
        }
        // Verify we can run first
        if (UseSVM() && !gUseHostPtr)
        {
            cl_device_svm_capabilities caps;
            cl_int error = clGetDeviceInfo(deviceID, CL_DEVICE_SVM_CAPABILITIES,
                                           sizeof(caps), &caps, 0);
            test_error(error, "clGetDeviceInfo failed");
            if ((caps & CL_DEVICE_SVM_ATOMICS) == 0)
            {
                log_info("\t%s - SVM_ATOMICS not supported\n",
                         DataType().AtomicTypeName());
                // implicit pass
                return 0;
            }
        }
        if (!DataType().IsSupported(deviceID))
        {
            log_info("\t%s not supported\n", DataType().AtomicTypeName());
            // implicit pass or host test (debug feature)
            if (UseSVM()) return 0;
            _maxDeviceThreads = 0;
        }
        if (_maxDeviceThreads + MaxHostThreads() == 0) return 0;
        return ExecuteForEachParameterSet(deviceID, context, queue);
    }
    virtual void HostFunction(cl_uint tid, cl_uint threadCount,
                              volatile HostAtomicType *destMemory,
                              HostDataType *oldValues)
    {
        log_info("Empty thread function %u\n", (cl_uint)tid);
    }
    AtomicTypeExtendedInfo<HostDataType> DataType() const
    {
        return AtomicTypeExtendedInfo<HostDataType>(_dataType);
    }
    cl_uint _maxDeviceThreads;
    virtual cl_uint MaxHostThreads()
    {
        if (UseSVM() || gHost)
            return MAX_HOST_THREADS;
        else
            return 0;
    }

    int CheckCapabilities(TExplicitMemoryScopeType memoryScope,
                          TExplicitMemoryOrderType memoryOrder)
    {
        /*
            Differentiation between atomic fence and other atomic operations
            does not need to occur here.

            The initialisation of this test checks that the minimum required
            capabilities are supported by this device.

            The following switches allow the test to skip if optional
           capabilites are not supported by the device.
          */
        switch (memoryScope)
        {
            case MEMORY_SCOPE_EMPTY: {
                break;
            }
            case MEMORY_SCOPE_WORK_GROUP: {
                if ((gAtomicMemCap & CL_DEVICE_ATOMIC_SCOPE_WORK_GROUP) == 0)
                {
                    return TEST_SKIPPED_ITSELF;
                }
                break;
            }
            case MEMORY_SCOPE_DEVICE: {
                if ((gAtomicMemCap & CL_DEVICE_ATOMIC_SCOPE_DEVICE) == 0)
                {
                    return TEST_SKIPPED_ITSELF;
                }
                break;
            }
            case MEMORY_SCOPE_ALL_DEVICES: // fallthough
            case MEMORY_SCOPE_ALL_SVM_DEVICES: {
                if ((gAtomicMemCap & CL_DEVICE_ATOMIC_SCOPE_ALL_DEVICES) == 0)
                {
                    return TEST_SKIPPED_ITSELF;
                }
                break;
            }
            default: {
                log_info("Invalid memory scope\n");
                break;
            }
        }

        switch (memoryOrder)
        {
            case MEMORY_ORDER_EMPTY: {
                break;
            }
            case MEMORY_ORDER_RELAXED: {
                if ((gAtomicMemCap & CL_DEVICE_ATOMIC_ORDER_RELAXED) == 0)
                {
                    return TEST_SKIPPED_ITSELF;
                }
                break;
            }
            case MEMORY_ORDER_ACQUIRE:
            case MEMORY_ORDER_RELEASE:
            case MEMORY_ORDER_ACQ_REL: {
                if ((gAtomicMemCap & CL_DEVICE_ATOMIC_ORDER_ACQ_REL) == 0)
                {
                    return TEST_SKIPPED_ITSELF;
                }
                break;
            }
            case MEMORY_ORDER_SEQ_CST: {
                if ((gAtomicMemCap & CL_DEVICE_ATOMIC_ORDER_SEQ_CST) == 0)
                {
                    return TEST_SKIPPED_ITSELF;
                }
                break;
            }
            default: {
                log_info("Invalid memory order\n");
                break;
            }
        }

        return 0;
    }
    virtual bool SVMDataBufferAllSVMConsistent() { return false; }
    bool UseSVM() { return _useSVM; }
    void StartValue(HostDataType startValue) { _startValue = startValue; }
    HostDataType StartValue() { return _startValue; }
    void LocalMemory(bool local) { _localMemory = local; }
    bool LocalMemory() { return _localMemory; }
    void DeclaredInProgram(bool declaredInProgram)
    {
        _declaredInProgram = declaredInProgram;
    }
    bool DeclaredInProgram() { return _declaredInProgram; }
    void UsedInFunction(bool local) { _usedInFunction = local; }
    bool UsedInFunction() { return _usedInFunction; }
    void GenericAddrSpace(bool genericAddrSpace)
    {
        _genericAddrSpace = genericAddrSpace;
    }
    bool GenericAddrSpace() { return _genericAddrSpace; }
    void OldValueCheck(bool check) { _oldValueCheck = check; }
    bool OldValueCheck() { return _oldValueCheck; }
    void LocalRefValues(bool localRefValues)
    {
        _localRefValues = localRefValues;
    }
    bool LocalRefValues() { return _localRefValues; }
    void MaxGroupSize(cl_uint maxGroupSize) { _maxGroupSize = maxGroupSize; }
    cl_uint MaxGroupSize() { return _maxGroupSize; }
    void CurrentGroupSize(cl_uint currentGroupSize)
    {
        if (MaxGroupSize() && MaxGroupSize() < currentGroupSize)
            _currentGroupSize = MaxGroupSize();
        else
            _currentGroupSize = currentGroupSize;
    }
    cl_uint CurrentGroupSize() { return _currentGroupSize; }
    virtual cl_uint CurrentGroupNum(cl_uint threadCount)
    {
        if (threadCount == 0) return 0;
        if (LocalMemory()) return 1;
        return threadCount / CurrentGroupSize();
    }
    cl_int Iterations() { return _iterations; }
    std::string IterationsStr()
    {
        std::stringstream ss;
        ss << _iterations;
        return ss.str();
    }

private:
    const TExplicitAtomicType _dataType;
    const bool _useSVM;
    HostDataType _startValue;
    bool _localMemory;
    bool _declaredInProgram;
    bool _usedInFunction;
    bool _genericAddrSpace;
    bool _oldValueCheck;
    bool _localRefValues;
    cl_uint _maxGroupSize;
    cl_uint _currentGroupSize;
    cl_uint _passCount;
    const cl_int _iterations;
};

template <typename HostAtomicType, typename HostDataType>
class CBasicTestMemOrderScope
    : public CBasicTest<HostAtomicType, HostDataType> {
public:
    using CBasicTest<HostAtomicType, HostDataType>::LocalMemory;
    using CBasicTest<HostAtomicType, HostDataType>::MaxGroupSize;
    using CBasicTest<HostAtomicType, HostDataType>::CheckCapabilities;
    CBasicTestMemOrderScope(TExplicitAtomicType dataType, bool useSVM = false)
        : CBasicTest<HostAtomicType, HostDataType>(dataType, useSVM)
    {}
    virtual std::string ProgramHeader(cl_uint maxNumDestItems)
    {
        std::string header;
        if (gOldAPI)
        {
            std::string s = MemoryScope() == MEMORY_SCOPE_EMPTY ? "" : ",s";
            header += "#define atomic_store_explicit(x,y,o" + s
                + ")                     atomic_store(x,y)\n"
                  "#define atomic_load_explicit(x,o"
                + s
                + ")                        atomic_load(x)\n"
                  "#define atomic_exchange_explicit(x,y,o"
                + s
                + ")                  atomic_exchange(x,y)\n"
                  "#define atomic_compare_exchange_strong_explicit(x,y,z,os,of"
                + s
                + ") atomic_compare_exchange_strong(x,y,z)\n"
                  "#define atomic_compare_exchange_weak_explicit(x,y,z,os,of"
                + s
                + ")   atomic_compare_exchange_weak(x,y,z)\n"
                  "#define atomic_fetch_add_explicit(x,y,o"
                + s
                + ")                 atomic_fetch_add(x,y)\n"
                  "#define atomic_fetch_sub_explicit(x,y,o"
                + s
                + ")                 atomic_fetch_sub(x,y)\n"
                  "#define atomic_fetch_or_explicit(x,y,o"
                + s
                + ")                  atomic_fetch_or(x,y)\n"
                  "#define atomic_fetch_xor_explicit(x,y,o"
                + s
                + ")                 atomic_fetch_xor(x,y)\n"
                  "#define atomic_fetch_and_explicit(x,y,o"
                + s
                + ")                 atomic_fetch_and(x,y)\n"
                  "#define atomic_fetch_min_explicit(x,y,o"
                + s
                + ")                 atomic_fetch_min(x,y)\n"
                  "#define atomic_fetch_max_explicit(x,y,o"
                + s
                + ")                 atomic_fetch_max(x,y)\n"
                  "#define atomic_flag_test_and_set_explicit(x,o"
                + s
                + ")           atomic_flag_test_and_set(x)\n"
                  "#define atomic_flag_clear_explicit(x,o"
                + s + ")                  atomic_flag_clear(x)\n";
        }
        return header
            + CBasicTest<HostAtomicType, HostDataType>::ProgramHeader(
                   maxNumDestItems);
    }
    virtual std::string SingleTestName()
    {
        std::string testName =
            CBasicTest<HostAtomicType, HostDataType>::SingleTestName();
        if (MemoryOrder() != MEMORY_ORDER_EMPTY)
        {
            testName += std::string(", ")
                + std::string(get_memory_order_type_name(MemoryOrder()))
                      .substr(sizeof("memory"));
        }
        if (MemoryScope() != MEMORY_SCOPE_EMPTY)
        {
            testName += std::string(", ")
                + std::string(get_memory_scope_type_name(MemoryScope()))
                      .substr(sizeof("memory"));
        }
        return testName;
    }
    virtual int ExecuteSingleTest(cl_device_id deviceID, cl_context context,
                                  cl_command_queue queue)
    {
        if (LocalMemory() && MemoryScope() != MEMORY_SCOPE_EMPTY
            && MemoryScope()
                != MEMORY_SCOPE_WORK_GROUP) // memory scope should only be used
                                            // for global memory
            return 0;
        if (MemoryScope() == MEMORY_SCOPE_DEVICE)
            MaxGroupSize(
                16); // increase number of groups by forcing smaller group size
        else
            MaxGroupSize(0); // group size limited by device capabilities

        if (CheckCapabilities(MemoryScope(), MemoryOrder())
            == TEST_SKIPPED_ITSELF)
            return 0; // skip test - not applicable

        return CBasicTest<HostAtomicType, HostDataType>::ExecuteSingleTest(
            deviceID, context, queue);
    }
    virtual int ExecuteForEachParameterSet(cl_device_id deviceID,
                                           cl_context context,
                                           cl_command_queue queue)
    {
        // repeat test for each reasonable memory order/scope combination
        std::vector<TExplicitMemoryOrderType> memoryOrder;
        std::vector<TExplicitMemoryScopeType> memoryScope;
        int error = 0;

        // For OpenCL-3.0 and later some orderings and scopes are optional, so
        // here we query for the supported ones.
        test_error_ret(getSupportedMemoryOrdersAndScopes(deviceID, memoryOrder,
                                                         memoryScope),
                       "getSupportedMemoryOrdersAndScopes failed\n", TEST_FAIL);

        for (unsigned oi = 0; oi < memoryOrder.size(); oi++)
        {
            for (unsigned si = 0; si < memoryScope.size(); si++)
            {
                if (memoryOrder[oi] == MEMORY_ORDER_EMPTY
                    && memoryScope[si] != MEMORY_SCOPE_EMPTY)
                    continue;
                MemoryOrder(memoryOrder[oi]);
                MemoryScope(memoryScope[si]);
                EXECUTE_TEST(
                    error,
                    (CBasicTest<HostAtomicType, HostDataType>::
                         ExecuteForEachParameterSet(deviceID, context, queue)));
            }
        }
        return error;
    }
    void MemoryOrder(TExplicitMemoryOrderType memoryOrder)
    {
        _memoryOrder = memoryOrder;
    }
    TExplicitMemoryOrderType MemoryOrder() { return _memoryOrder; }
    std::string MemoryOrderStr()
    {
        if (MemoryOrder() != MEMORY_ORDER_EMPTY)
            return std::string(", ")
                + get_memory_order_type_name(MemoryOrder());
        return "";
    }
    void MemoryScope(TExplicitMemoryScopeType memoryScope)
    {
        _memoryScope = memoryScope;
    }
    TExplicitMemoryScopeType MemoryScope() { return _memoryScope; }
    std::string MemoryScopeStr()
    {
        if (MemoryScope() != MEMORY_SCOPE_EMPTY)
            return std::string(", ")
                + get_memory_scope_type_name(MemoryScope());
        return "";
    }
    std::string MemoryOrderScopeStr()
    {
        return MemoryOrderStr() + MemoryScopeStr();
    }
    virtual cl_uint CurrentGroupNum(cl_uint threadCount)
    {
        if (MemoryScope() == MEMORY_SCOPE_WORK_GROUP) return 1;
        return CBasicTest<HostAtomicType, HostDataType>::CurrentGroupNum(
            threadCount);
    }
    virtual cl_uint MaxHostThreads()
    {
        // block host threads execution for memory scope different than
        // memory_scope_all_svm_devices
        if (MemoryScope() == MEMORY_SCOPE_ALL_DEVICES
            || MemoryScope() == MEMORY_SCOPE_ALL_SVM_DEVICES || gHost)
        {
            return CBasicTest<HostAtomicType, HostDataType>::MaxHostThreads();
        }
        else
        {
            return 0;
        }
    }

private:
    TExplicitMemoryOrderType _memoryOrder;
    TExplicitMemoryScopeType _memoryScope;
};

template <typename HostAtomicType, typename HostDataType>
class CBasicTestMemOrder2Scope
    : public CBasicTestMemOrderScope<HostAtomicType, HostDataType> {
public:
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::LocalMemory;
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::MemoryOrder;
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::MemoryScope;
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::MemoryOrderStr;
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::MemoryScopeStr;
    using CBasicTest<HostAtomicType, HostDataType>::CheckCapabilities;

    CBasicTestMemOrder2Scope(TExplicitAtomicType dataType, bool useSVM = false)
        : CBasicTestMemOrderScope<HostAtomicType, HostDataType>(dataType,
                                                                useSVM)
    {}
    virtual std::string SingleTestName()
    {
        std::string testName =
            CBasicTest<HostAtomicType, HostDataType>::SingleTestName();
        if (MemoryOrder() != MEMORY_ORDER_EMPTY)
            testName += std::string(", ")
                + std::string(get_memory_order_type_name(MemoryOrder()))
                      .substr(sizeof("memory"));
        if (MemoryOrder2() != MEMORY_ORDER_EMPTY)
            testName += std::string(", ")
                + std::string(get_memory_order_type_name(MemoryOrder2()))
                      .substr(sizeof("memory"));
        if (MemoryScope() != MEMORY_SCOPE_EMPTY)
            testName += std::string(", ")
                + std::string(get_memory_scope_type_name(MemoryScope()))
                      .substr(sizeof("memory"));
        return testName;
    }
    virtual int ExecuteForEachParameterSet(cl_device_id deviceID,
                                           cl_context context,
                                           cl_command_queue queue)
    {
        // repeat test for each reasonable memory order/scope combination
        std::vector<TExplicitMemoryOrderType> memoryOrder;
        std::vector<TExplicitMemoryScopeType> memoryScope;
        int error = 0;

        // For OpenCL-3.0 and later some orderings and scopes are optional, so
        // here we query for the supported ones.
        test_error_ret(getSupportedMemoryOrdersAndScopes(deviceID, memoryOrder,
                                                         memoryScope),
                       "getSupportedMemoryOrdersAndScopes failed\n", TEST_FAIL);

        for (unsigned oi = 0; oi < memoryOrder.size(); oi++)
        {
            for (unsigned o2i = 0; o2i < memoryOrder.size(); o2i++)
            {
                for (unsigned si = 0; si < memoryScope.size(); si++)
                {
                    if ((memoryOrder[oi] == MEMORY_ORDER_EMPTY
                         || memoryOrder[o2i] == MEMORY_ORDER_EMPTY)
                        && memoryOrder[oi] != memoryOrder[o2i])
                        continue; // both memory order arguments must be set (or
                                  // none)
                    if ((memoryOrder[oi] == MEMORY_ORDER_EMPTY
                         || memoryOrder[o2i] == MEMORY_ORDER_EMPTY)
                        && memoryScope[si] != MEMORY_SCOPE_EMPTY)
                        continue; // memory scope without memory order is not
                                  // allowed
                    MemoryOrder(memoryOrder[oi]);
                    MemoryOrder2(memoryOrder[o2i]);
                    MemoryScope(memoryScope[si]);

                    if (CheckCapabilities(MemoryScope(), MemoryOrder())
                        == TEST_SKIPPED_ITSELF)
                        continue; // skip test - not applicable

                    if (CheckCapabilities(MemoryScope(), MemoryOrder2())
                        == TEST_SKIPPED_ITSELF)
                        continue; // skip test - not applicable

                    EXECUTE_TEST(error,
                                 (CBasicTest<HostAtomicType, HostDataType>::
                                      ExecuteForEachParameterSet(
                                          deviceID, context, queue)));
                }
            }
        }
        return error;
    }
    void MemoryOrder2(TExplicitMemoryOrderType memoryOrderFail)
    {
        _memoryOrder2 = memoryOrderFail;
    }
    TExplicitMemoryOrderType MemoryOrder2() { return _memoryOrder2; }
    std::string MemoryOrderFailStr()
    {
        if (MemoryOrder2() != MEMORY_ORDER_EMPTY)
            return std::string(", ")
                + get_memory_order_type_name(MemoryOrder2());
        return "";
    }
    std::string MemoryOrderScope()
    {
        return MemoryOrderStr() + MemoryOrderFailStr() + MemoryScopeStr();
    }

private:
    TExplicitMemoryOrderType _memoryOrder2;
};

template <typename HostAtomicType, typename HostDataType>
std::string
CBasicTest<HostAtomicType, HostDataType>::PragmaHeader(cl_device_id deviceID)
{
    std::string pragma;

    if (gOldAPI)
    {
        pragma += "#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : "
                  "enable\n";
        pragma += "#pragma OPENCL EXTENSION "
                  "cl_khr_local_int32_extended_atomics : enable\n";
        pragma += "#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : "
                  "enable\n";
        pragma += "#pragma OPENCL EXTENSION "
                  "cl_khr_global_int32_extended_atomics : enable\n";
    }
    // Create the pragma lines for this kernel
    if (DataType().Size(deviceID) == 8)
    {
        pragma +=
            "#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable\n";
        pragma +=
            "#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable\n";
    }
    if (_dataType == TYPE_ATOMIC_DOUBLE)
        pragma += "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
    return pragma;
}

template <typename HostAtomicType, typename HostDataType>
std::string
CBasicTest<HostAtomicType, HostDataType>::ProgramHeader(cl_uint maxNumDestItems)
{
    // Create the program header
    std::string header;
    std::string aTypeName = DataType().AtomicTypeName();
    std::string cTypeName = DataType().RegularTypeName();
    std::string argListForKernel;
    std::string argListForFunction;
    std::string argListNoTypes;
    std::string functionPrototype;
    std::string addressSpace = LocalMemory() ? "__local " : "__global ";

    if (gOldAPI)
    {
        header += std::string("#define ") + aTypeName + " " + cTypeName
            + "\n"
              "#define atomic_store(x,y)                                (*(x) "
              "= y)\n"
              "#define atomic_load(x)                                   "
              "(*(x))\n"
              "#define ATOMIC_VAR_INIT(x)                               (x)\n"
              "#define ATOMIC_FLAG_INIT                                 0\n"
              "#define atomic_init(x,y)                                 "
              "atomic_store(x,y)\n";
        if (aTypeName == "atomic_float")
            header += "#define atomic_exchange(x,y)                            "
                      " atomic_xchg(x,y)\n";
        else if (aTypeName == "atomic_double")
            header += "double atomic_exchange(volatile " + addressSpace
                + "atomic_double *x, double y)\n"
                  "{\n"
                  "  long tmp = *(long*)&y, res;\n"
                  "  volatile "
                + addressSpace + "long *tmpA = (volatile " + addressSpace
                + "long)x;\n"
                  "  res = atom_xchg(tmpA,tmp);\n"
                  "  return *(double*)&res;\n"
                  "}\n";
        else
            header += "#define atomic_exchange(x,y)                            "
                      " atom_xchg(x,y)\n";
        if (aTypeName != "atomic_float" && aTypeName != "atomic_double")
            header += "bool atomic_compare_exchange_strong(volatile "
                + addressSpace + " " + aTypeName + " *a, " + cTypeName
                + " *expected, " + cTypeName
                + " desired)\n"
                  "{\n"
                  "  "
                + cTypeName
                + " old = atom_cmpxchg(a, *expected, desired);\n"
                  "  if(old == *expected)\n"
                  "    return true;\n"
                  "  *expected = old;\n"
                  "  return false;\n"
                  "}\n"
                  "#define atomic_compare_exchange_weak                     "
                  "atomic_compare_exchange_strong\n";
        header += "#define atomic_fetch_add(x,y)                            "
                  "atom_add(x,y)\n"
                  "#define atomic_fetch_sub(x,y)                            "
                  "atom_sub(x,y)\n"
                  "#define atomic_fetch_or(x,y)                             "
                  "atom_or(x,y)\n"
                  "#define atomic_fetch_xor(x,y)                            "
                  "atom_xor(x,y)\n"
                  "#define atomic_fetch_and(x,y)                            "
                  "atom_and(x,y)\n"
                  "#define atomic_fetch_min(x,y)                            "
                  "atom_min(x,y)\n"
                  "#define atomic_fetch_max(x,y)                            "
                  "atom_max(x,y)\n"
                  "#define atomic_flag_test_and_set(x)                      "
                  "atomic_exchange(x,1)\n"
                  "#define atomic_flag_clear(x)                             "
                  "atomic_store(x,0)\n"
                  "\n";
    }
    if (!LocalMemory() && DeclaredInProgram())
    {
        // additional atomic variable for results copying (last thread will do
        // this)
        header += "__global volatile atomic_uint finishedThreads = "
                  "ATOMIC_VAR_INIT(0);\n";
        // atomic variables declared in program scope - test data
        std::stringstream ss;
        ss << maxNumDestItems;
        header += std::string("__global volatile ") + aTypeName + " destMemory["
            + ss.str() + "] = {\n";
        ss.str("");
        ss << _startValue;
        for (cl_uint i = 0; i < maxNumDestItems; i++)
        {
            if (aTypeName == "atomic_flag")
                header += "  ATOMIC_FLAG_INIT";
            else
                header += "  ATOMIC_VAR_INIT(" + ss.str() + ")";
            if (i + 1 < maxNumDestItems) header += ",";
            header += "\n";
        }
        header += "};\n"
                  "\n";
    }
    return header;
}

template <typename HostAtomicType, typename HostDataType>
std::string CBasicTest<HostAtomicType, HostDataType>::FunctionCode()
{
    if (!UsedInFunction()) return "";
    std::string addressSpace = LocalMemory() ? "__local " : "__global ";
    std::string code = "void test_atomic_function(uint tid, uint threadCount, "
                       "uint numDestItems, volatile ";
    if (!GenericAddrSpace()) code += addressSpace;
    code += std::string(DataType().AtomicTypeName()) + " *destMemory, __global "
        + DataType().RegularTypeName() + " *oldValues";
    if (LocalRefValues())
        code += std::string(", __local ") + DataType().RegularTypeName()
            + " *localValues";
    code += ")\n"
            "{\n";
    code += ProgramCore();
    code += "}\n"
            "\n";
    return code;
}

template <typename HostAtomicType, typename HostDataType>
std::string
CBasicTest<HostAtomicType, HostDataType>::KernelCode(cl_uint maxNumDestItems)
{
    std::string aTypeName = DataType().AtomicTypeName();
    std::string cTypeName = DataType().RegularTypeName();
    std::string addressSpace = LocalMemory() ? "__local " : "__global ";
    std::string code = "__kernel void test_atomic_kernel(uint threadCount, "
                       "uint numDestItems, ";

    // prepare list of arguments for kernel
    if (LocalMemory())
    {
        code += std::string("__global ") + cTypeName + " *finalDest, __global "
            + cTypeName
            + " *oldValues,"
              " volatile "
            + addressSpace + aTypeName + " *"
            + (DeclaredInProgram() ? "notUsed" : "") + "destMemory";
    }
    else
    {
        code += "volatile " + addressSpace
            + (DeclaredInProgram() ? (cTypeName + " *finalDest")
                                   : (aTypeName + " *destMemory"))
            + ", __global " + cTypeName + " *oldValues";
    }
    if (LocalRefValues())
        code += std::string(", __local ") + cTypeName + " *localValues";
    code += ")\n"
            "{\n";
    if (LocalMemory() && DeclaredInProgram())
    {
        // local atomics declared in kernel scope
        std::stringstream ss;
        ss << maxNumDestItems;
        code += std::string("  __local volatile ") + aTypeName + " destMemory["
            + ss.str() + "];\n";
    }
    code += "  uint  tid = get_global_id(0);\n"
            "\n";
    if (LocalMemory())
    {
        // memory_order_relaxed is sufficient for these initialization
        // operations as the barrier below will act as a fence, providing an
        // order to the operations. memory_scope_work_group is sufficient as
        // local memory is only visible within the work-group.
        code += R"(
              // initialize atomics not reachable from host (first thread
              // is doing this, other threads are waiting on barrier)
              if(get_local_id(0) == 0)
                for(uint dstItemIdx = 0; dstItemIdx < numDestItems; dstItemIdx++)
                {)";
        if (aTypeName == "atomic_flag")
        {
            code += R"(
                  if(finalDest[dstItemIdx])
                    atomic_flag_test_and_set_explicit(destMemory+dstItemIdx,
                                                      memory_order_relaxed,
                                                      memory_scope_work_group);
                  else
                    atomic_flag_clear_explicit(destMemory+dstItemIdx,
                                               memory_order_relaxed,
                                               memory_scope_work_group);)";
        }
        else
        {
            code += R"(
                atomic_store_explicit(destMemory+dstItemIdx,
                                      finalDest[dstItemIdx],
                                      memory_order_relaxed,
                                      memory_scope_work_group);)";
        }
        code += "    }\n"
                "  barrier(CLK_LOCAL_MEM_FENCE);\n"
                "\n";
    }
    if (LocalRefValues())
    {
        code += "  // Copy input reference values into local memory\n";
        if (NumNonAtomicVariablesPerThread() == 1)
            code += "  localValues[get_local_id(0)] = oldValues[tid];\n";
        else
        {
            std::stringstream ss;
            ss << NumNonAtomicVariablesPerThread();
            code += "  for(uint rfId = 0; rfId < " + ss.str()
                + "; rfId++)\n"
                  "    localValues[get_local_id(0)*"
                + ss.str() + "+rfId] = oldValues[tid*" + ss.str() + "+rfId];\n";
        }
        code += "  barrier(CLK_LOCAL_MEM_FENCE);\n"
                "\n";
    }
    if (UsedInFunction())
        code += std::string("  test_atomic_function(tid, threadCount, "
                            "numDestItems, destMemory, oldValues")
            + (LocalRefValues() ? ", localValues" : "") + ");\n";
    else
        code += ProgramCore();
    code += "\n";
    if (LocalRefValues())
    {
        code += "  // Copy local reference values into output array\n"
                "  barrier(CLK_LOCAL_MEM_FENCE);\n";
        if (NumNonAtomicVariablesPerThread() == 1)
            code += "  oldValues[tid] = localValues[get_local_id(0)];\n";
        else
        {
            std::stringstream ss;
            ss << NumNonAtomicVariablesPerThread();
            code += "  for(uint rfId = 0; rfId < " + ss.str()
                + "; rfId++)\n"
                  "    oldValues[tid*"
                + ss.str() + "+rfId] = localValues[get_local_id(0)*" + ss.str()
                + "+rfId];\n";
        }
        code += "\n";
    }
    if (LocalMemory())
    {
        code += "  // Copy final values to host reachable buffer\n";
        code += "  barrier(CLK_LOCAL_MEM_FENCE);\n"
                "  if(get_local_id(0) == 0) // first thread in workgroup\n";
        code += "    for(uint dstItemIdx = 0; dstItemIdx < numDestItems; "
                "dstItemIdx++)\n";
        if (aTypeName == "atomic_flag")
        {
            code += R"(
                finalDest[dstItemIdx] =
                    atomic_flag_test_and_set_explicit(destMemory+dstItemIdx,
                                                      memory_order_relaxed,
                                                      memory_scope_work_group);)";
        }
        else
        {
            code += R"(
                finalDest[dstItemIdx] =
                    atomic_load_explicit(destMemory+dstItemIdx,
                                         memory_order_relaxed,
                                         memory_scope_work_group);)";
        }
    }
    else if (DeclaredInProgram())
    {
        // global atomics declared in program scope
        code += "  // Copy final values to host reachable buffer\n";
        code += R"(
            if(atomic_fetch_add_explicit(&finishedThreads, 1u,
                                         memory_order_acq_rel,
                                         memory_scope_device)
                   == get_global_size(0)-1) // last finished thread
                )";
        code += "    for(uint dstItemIdx = 0; dstItemIdx < numDestItems; "
                "dstItemIdx++)\n";
        if (aTypeName == "atomic_flag")
        {
            code += R"(
                finalDest[dstItemIdx] =
                    atomic_flag_test_and_set_explicit(destMemory+dstItemIdx,
                                                      memory_order_relaxed,
                                                      memory_scope_device);)";
        }
        else
        {
            code += R"(
                finalDest[dstItemIdx] =
                    atomic_load_explicit(destMemory+dstItemIdx,
                                         memory_order_relaxed,
                                         memory_scope_device);)";
        }
    }
    code += "}\n"
            "\n";
    return code;
}

template <typename HostAtomicType, typename HostDataType>
int CBasicTest<HostAtomicType, HostDataType>::ExecuteSingleTest(
    cl_device_id deviceID, cl_context context, cl_command_queue queue)
{
    int error;
    clProgramWrapper program;
    clKernelWrapper kernel;
    size_t threadNum[1];
    clMemWrapper streams[2];
    std::vector<HostAtomicType> destItems;
    HostAtomicType *svmAtomicBuffer = 0;
    std::vector<HostDataType> refValues, startRefValues;
    HostDataType *svmDataBuffer = 0;
    cl_uint deviceThreadCount, hostThreadCount, threadCount;
    size_t groupSize = 0;
    std::string programSource;
    const char *programLine;
    MTdata d;
    size_t typeSize = DataType().Size(deviceID);

    deviceThreadCount = _maxDeviceThreads;
    hostThreadCount = MaxHostThreads();
    threadCount = deviceThreadCount + hostThreadCount;

    // log_info("\t%s %s%s...\n", local ? "local" : "global",
    // DataType().AtomicTypeName(), memoryOrderScope.c_str());
    log_info("\t%s...\n", SingleTestName().c_str());

    if (!LocalMemory() && DeclaredInProgram()
        && gNoGlobalVariables) // no support for program scope global variables
    {
        log_info("\t\tTest disabled\n");
        return 0;
    }
    if (UsedInFunction() && GenericAddrSpace() && gNoGenericAddressSpace)
    {
        log_info("\t\tTest disabled\n");
        return 0;
    }
    if (!LocalMemory() && DeclaredInProgram())
    {
        if (((gAtomicMemCap & CL_DEVICE_ATOMIC_SCOPE_DEVICE) == 0)
            || ((gAtomicMemCap & CL_DEVICE_ATOMIC_ORDER_ACQ_REL) == 0))
        {
            log_info("\t\tTest disabled\n");
            return 0;
        }
    }

    // set up work sizes based on device capabilities and test configuration
    error = clGetDeviceInfo(deviceID, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                            sizeof(groupSize), &groupSize, NULL);
    test_error(error, "Unable to obtain max work group size for device");
    CurrentGroupSize((cl_uint)groupSize);
    if (CurrentGroupSize() > deviceThreadCount)
        CurrentGroupSize(deviceThreadCount);
    if (CurrentGroupNum(deviceThreadCount) == 1 || gOldAPI)
        deviceThreadCount =
            CurrentGroupSize() * CurrentGroupNum(deviceThreadCount);
    threadCount = deviceThreadCount + hostThreadCount;

    // If we're given a num_results function, we need to determine how many
    // result objects we need. This is the first assessment for current maximum
    // number of threads (exact thread count is not known here)
    // - needed for program source code generation (arrays of atomics declared
    // in program)
    cl_uint numDestItems = NumResults(threadCount, deviceID);

    if (deviceThreadCount > 0)
    {
        // This loop iteratively reduces the workgroup size by 2 and then
        // re-generates the kernel with the reduced
        // workgroup size until we find a size which is admissible for the
        // kernel being run or reduce the wg size to the trivial case of 1
        // (which was separately verified to be accurate for the kernel being
        // run)

        while ((CurrentGroupSize() > 1))
        {
            // Re-generate the kernel code with the current group size
            if (kernel) clReleaseKernel(kernel);
            if (program) clReleaseProgram(program);
            programSource = PragmaHeader(deviceID) + ProgramHeader(numDestItems)
                + FunctionCode() + KernelCode(numDestItems);
            programLine = programSource.c_str();
            if (create_single_kernel_helper_with_build_options(
                    context, &program, &kernel, 1, &programLine,
                    "test_atomic_kernel", gOldAPI ? "" : nullptr))
            {
                return -1;
            }
            // Get work group size for the new kernel
            error = clGetKernelWorkGroupInfo(
                kernel, deviceID, CL_KERNEL_WORK_GROUP_SIZE, sizeof(groupSize),
                &groupSize, NULL);
            test_error(error,
                       "Unable to obtain max work group size for device and "
                       "kernel combo");

            if (LocalMemory())
            {
                cl_ulong usedLocalMemory;
                cl_ulong totalLocalMemory;
                cl_uint maxWorkGroupSize;

                error = clGetKernelWorkGroupInfo(
                    kernel, deviceID, CL_KERNEL_LOCAL_MEM_SIZE,
                    sizeof(usedLocalMemory), &usedLocalMemory, NULL);
                test_error(error, "clGetKernelWorkGroupInfo failed");

                error = clGetDeviceInfo(deviceID, CL_DEVICE_LOCAL_MEM_SIZE,
                                        sizeof(totalLocalMemory),
                                        &totalLocalMemory, NULL);
                test_error(error, "clGetDeviceInfo failed");

                // We know that each work-group is going to use typeSize *
                // deviceThreadCount bytes of local memory
                // so pick the maximum value for deviceThreadCount that uses all
                // the local memory.
                maxWorkGroupSize =
                    ((totalLocalMemory - usedLocalMemory) / typeSize);

                if (maxWorkGroupSize < groupSize) groupSize = maxWorkGroupSize;
            }
            if (CurrentGroupSize() <= groupSize)
                break;
            else
                CurrentGroupSize(CurrentGroupSize() / 2);
        }
        if (CurrentGroupSize() > deviceThreadCount)
            CurrentGroupSize(deviceThreadCount);
        if (CurrentGroupNum(deviceThreadCount) == 1 || gOldAPI)
            deviceThreadCount =
                CurrentGroupSize() * CurrentGroupNum(deviceThreadCount);
        threadCount = deviceThreadCount + hostThreadCount;
    }
    if (gDebug)
    {
        log_info("Program source:\n");
        log_info("%s\n", programLine);
    }
    if (deviceThreadCount > 0)
        log_info("\t\t(thread count %u, group size %u)\n", deviceThreadCount,
                 CurrentGroupSize());
    if (hostThreadCount > 0)
        log_info("\t\t(host threads %u)\n", hostThreadCount);

    refValues.resize(threadCount * NumNonAtomicVariablesPerThread());

    // Generate ref data if we have a ref generator provided
    d = init_genrand(gRandomSeed);
    startRefValues.resize(threadCount * NumNonAtomicVariablesPerThread());
    if (GenerateRefs(threadCount, &startRefValues[0], d))
    {
        // copy ref values for host threads
        memcpy(&refValues[0], &startRefValues[0],
               sizeof(HostDataType) * threadCount
                   * NumNonAtomicVariablesPerThread());
    }
    else
    {
        startRefValues.resize(0);
    }
    free_mtdata(d);
    d = NULL;

    // If we're given a num_results function, we need to determine how many
    // result objects we need. If we don't have it, we assume it's just 1 This
    // is final value (exact thread count is known in this place)
    numDestItems = NumResults(threadCount, deviceID);

    destItems.resize(numDestItems);
    for (cl_uint i = 0; i < numDestItems; i++) destItems[i] = _startValue;

    // Create main buffer with atomic variables (array size dependent on
    // particular test)
    if (UseSVM())
    {
        if (gUseHostPtr)
            svmAtomicBuffer = (HostAtomicType *)malloc(typeSize * numDestItems);
        else
            svmAtomicBuffer = (HostAtomicType *)clSVMAlloc(
                context, CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS,
                typeSize * numDestItems, 0);
        if (!svmAtomicBuffer)
        {
            log_error("ERROR: clSVMAlloc failed!\n");
            return -1;
        }
        memcpy(svmAtomicBuffer, &destItems[0], typeSize * numDestItems);
        streams[0] =
            clCreateBuffer(context, CL_MEM_USE_HOST_PTR,
                           typeSize * numDestItems, svmAtomicBuffer, NULL);
    }
    else
    {
        streams[0] =
            clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                           typeSize * numDestItems, &destItems[0], NULL);
    }
    if (!streams[0])
    {
        log_error("ERROR: Creating output array failed!\n");
        return -1;
    }
    // Create buffer for per-thread input/output data
    if (UseSVM())
    {
        if (gUseHostPtr)
            svmDataBuffer = (HostDataType *)malloc(
                typeSize * threadCount * NumNonAtomicVariablesPerThread());
        else
            svmDataBuffer = (HostDataType *)clSVMAlloc(
                context,
                CL_MEM_SVM_FINE_GRAIN_BUFFER
                    | (SVMDataBufferAllSVMConsistent() ? CL_MEM_SVM_ATOMICS
                                                       : 0),
                typeSize * threadCount * NumNonAtomicVariablesPerThread(), 0);
        if (!svmDataBuffer)
        {
            log_error("ERROR: clSVMAlloc failed!\n");
            return -1;
        }
        if (startRefValues.size())
            memcpy(svmDataBuffer, &startRefValues[0],
                   typeSize * threadCount * NumNonAtomicVariablesPerThread());
        streams[1] = clCreateBuffer(context, CL_MEM_USE_HOST_PTR,
                                    typeSize * threadCount
                                        * NumNonAtomicVariablesPerThread(),
                                    svmDataBuffer, NULL);
    }
    else
    {
        streams[1] = clCreateBuffer(
            context,
            ((startRefValues.size() ? CL_MEM_COPY_HOST_PTR
                                    : CL_MEM_READ_WRITE)),
            typeSize * threadCount * NumNonAtomicVariablesPerThread(),
            startRefValues.size() ? &startRefValues[0] : 0, NULL);
    }
    if (!streams[1])
    {
        log_error("ERROR: Creating reference array failed!\n");
        return -1;
    }
    if (deviceThreadCount > 0)
    {
        cl_uint argInd = 0;
        /* Set the arguments */
        error =
            clSetKernelArg(kernel, argInd++, sizeof(threadCount), &threadCount);
        test_error(error, "Unable to set kernel argument");
        error = clSetKernelArg(kernel, argInd++, sizeof(numDestItems),
                               &numDestItems);
        test_error(error, "Unable to set indexed kernel argument");
        error =
            clSetKernelArg(kernel, argInd++, sizeof(streams[0]), &streams[0]);
        test_error(error, "Unable to set indexed kernel arguments");
        error =
            clSetKernelArg(kernel, argInd++, sizeof(streams[1]), &streams[1]);
        test_error(error, "Unable to set indexed kernel arguments");
        if (LocalMemory())
        {
            error =
                clSetKernelArg(kernel, argInd++, typeSize * numDestItems, NULL);
            test_error(error, "Unable to set indexed local kernel argument");
        }
        if (LocalRefValues())
        {
            error =
                clSetKernelArg(kernel, argInd++,
                               LocalRefValues() ? typeSize
                                       * (CurrentGroupSize()
                                          * NumNonAtomicVariablesPerThread())
                                                : 1,
                               NULL);
            test_error(error, "Unable to set indexed kernel argument");
        }
    }
    /* Configure host threads */
    std::vector<THostThreadContext> hostThreadContexts(hostThreadCount);
    for (unsigned int t = 0; t < hostThreadCount; t++)
    {
        hostThreadContexts[t].test = this;
        hostThreadContexts[t].tid = deviceThreadCount + t;
        hostThreadContexts[t].threadCount = threadCount;
        hostThreadContexts[t].destMemory =
            UseSVM() ? svmAtomicBuffer : &destItems[0];
        hostThreadContexts[t].oldValues =
            UseSVM() ? svmDataBuffer : &refValues[0];
    }

    if (deviceThreadCount > 0)
    {
        /* Run the kernel */
        threadNum[0] = deviceThreadCount;
        groupSize = CurrentGroupSize();
        error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, threadNum,
                                       &groupSize, 0, NULL, NULL);
        test_error(error, "Unable to execute test kernel");
        /* start device threads */
        error = clFlush(queue);
        test_error(error, "clFlush failed");
    }

    /* Start host threads and wait for finish */
    if (hostThreadCount > 0)
        ThreadPool_Do(HostThreadFunction, hostThreadCount,
                      &hostThreadContexts[0]);

    if (UseSVM())
    {
        error = clFinish(queue);
        test_error(error, "clFinish failed");
        memcpy(&destItems[0], svmAtomicBuffer, typeSize * numDestItems);
        memcpy(&refValues[0], svmDataBuffer,
               typeSize * threadCount * NumNonAtomicVariablesPerThread());
    }
    else
    {
        if (deviceThreadCount > 0)
        {
            error = clEnqueueReadBuffer(queue, streams[0], CL_TRUE, 0,
                                        typeSize * numDestItems, &destItems[0],
                                        0, NULL, NULL);
            test_error(error, "Unable to read result value!");
            error = clEnqueueReadBuffer(queue, streams[1], CL_TRUE, 0,
                                        typeSize * deviceThreadCount
                                            * NumNonAtomicVariablesPerThread(),
                                        &refValues[0], 0, NULL, NULL);
            test_error(error, "Unable to read reference values!");
        }
    }
    bool dataVerified = false;
    // If we have an expectedFn, then we need to generate a final value to
    // compare against. If we don't have one, it's because we're comparing ref
    // values only
    for (cl_uint i = 0; i < numDestItems; i++)
    {
        HostDataType expected;

        if (!ExpectedValue(expected, threadCount,
                           startRefValues.size() ? &startRefValues[0] : 0, i))
            break; // no expected value function provided

        if (expected != destItems[i])
        {
            std::stringstream logLine;
            logLine << "ERROR: Result " << i
                    << " from kernel does not validate! (should be " << expected
                    << ", was " << destItems[i] << ")\n";
            log_error("%s", logLine.str().c_str());
            for (i = 0; i < threadCount; i++)
            {
                logLine.str("");
                logLine << " --- " << i << " - ";
                if (startRefValues.size())
                    logLine << startRefValues[i] << " -> " << refValues[i];
                else
                    logLine << refValues[i];
                logLine << " --- ";
                if (i < numDestItems) logLine << destItems[i];
                logLine << "\n";
                log_info("%s", logLine.str().c_str());
            }
            if (!gDebug)
            {
                log_info("Program source:\n");
                log_info("%s\n", programLine);
            }
            return -1;
        }
        dataVerified = true;
    }

    bool dataCorrect = false;
    /* Use the verify function (if provided) to also check the results */
    if (VerifyRefs(dataCorrect, threadCount, &refValues[0], &destItems[0]))
    {
        if (!dataCorrect)
        {
            log_error("ERROR: Reference values did not validate!\n");
            std::stringstream logLine;
            for (cl_uint i = 0; i < threadCount; i++)
                for (cl_uint j = 0; j < NumNonAtomicVariablesPerThread(); j++)
                {
                    logLine.str("");
                    logLine
                        << " --- " << i << " - "
                        << refValues[i * NumNonAtomicVariablesPerThread() + j]
                        << " --- ";
                    if (j == 0 && i < numDestItems) logLine << destItems[i];
                    logLine << "\n";
                    log_info("%s", logLine.str().c_str());
                }
            if (!gDebug)
            {
                log_info("Program source:\n");
                log_info("%s\n", programLine);
            }
            return -1;
        }
    }
    else if (!dataVerified)
    {
        log_error("ERROR: Test doesn't check total or refs; no values are "
                  "verified!\n");
        return -1;
    }

    if (OldValueCheck()
        && !(DeclaredInProgram()
             && !LocalMemory())) // don't test for programs scope global atomics
                                 // 'old' value has been overwritten by previous
                                 // clEnqueueNDRangeKernel
    {
        /* Re-write the starting value */
        for (size_t i = 0; i < numDestItems; i++) destItems[i] = _startValue;
        refValues[0] = 0;
        if (deviceThreadCount > 0)
        {
            error = clEnqueueWriteBuffer(queue, streams[0], CL_TRUE, 0,
                                         typeSize * numDestItems, &destItems[0],
                                         0, NULL, NULL);
            test_error(error, "Unable to write starting values!");

            /* Run the kernel once for a single thread, so we can verify that
             * the returned value is the original one */
            threadNum[0] = 1;
            error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, threadNum,
                                           threadNum, 0, NULL, NULL);
            test_error(error, "Unable to execute test kernel");

            error = clEnqueueReadBuffer(queue, streams[1], CL_TRUE, 0, typeSize,
                                        &refValues[0], 0, NULL, NULL);
            test_error(error, "Unable to read reference values!");
        }
        else
        {
            /* Start host thread */
            HostFunction(0, 1, &destItems[0], &refValues[0]);
        }

        if (refValues[0] != _startValue) // destItems[0])
        {
            std::stringstream logLine;
            logLine << "ERROR: atomic function operated correctly but did NOT "
                       "return correct 'old' value "
                       " (should have been "
                    << destItems[0] << ", returned " << refValues[0] << ")!\n";
            log_error("%s", logLine.str().c_str());
            if (!gDebug)
            {
                log_info("Program source:\n");
                log_info("%s\n", programLine);
            }
            return -1;
        }
    }
    if (UseSVM())
    {
        // the buffer object must first be released before the SVM buffer is
        // freed. The Wrapper Class method reset() will do that
        streams[0].reset();
        if (gUseHostPtr)
            free(svmAtomicBuffer);
        else
            clSVMFree(context, svmAtomicBuffer);
        streams[1].reset();
        if (gUseHostPtr)
            free(svmDataBuffer);
        else
            clSVMFree(context, svmDataBuffer);
    }
    _passCount++;
    return 0;
}

#endif //_COMMON_H_
