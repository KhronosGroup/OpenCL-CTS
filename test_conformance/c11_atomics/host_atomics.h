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
#ifndef HOST_ATOMICS_H_
#define HOST_ATOMICS_H_

#include "harness/testHarness.h"
#include <mutex>
#include "CL/cl_half.h"

#ifdef WIN32
#include "Windows.h"
#endif

extern cl_half_rounding_mode gHalfRoundingMode;

//flag for test verification (good test should discover non-atomic functions and fail)
//#define NON_ATOMIC_FUNCTIONS

enum TExplicitMemoryOrderType
{
  MEMORY_ORDER_EMPTY,
  MEMORY_ORDER_RELAXED,
  MEMORY_ORDER_ACQUIRE,
  MEMORY_ORDER_RELEASE,
  MEMORY_ORDER_ACQ_REL,
  MEMORY_ORDER_SEQ_CST
};

// Wrapper class for half-precision
class HostHalf {
public:
    // Convert from semantic values
    HostHalf(cl_uint value = 0)
        : value(
            cl_half_from_float(static_cast<float>(value), gHalfRoundingMode))
    {}
    HostHalf(int value): HostHalf(static_cast<cl_uint>(value)) {}
    HostHalf(float value): value(cl_half_from_float(value, gHalfRoundingMode))
    {}
    HostHalf(double value): HostHalf(static_cast<float>(value)) {}

    // Convert to semantic values
    operator cl_uint() const
    {
        return static_cast<cl_uint>(cl_half_to_float(value));
    }
    operator float() const { return cl_half_to_float(value); }
    operator double() const
    {
        return static_cast<double>(cl_half_to_float(value));
    }

    // Construct from bit representation
    HostHalf(cl_half value): value(value) {}

    // Get the underlying bit representation
    operator cl_half() const { return value; }

    HostHalf operator-() const
    {
        return HostHalf(
            cl_half_from_float(-cl_half_to_float(value), gHalfRoundingMode));
    }

#define GENERIC_OP(RetType, op)                                                \
    RetType operator op(const HostHalf &other) const                           \
    {                                                                          \
        return RetType(cl_half_to_float(value)                                 \
                           op cl_half_to_float(other.value));                  \
    }

    GENERIC_OP(bool, ==)
    GENERIC_OP(bool, !=)
    GENERIC_OP(bool, <)
    GENERIC_OP(bool, <=)
    GENERIC_OP(bool, >)
    GENERIC_OP(bool, >=)
    GENERIC_OP(HostHalf, +)
    GENERIC_OP(HostHalf, -)
    GENERIC_OP(HostHalf, *)
    GENERIC_OP(HostHalf, /)
#undef GENERIC_OP

#define INPLACE_OP(op)                                                         \
    HostHalf &operator op##=(const HostHalf &other)                            \
    {                                                                          \
        value = cl_half_from_float(cl_half_to_float(value)                     \
                                       op cl_half_to_float(other.value),       \
                                   gHalfRoundingMode);                         \
        return *this;                                                          \
    }
    INPLACE_OP(+)
    INPLACE_OP(-)
    INPLACE_OP(*)
    INPLACE_OP(/)
#undef INPLACE_OP

    friend std::ostream &operator<<(std::ostream &os, const HostHalf &hh)
    {
        float f = cl_half_to_float(hh.value);
        os << f;
        return os;
    }

private:
    cl_half value;
};

namespace std {
inline HostHalf abs(const HostHalf &value)
{
    return value < HostHalf(0) ? -value : value;
}
} // namespace std

// host atomic types (applicable for atomic functions supported on host OS)
#ifdef WIN32
#define HOST_ATOMIC_INT         unsigned long
#define HOST_ATOMIC_UINT        unsigned long
#define HOST_ATOMIC_LONG        unsigned long long
#define HOST_ATOMIC_ULONG       unsigned long long
#define HOST_ATOMIC_HALF unsigned short
#define HOST_ATOMIC_FLOAT       float
#define HOST_ATOMIC_DOUBLE      double
#else
#define HOST_ATOMIC_INT         cl_int
#define HOST_ATOMIC_UINT        cl_uint
#define HOST_ATOMIC_LONG        cl_long
#define HOST_ATOMIC_ULONG       cl_ulong
#define HOST_ATOMIC_HALF cl_half
#define HOST_ATOMIC_FLOAT       cl_float
#define HOST_ATOMIC_DOUBLE      cl_double
#endif

#define HOST_ATOMIC_INTPTR_T32  HOST_ATOMIC_INT
#define HOST_ATOMIC_UINTPTR_T32 HOST_ATOMIC_UINT
#define HOST_ATOMIC_SIZE_T32    HOST_ATOMIC_UINT
#define HOST_ATOMIC_PTRDIFF_T32 HOST_ATOMIC_INT

#define HOST_ATOMIC_INTPTR_T64  HOST_ATOMIC_LONG
#define HOST_ATOMIC_UINTPTR_T64 HOST_ATOMIC_ULONG
#define HOST_ATOMIC_SIZE_T64    HOST_ATOMIC_ULONG
#define HOST_ATOMIC_PTRDIFF_T64 HOST_ATOMIC_LONG

#define HOST_ATOMIC_FLAG        HOST_ATOMIC_INT

// host regular types corresponding to atomic types
#define HOST_INT                cl_int
#define HOST_UINT               cl_uint
#define HOST_LONG               cl_long
#define HOST_ULONG              cl_ulong
#define HOST_HALF HostHalf
#define HOST_FLOAT              cl_float
#define HOST_DOUBLE             cl_double

#define HOST_INTPTR_T32         cl_int
#define HOST_UINTPTR_T32        cl_uint
#define HOST_SIZE_T32           cl_uint
#define HOST_PTRDIFF_T32        cl_int

#define HOST_INTPTR_T64         cl_long
#define HOST_UINTPTR_T64        cl_ulong
#define HOST_SIZE_T64           cl_ulong
#define HOST_PTRDIFF_T64        cl_long

#define HOST_FLAG cl_int

extern cl_half_rounding_mode gHalfRoundingMode;

template <typename HostAtomicType>
constexpr bool is_host_atomic_fp_v =
    std::disjunction_v<std::is_same<HostAtomicType, HOST_ATOMIC_HALF>,
                       std::is_same<HostAtomicType, HOST_ATOMIC_FLOAT>,
                       std::is_same<HostAtomicType, HOST_ATOMIC_DOUBLE>>;

template <typename HostDataType>
constexpr bool is_host_fp_v =
    std::disjunction_v<std::is_same<HostDataType, HOST_HALF>,
                       std::is_same<HostDataType, HOST_FLOAT>,
                       std::is_same<HostDataType, HOST_DOUBLE>>;

// host atomic functions
void host_atomic_thread_fence(TExplicitMemoryOrderType order);

template <typename AtomicType, typename CorrespondingType>
CorrespondingType host_atomic_fetch_add(volatile AtomicType *a, CorrespondingType c,
                                        TExplicitMemoryOrderType order)
{
    if constexpr (is_host_atomic_fp_v<AtomicType>)
    {
        static std::mutex mx;
        std::lock_guard<std::mutex> lock(mx);
        CorrespondingType old_value = *a;
        CorrespondingType new_value = old_value + c;
        *a = static_cast<AtomicType>(new_value);
        return old_value;
    }
    else
    {
#if defined(_MSC_VER) || (defined(__INTEL_COMPILER) && defined(WIN32))
        return InterlockedExchangeAdd(a, c);
#elif defined(__GNUC__)
        return __sync_fetch_and_add(a, c);
#else
        log_info("Host function not implemented: atomic_fetch_add\n");
        return 0;
#endif
    }
}

template <typename AtomicType, typename CorrespondingType>
CorrespondingType host_atomic_fetch_sub(volatile AtomicType *a, CorrespondingType c,
                                        TExplicitMemoryOrderType order)
{
    if constexpr (is_host_atomic_fp_v<AtomicType>)
    {
        static std::mutex mx;
        std::lock_guard<std::mutex> lock(mx);
        CorrespondingType old_value = *a;
        CorrespondingType new_value = old_value - c;
        *a = static_cast<AtomicType>(new_value);
        return old_value;
    }
    else
    {
#if defined(_MSC_VER) || (defined(__INTEL_COMPILER) && defined(WIN32))
        return InterlockedExchangeSubtract(a, c);
#elif defined(__GNUC__)
        return __sync_fetch_and_sub(a, c);
#else
        log_info("Host function not implemented: atomic_fetch_sub\n");
        return 0;
#endif
    }
}

template <typename AtomicType, typename CorrespondingType>
CorrespondingType host_atomic_exchange(volatile AtomicType *a, CorrespondingType c,
                                       TExplicitMemoryOrderType order)
{
#if defined( _MSC_VER ) || (defined( __INTEL_COMPILER ) && defined(WIN32))
    if constexpr (sizeof(CorrespondingType) == 2)
        return InterlockedExchange16(reinterpret_cast<volatile SHORT *>(a),
                                     *reinterpret_cast<SHORT *>(&c));
    else
        return InterlockedExchange(reinterpret_cast<volatile LONG *>(a),
                                   *reinterpret_cast<LONG *>(&c));
#elif defined(__GNUC__)
    return __sync_lock_test_and_set(a, *reinterpret_cast<AtomicType *>(&c));
#else
  log_info("Host function not implemented: atomic_exchange\n");
  return 0;
#endif
}
template <> HOST_FLOAT host_atomic_exchange(volatile HOST_ATOMIC_FLOAT *a, HOST_FLOAT c,
                                            TExplicitMemoryOrderType order);
template <> HOST_DOUBLE host_atomic_exchange(volatile HOST_ATOMIC_DOUBLE *a, HOST_DOUBLE c,
                                             TExplicitMemoryOrderType order);

template <typename AtomicType, typename CorrespondingType>
bool host_atomic_compare_exchange(volatile AtomicType *a, CorrespondingType *expected, CorrespondingType desired,
                                  TExplicitMemoryOrderType order_success,
                                  TExplicitMemoryOrderType order_failure)
{
    CorrespondingType tmp;
    if constexpr (is_host_atomic_fp_v<AtomicType>)
    {
        static std::mutex mtx;
        std::lock_guard<std::mutex> lock(mtx);
        tmp = static_cast<CorrespondingType>(*a);
        if (tmp == *expected)
        {
            *a = static_cast<AtomicType>(desired);
            return true;
        }
        *expected = tmp;
    }
    else
    {
#if defined(_MSC_VER) || (defined(__INTEL_COMPILER) && defined(WIN32))
        tmp = InterlockedCompareExchange(a, desired, *expected);
#elif defined(__GNUC__)
        tmp = __sync_val_compare_and_swap(a, *expected, desired);
#else
        log_info("Host function not implemented: atomic_compare_exchange\n");
        tmp = 0;
#endif
        if (tmp == *expected) return true;
        *expected = tmp;
    }
    return false;
}

template <typename AtomicType, typename CorrespondingType>
CorrespondingType host_atomic_load(volatile AtomicType *a,
                                   TExplicitMemoryOrderType order)
{
#if defined( _MSC_VER ) || (defined( __INTEL_COMPILER ) && defined(WIN32))
    if constexpr (sizeof(CorrespondingType) == 2)
        return InterlockedOr16(reinterpret_cast<volatile SHORT *>(a), 0);
    else
        return InterlockedExchangeAdd(reinterpret_cast<volatile LONG *>(a), 0);
#elif defined(__GNUC__)
  return __sync_add_and_fetch(a, 0);
#else
  log_info("Host function not implemented: atomic_load\n");
  return 0;
#endif
}
template <> HOST_FLOAT host_atomic_load(volatile HOST_ATOMIC_FLOAT *a,
                                        TExplicitMemoryOrderType order);
template <> HOST_DOUBLE host_atomic_load(volatile HOST_ATOMIC_DOUBLE *a,
                                         TExplicitMemoryOrderType order);

template <typename AtomicType, typename CorrespondingType>
void host_atomic_store(volatile AtomicType* a, CorrespondingType c,
                       TExplicitMemoryOrderType order)
{
  host_atomic_exchange(a, c, order);
}

template <typename AtomicType, typename CorrespondingType>
void host_atomic_init(volatile AtomicType* a, CorrespondingType c)
{
  host_atomic_exchange(a, c, MEMORY_ORDER_RELAXED);
}

template <typename AtomicType, typename CorrespondingType>
CorrespondingType host_atomic_fetch_or(volatile AtomicType *a, CorrespondingType c,
                                       TExplicitMemoryOrderType order)
{
  CorrespondingType expected = host_atomic_load<AtomicType, CorrespondingType>(a, order);
  CorrespondingType desired;
  do
  desired = expected | c;
  while(!host_atomic_compare_exchange(a, &expected, desired, order, order));
  return expected;
}

template <typename AtomicType, typename CorrespondingType>
CorrespondingType host_atomic_fetch_and(volatile AtomicType *a, CorrespondingType c,
                                        TExplicitMemoryOrderType order)
{
  CorrespondingType expected = host_atomic_load<AtomicType, CorrespondingType>(a, order);
  CorrespondingType desired;
  do
  desired = expected & c;
  while(!host_atomic_compare_exchange(a, &expected, desired, order, order));
  return expected;
}

template <typename AtomicType, typename CorrespondingType>
CorrespondingType host_atomic_fetch_xor(volatile AtomicType *a, CorrespondingType c,
                                        TExplicitMemoryOrderType order)
{
  CorrespondingType expected = host_atomic_load<AtomicType, CorrespondingType>(a, order);
  CorrespondingType desired;
  do
  desired = expected ^ c;
  while(!host_atomic_compare_exchange(a, &expected, desired, order, order));
  return expected;
}

template <typename AtomicType, typename CorrespondingType>
CorrespondingType host_atomic_fetch_min(volatile AtomicType *a, CorrespondingType c,
                                        TExplicitMemoryOrderType order)
{
  CorrespondingType expected = host_atomic_load<AtomicType, CorrespondingType>(a, order);
  CorrespondingType desired;
  do
  desired = expected < c ? expected : c;
  while(!host_atomic_compare_exchange(a, &expected, desired, order, order));
  return expected;
}

template <typename AtomicType, typename CorrespondingType>
CorrespondingType host_atomic_fetch_max(volatile AtomicType *a, CorrespondingType c,
                                        TExplicitMemoryOrderType order)
{
  CorrespondingType expected = host_atomic_load<AtomicType, CorrespondingType>(a, order);
  CorrespondingType desired;
  do
  desired = expected > c ? expected : c;
  while(!host_atomic_compare_exchange(a, &expected, desired, order, order));
  return expected;
}

bool host_atomic_flag_test_and_set(volatile HOST_ATOMIC_FLAG *a, TExplicitMemoryOrderType order);
void host_atomic_flag_clear(volatile HOST_ATOMIC_FLAG *a, TExplicitMemoryOrderType order);

#endif // HOST_ATOMICS_H_
