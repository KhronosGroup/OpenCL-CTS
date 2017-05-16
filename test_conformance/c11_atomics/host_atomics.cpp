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
#include "host_atomics.h"

void host_atomic_thread_fence(TExplicitMemoryOrderType order)
{
  if(order != MEMORY_ORDER_RELAXED) {
#if defined( _MSC_VER ) || (defined( __INTEL_COMPILER ) && defined(WIN32))
    MemoryBarrier();
#elif defined(__GNUC__)
    __sync_synchronize();
#else
    log_info("Host function not implemented: host_atomic_thread_fence\n");
#endif
  }
}

template <>
HOST_FLOAT host_atomic_exchange(volatile HOST_ATOMIC_FLOAT* a, HOST_FLOAT c, TExplicitMemoryOrderType order)
{
  HOST_UINT tmp = host_atomic_exchange((volatile HOST_ATOMIC_UINT*)a, *(HOST_UINT*)&c, order);
  return *(float*)&tmp;
}
template <>
HOST_DOUBLE host_atomic_exchange(volatile HOST_ATOMIC_DOUBLE* a, HOST_DOUBLE c, TExplicitMemoryOrderType order)
{
  HOST_ULONG tmp = host_atomic_exchange((volatile HOST_ATOMIC_ULONG*)a, *(HOST_ULONG*)&c, order);
  return *(double*)&tmp;
}

template <>
HOST_FLOAT host_atomic_load(volatile HOST_ATOMIC_FLOAT* a, TExplicitMemoryOrderType order)
{
  HOST_UINT tmp = host_atomic_load<HOST_ATOMIC_UINT, HOST_UINT>((volatile HOST_ATOMIC_UINT*)a, order);
  return *(float*)&tmp;
}
template <>
HOST_DOUBLE host_atomic_load(volatile HOST_ATOMIC_DOUBLE* a, TExplicitMemoryOrderType order)
{
  HOST_ULONG tmp = host_atomic_load<HOST_ATOMIC_ULONG, HOST_ULONG>((volatile HOST_ATOMIC_ULONG*)a, order);
  return *(double*)&tmp;
}

bool host_atomic_flag_test_and_set(volatile HOST_ATOMIC_FLAG *a, TExplicitMemoryOrderType order)
{
  HOST_FLAG old = host_atomic_exchange(a, 1, order);
  return old != 0;
}

void host_atomic_flag_clear(volatile HOST_ATOMIC_FLAG *a, TExplicitMemoryOrderType order)
{
  host_atomic_store(a, 0, order);
}
