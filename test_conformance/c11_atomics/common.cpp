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
#include "harness/testHarness.h"
#include "harness/kernelHelpers.h"
#include "harness/typeWrappers.h"

#include "common.h"

const char *get_memory_order_type_name(TExplicitMemoryOrderType orderType)
{
  switch (orderType)
  {
  case MEMORY_ORDER_EMPTY:
    return "";
  case MEMORY_ORDER_RELAXED:
    return "memory_order_relaxed";
  case MEMORY_ORDER_ACQUIRE:
    return "memory_order_acquire";
  case MEMORY_ORDER_RELEASE:
    return "memory_order_release";
  case MEMORY_ORDER_ACQ_REL:
    return "memory_order_acq_rel";
  case MEMORY_ORDER_SEQ_CST:
    return "memory_order_seq_cst";
  default:
    return 0;
  }
}

const char *get_memory_scope_type_name(TExplicitMemoryScopeType scopeType)
{
  switch (scopeType)
  {
  case MEMORY_SCOPE_EMPTY:
    return "";
  case MEMORY_SCOPE_WORK_GROUP:
    return "memory_scope_work_group";
  case MEMORY_SCOPE_DEVICE:
    return "memory_scope_device";
  case MEMORY_SCOPE_ALL_SVM_DEVICES:
    return "memory_scope_all_svm_devices";
  default:
    return 0;
  }
}


cl_uint AtomicTypeInfo::Size(cl_device_id device)
{
  switch(_type)
  {
  case TYPE_ATOMIC_INT:
  case TYPE_ATOMIC_UINT:
  case TYPE_ATOMIC_FLOAT:
  case TYPE_ATOMIC_FLAG:
    return sizeof(cl_int);
  case TYPE_ATOMIC_LONG:
  case TYPE_ATOMIC_ULONG:
  case TYPE_ATOMIC_DOUBLE:
    return sizeof(cl_long);
  case TYPE_ATOMIC_INTPTR_T:
  case TYPE_ATOMIC_UINTPTR_T:
  case TYPE_ATOMIC_SIZE_T:
  case TYPE_ATOMIC_PTRDIFF_T:
    {
      int error;
      cl_uint addressBits = 0;

      error = clGetDeviceInfo(device, CL_DEVICE_ADDRESS_BITS, sizeof(addressBits), &addressBits, 0);
      test_error_ret(error, "clGetDeviceInfo", 0);

      return addressBits/8;
    }
  default:
    return 0;
  }
}

const char *AtomicTypeInfo::AtomicTypeName()
{
  switch(_type)
  {
  case TYPE_ATOMIC_INT:
    return "atomic_int";
  case TYPE_ATOMIC_UINT:
    return "atomic_uint";
  case TYPE_ATOMIC_FLOAT:
    return "atomic_float";
  case TYPE_ATOMIC_FLAG:
    return "atomic_flag";
  case TYPE_ATOMIC_LONG:
    return "atomic_long";
  case TYPE_ATOMIC_ULONG:
    return "atomic_ulong";
  case TYPE_ATOMIC_DOUBLE:
    return "atomic_double";
  case TYPE_ATOMIC_INTPTR_T:
    return "atomic_intptr_t";
  case TYPE_ATOMIC_UINTPTR_T:
    return "atomic_uintptr_t";
  case TYPE_ATOMIC_SIZE_T:
    return "atomic_size_t";
  case TYPE_ATOMIC_PTRDIFF_T:
    return "atomic_ptrdiff_t";
  default:
    return 0;
  }
}

const char *AtomicTypeInfo::RegularTypeName()
{
  switch(_type)
  {
  case TYPE_ATOMIC_INT:
    return "int";
  case TYPE_ATOMIC_UINT:
    return "uint";
  case TYPE_ATOMIC_FLOAT:
    return "float";
  case TYPE_ATOMIC_FLAG:
    return "int";
  case TYPE_ATOMIC_LONG:
    return "long";
  case TYPE_ATOMIC_ULONG:
    return "ulong";
  case TYPE_ATOMIC_DOUBLE:
    return "double";
  case TYPE_ATOMIC_INTPTR_T:
    return "intptr_t";
  case TYPE_ATOMIC_UINTPTR_T:
    return "uintptr_t";
  case TYPE_ATOMIC_SIZE_T:
    return "size_t";
  case TYPE_ATOMIC_PTRDIFF_T:
    return "ptrdiff_t";
  default:
    return 0;
  }
}

const char *AtomicTypeInfo::AddSubOperandTypeName()
{
  switch(_type)
  {
  case TYPE_ATOMIC_INTPTR_T:
  case TYPE_ATOMIC_UINTPTR_T:
    return AtomicTypeInfo(TYPE_ATOMIC_PTRDIFF_T).RegularTypeName();
  default:
    return RegularTypeName();
  }
}

int AtomicTypeInfo::IsSupported(cl_device_id device)
{
  switch(_type)
  {
  case TYPE_ATOMIC_INT:
  case TYPE_ATOMIC_UINT:
  case TYPE_ATOMIC_FLOAT:
  case TYPE_ATOMIC_FLAG:
    return 1;
  case TYPE_ATOMIC_LONG:
  case TYPE_ATOMIC_ULONG:
    return is_extension_available(device, "cl_khr_int64_base_atomics") &&
      is_extension_available(device, "cl_khr_int64_extended_atomics");
  case TYPE_ATOMIC_DOUBLE:
    return is_extension_available(device, "cl_khr_int64_base_atomics") &&
      is_extension_available(device, "cl_khr_int64_extended_atomics") &&
      is_extension_available(device, "cl_khr_fp64");
  case TYPE_ATOMIC_INTPTR_T:
  case TYPE_ATOMIC_UINTPTR_T:
  case TYPE_ATOMIC_SIZE_T:
  case TYPE_ATOMIC_PTRDIFF_T:
    if(Size(device) == 4)
      return 1;
    return is_extension_available(device, "cl_khr_int64_base_atomics") &&
      is_extension_available(device, "cl_khr_int64_extended_atomics");
  default:
    return 0;
  }
}

template<> cl_int AtomicTypeExtendedInfo<cl_int>::MinValue() {return CL_INT_MIN;}
template<> cl_uint AtomicTypeExtendedInfo<cl_uint>::MinValue() {return 0;}
template<> cl_long AtomicTypeExtendedInfo<cl_long>::MinValue() {return CL_LONG_MIN;}
template<> cl_ulong AtomicTypeExtendedInfo<cl_ulong>::MinValue() {return 0;}
template<> cl_float AtomicTypeExtendedInfo<cl_float>::MinValue() {return CL_FLT_MIN;}
template<> cl_double AtomicTypeExtendedInfo<cl_double>::MinValue() {return CL_DBL_MIN;}

template<> cl_int AtomicTypeExtendedInfo<cl_int>::MaxValue() {return CL_INT_MAX;}
template<> cl_uint AtomicTypeExtendedInfo<cl_uint>::MaxValue() {return CL_UINT_MAX;}
template<> cl_long AtomicTypeExtendedInfo<cl_long>::MaxValue() {return CL_LONG_MAX;}
template<> cl_ulong AtomicTypeExtendedInfo<cl_ulong>::MaxValue() {return CL_ULONG_MAX;}
template<> cl_float AtomicTypeExtendedInfo<cl_float>::MaxValue() {return CL_FLT_MAX;}
template<> cl_double AtomicTypeExtendedInfo<cl_double>::MaxValue() {return CL_DBL_MAX;}
