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
#ifndef __COMMON_H__
#define __COMMON_H__

#include "../../test_common/harness/compat.h"
#include "../../test_common/harness/testHarness.h"
#include "../../test_common/harness/errorHelpers.h"
#include "../../test_common/harness/kernelHelpers.h"
#include "../../test_common/harness/typeWrappers.h"

#if (defined(_WIN32) || defined(_WIN64)) && defined(_MSC_VER)
    #include <windows.h>
#endif

typedef enum {
    memory_order_relaxed,
    memory_order_acquire,
    memory_order_release,
    memory_order_acq_rel,
    memory_order_seq_cst
} cl_memory_order;

cl_int AtomicLoadExplicit(volatile cl_int * pValue, cl_memory_order order);
cl_int AtomicFetchAddExplicit(volatile cl_int *object, cl_int operand, cl_memory_order o);

template <typename T>
bool AtomicCompareExchangeStrongExplicit(volatile T *a, T *expected, T desired,
                                  cl_memory_order order_success,
                                  cl_memory_order order_failure)
{
    T tmp;
#if defined( _MSC_VER ) || (defined( __INTEL_COMPILER ) && defined(WIN32))
    tmp = (sizeof(void*) == 8) ? (T)InterlockedCompareExchange64((volatile LONG64 *)a, (LONG64)desired, *(LONG64 *)expected) :
      (T)InterlockedCompareExchange((volatile LONG*)a, (LONG)desired, *(LONG*)expected);
#elif defined(__GNUC__)
    tmp = (T)__sync_val_compare_and_swap((volatile intptr_t*)a, (intptr_t)(*expected), (intptr_t)desired);
#else
    log_info("Host function not implemented: atomic_compare_exchange\n");
    tmp = 0;
#endif
    if(tmp == *expected)
        return true;
    *expected = tmp;
    return false;
}

// this checks for a NULL ptr and/or an error code
#define test_error2(error_code, ptr, msg)  { if(error != 0)  { test_error(error_code, msg); } else  { if(NULL == ptr)  {print_null_error(msg); return -1;} } }
#define print_null_error(msg) log_error("ERROR: %s! (NULL pointer detected %s:%d)\n", msg, __FILE__, __LINE__ );

// max possible number of queues needed, 1 for each device in platform.
#define MAXQ 32

typedef struct Node{
    cl_int global_id;
    cl_int position_in_list;
    struct Node* pNext;
} Node;

extern void   create_linked_lists(Node* pNodes, size_t num_lists, int list_length);
extern cl_int verify_linked_lists(Node* pNodes, size_t num_lists, int list_length);

extern cl_int        create_linked_lists_on_device(int qi, cl_command_queue q, cl_mem allocator,     cl_kernel k, size_t numLists  );
extern cl_int        verify_linked_lists_on_device(int qi, cl_command_queue q, cl_mem num_correct,   cl_kernel k, cl_int ListLength, size_t numLists  );
extern cl_int create_linked_lists_on_device_no_map(int qi, cl_command_queue q, size_t *pAllocator,   cl_kernel k, size_t numLists  );
extern cl_int verify_linked_lists_on_device_no_map(int qi, cl_command_queue q, cl_int *pNum_correct, cl_kernel k, cl_int ListLength, size_t numLists  );

extern int    test_svm_byte_granularity(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int    test_svm_set_kernel_exec_info_svm_ptrs(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int    test_svm_fine_grain_memory_consistency(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int    test_svm_fine_grain_sync_buffers(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int    test_svm_shared_address_space_coarse_grain_old_api(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int    test_svm_shared_address_space_coarse_grain_new_api(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int    test_svm_shared_address_space_fine_grain_buffers(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int    test_svm_shared_address_space_fine_grain(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int    test_svm_cross_buffer_pointers_coarse_grain(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int    test_svm_pointer_passing(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int    test_svm_allocate_shared_buffer(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int    test_svm_shared_sub_buffers(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int    test_svm_enqueue_api(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int    test_svm_migrate(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);

extern cl_int create_cl_objects(cl_device_id device_from_harness, const char** ppCodeString, cl_context* context, cl_program *program, cl_command_queue *queues, cl_uint *num_devices, cl_device_svm_capabilities required_svm_caps);

extern const char *linked_list_create_and_verify_kernels[];

#endif    // #ifndef __COMMON_H__

