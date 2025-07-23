# cl_khr_unified_svm

This document describe the test plan for the [`cl_khr_unified_svm`](https://github.com/KhronosGroup/OpenCL-Docs/pull/1282) extension, which may currently be found here:

https://github.com/KhronosGroup/OpenCL-Docs/pull/1282

## Prior Work

Existing OpenCL 2.0 SVM CTS tests may be found here:

https://github.com/KhronosGroup/OpenCL-CTS/tree/main/test_conformance/SVM

Some prior tests for the Intel USM extension [`cl_intel_unified_shared_memory`](https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_unified_shared_memory.html) may be found here:

https://github.com/intel/compute-samples/tree/master/compute_samples/tests/test_cl_unified_shared_memory

## Test Plan

### Consistency Check

As an initial test, perform a consistency check to ensure that the platform and the test device enumerate a consistent set of SVM capabilities:

* [X] For each device in the platform, check that the platform and device report the same number of SVM capability combinations.
* [X] For each SVM capability combination reported by the platform, check that the reported platform capabilities at an index are the intersection of all non-zero device capabilities at the same index.
* [X] For each SVM capability combination reported by the test device, check that the device SVM capabilities are either a super-set of the platform SVM capabilities or are zero, indicating that this SVM type is not supported.

NOTE: Added by https://github.com/KhronosGroup/OpenCL-CTS/pull/2174.

### Testing SVM Capabilities

* [X] `CL_SVM_CAPABILITY_SINGLE_ADDRESS_SPACE_KHR`
    * Testing options:
        1. Pass a pointer-to-a-pointer as a kernel argument.
           Read the pointer from the kernel argument and write to it.
           Ensure the correct value was written on the host.
        2. Pass the pointer as a kernel argument.
           Write the kernel argument to another allocation, which could even be an OpenCL buffer memory object.
           Ensure the value written on the device matches the value on the host.
* [X] `CL_SVM_CAPABILITY_SYSTEM_ALLOCATED_KHR`
    * [X] When allocating memory to test, use the system `malloc` rather than `clSVMAllocWithPropertiesKHR`.
* [ ] `CL_SVM_CAPABILITY_DEVICE_OWNED_KHR`
    * TBD
* [X] `CL_SVM_CAPABILITY_DEVICE_UNASSOCIATED_KHR`
    * [X] When allocating memory to test, do not pass the test device via a `CL_SVM_ALLOC_ASSOCIATED_DEVICE_HANDLE_KHR` property.
    * [X] Include at least one targeted test that passes the `CL_SVM_ALLOC_ASSOCIATED_DEVICE_HANDLE_KHR` property anyhow.
* [ ] `CL_SVM_CAPABILITY_CONTEXT_ACCESS_KHR`
    * TBD: Create a multi-device context, use the allocation on all of the devices in the context?
* [ ] `CL_SVM_CAPABILITY_HOST_OWNED_KHR`
    * TBD
* [X] `CL_SVM_CAPABILITY_HOST_READ_KHR`
    * [X] When verifying test results, read from the allocation directly on the host, without mapping or copying explicitly.
    * [X] For devices that also support `CL_SVM_CAPABILITY_DEVICE_WRITE_KHR`, also include a targeted test that writes on the device and reads the results on the host without mapping or copying explicitly.
* [X] `CL_SVM_CAPABILITY_HOST_WRITE_KHR`
    * [X] When initializing test data, write to the allocation directly from the host, without mapping or copying explicitly.
    * [X] For devices that also support `CL_SVM_CAPABILITY_DEVICE_READ_KHR`, also include a targeted test that writes on the host without mapping or copying explicitly, then read the results on the device and writes it to an an OpenCL buffer memory object.
* [X] `CL_SVM_CAPABILITY_HOST_MAP_KHR`
    * [X] When initializing test data or verifying test results, map the allocation for access from the host, rather than copying explicitly.
    * [X] For devices that also support `CL_SVM_CAPABILITY_DEVICE_WRITE_KHR`, also include a targeted test that writes on the device and reads the results on the host by mapping.
    * [X] For devices that also support `CL_SVM_CAPABILITY_DEVICE_READ_KHR`, also include a targeted test that writes on the host by mapping, then reads the results on the device and writes it to an OpenCL buffer memory object.
* [X] `CL_SVM_CAPABILITY_DEVICE_READ_KHR`
    * [X] Populate an allocation via direct access from the host, via mapping, or via device memcpy, depending on supported capabilities.
          Then, read the value on the device and write it to an OpenCL buffer memory object.
    * Mechanisms to read from the allocation on the device are:
        * [X] Via a kernel that reads from the allocation as a kernel argument.
        * [X] Via `clEnqueueSVMMemcpy`.
* [X] `CL_SVM_CAPABILITY_DEVICE_WRITE_KHR`
    * [X] Populate an OpenCL buffer memory object with values.
          Read the values from the OpenCL buffer memory object on the device and write them to the memory allocation.
          Verify that the values were written correctly via direct access from the host, via mapping, or via memcpy, depending on supported capabilities.
    * Mechanisms to write to the allocation on the device are:
       * [X] Via a kernel that writes to the allocation as a kernel argument.
       * [X] Via `clEnqueueSVMMemcpy`.
       * [X] Via `clEnqueueSVMMemFill`.
* [X] `CL_SVM_CAPABILITY_DEVICE_ATOMIC_ACCESS_KHR`
    * [X] Initialize a memory allocation with zero.
          Atomically increment the memory allocation from the device.
          Verify that the correct updates were made via direct access from the host, via mapping, or via memcpy, depending on supported capabilities.
* [ ] `CL_SVM_CAPABILITY_CONCURRENT_ACCESS_KHR`
    * Note, as described, these tests will require `CL_SVM_CAPABILITY_HOST_READ_KHR`, `CL_SVM_CAPABILITY_HOST_WRITE_KHR`, `CL_SVM_CAPABILITY_DEVICE_READ_KHR`, and `CL_SVM_CAPABILITY_DEVICE_WRITE_KHR` also.
    * Details TBD, but the rough idea will be:
        1. Allocate a small amount of memory, some which will be accessed via the host (or eventually, another device?), and some which will be accessed via the device.
        2. From the host, perform some number of read-modify-write accesses to the memory.
           If just one host thread is accessing memory, these read-modify-write accesses may be done non-atomically.
        3. From the device, perform some number of read-modify-write accesses to the memory.
           If just one work-item is accessing each memory location (but multiple work-items are accessing the allocation) these accesses may also be done non-atomically.
        4. Verify that the correct updates were made via direct access from the host, via mapping, or via memcpy, depending on supported capabilities.
    * May want to test multiple iterations, or to take other steps to increase the likelihood that accesses are made concurrently.
    * Are there any best practices we can borrow from fine-grain SVM testing?
    * TODO: See document issue 1, regarding the minimum supported granularity for concurrent access.
* [ ] `CL_SVM_CAPABILITY_CONCURRENT_ATOMIC_ACCESS_KHR`
    * Note, as described, these tests will require `CL_SVM_CAPABILITY_HOST_READ_KHR` and `CL_SVM_CAPABILITY_HOST_WRITE_KHR`, also.
    * [ ] Initialize a memory allocation with zero.
          Atomically increment the memory allocation from the device and the host (relaxed atomics, device scope).
          Verify that the correct updates were made via direct access from the host, via mapping, or via memcpy, depending on supported capabilities.
        * Note, requires `CL_DEVICE_ATOMIC_ORDER_RELAXED` and `CL_DEVICE_ATOMIC_SCOPE_ALL_DEVICES` capabilities to be included in `CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES`.
    * [ ] Initialize a memory allocation.
          Write values non-atomically to part of the allocation from the device, then write a flag to memory using a store-release, all svm devices atomic.
          On the host, poll the flag value using a load-acquire atomic.
          When the updated flag value is seen, read and verify the values that were written non-atomically.
          Write updated values non-atomically to part of the allocation from the host, then write a flag to memory using a store-release atomic.
          On the device, poll for the updated flag value using a load-acquire, all svm devices atomic.
          When the updated flag value is seen, read and verify the values that were written non-atomically.
          Repeat as needed.
    * May want to test multiple iterations, or to take other steps to increase the likelihood that accesses are made concurrently.
    * Are there any best practices we can borrow from fine-grain SVM testing?
* [X] `CL_SVM_CAPABILITY_INDIRECT_ACCESS_KHR`
    * [X] For devices that support `CL_SVM_CAPABILITY_DEVICE_READ_KHR`, initialize a memory allocation with a known value.
          On the host, embed the pointer to the allocation into an OpenCL buffer memory object.
          On the device, read the pointer out of the OpenCL buffer memory object, then read a value from the pointer.
          Store the value read to another OpenCL buffer memory object.
          Back on the host, verify the known value was read.

NOTE: Added by https://github.com/KhronosGroup/OpenCL-CTS/pull/2210.

### Testing New SVM APIs

* [X] `clSVMAllocWithPropertiesKHR`
    * [X] Test without a `CL_SVM_ALLOC_ALIGNMENT_KHR` property.
    * [X] Test without a `CL_SVM_ALLOC_ACCESS_FLAGS_KHR` property.
    * [X] For SVM types supporting `CL_SVM_CAPABILITY_DEVICE_UNASSOCIATED_KHR`, test without a `CL_SVM_ALLOC_ASSOCIATED_DEVICE_HANDLE_KHR` device.
    * [X] Test with varying the `CL_SVM_ALLOC_ALIGNMENT_KHR` property - all powers of two from 1 to 128 inclusive?
    * [X] Test with varying the `CL_SVM_ALLOC_ACCESS_FLAGS_KHR` property - all combinations?
    * [X] Include at least one test with all properties: `CL_SVM_ALLOC_ASSOCIATED_DEVICE_HANDLE_KHR` plus `CL_SVM_ALLOC_ALIGNMENT_KHR` plus `CL_SVM_ALLOC_ACCESS_FLAGS_KHR`.
    * [ ] TODO: Test zero-byte allocation?
* [X] `clSVMFreeWithPropertiesKHR`
    * Tested by the fixture; no flags or properties to test..
* [X] `clGetSVMPointerInfoKHR`
    * After allocating, perform each of the queries, both with and without an explicit `device` parameter, for the base pointer returned by the `clSVMAllocWithPropertiesKHR` and a pointer computed from the base pointer.
        * [X] `CL_SVM_INFO_TYPE_INDEX_KHR` - must match the index passed during allocation.
        * [X] `CL_SVM_INFO_CAPABILITIES_KHR` - must match the device capabilities for the explicit `device` parameter, or be a super-set of the platform capabilities otherwise.
        * [X] `CL_SVM_INFO_PROPERTIES_KHR` - must match the properties passed during allocation, unless the properties during allocation were `NULL`.
        * [X] `CL_SVM_INFO_ACCESS_FLAGS_KHR` - must match the access flags passed during allocation, or be zero.
        * [X] `CL_SVM_INFO_BASE_PTR_KHR` - must match the base of the allocation.
        * [X] `CL_SVM_INFO_SIZE_KHR` - must match the size passed during allocation.
        * [X] `CL_SVM_INFO_ASSOCIATED_DEVICE_HANDLE_KHR` - must match the associated device, or be `NULL`.
    * Test each of the queries for a bogus pointer (both with and without an explicit `device` parameter?).
        * [X] `CL_SVM_INFO_TYPE_INDEX_KHR` - must return `CL_UINT_MAX`.
        * [X] `CL_SVM_INFO_CAPABILITIES_KHR` - must return `0`.
        * [X] `CL_SVM_INFO_PROPERTIES_KHR` - must return size equal to `0`.
        * [X] `CL_SVM_INFO_ACCESS_FLAGS_KHR` - must return `0`?  See doc issue.
        * [X] `CL_SVM_INFO_BASE_PTR_KHR` - must return `NULL`.
        * [X] `CL_SVM_INFO_SIZE_KHR` - must return `0`.
        * [X] `CL_SVM_INFO_ASSOCIATED_DEVICE_HANDLE_KHR` - must return `NULL`.
* [X] `clGetSVMSuggestedTypeIndexKHR`
    * [X] Pass each of the supported device capabilities as `required_capabilities` and verify that the capabilities at `suggested_type_index` satisfy the required capabilities.
    * [X] Test without a `CL_SVM_ALLOC_ALIGNMENT_KHR` property.
    * [X] Test without a `CL_SVM_ALLOC_ACCESS_FLAGS_KHR` property.
    * [X] For SVM types supporting `CL_SVM_CAPABILITY_DEVICE_UNASSOCIATED_KHR`, test without a `CL_SVM_ALLOC_ASSOCIATED_DEVICE_HANDLE_KHR` device.
    * [X] Test with varying the `CL_SVM_ALLOC_ALIGNMENT_KHR` property - all powers of two from 1 to 128?
    * [X] Test with varying the `CL_SVM_ALLOC_ACCESS_FLAGS_KHR` property - all combinations?
    * [X] Include at least one test with all properties: `CL_SVM_ALLOC_ASSOCIATED_DEVICE_HANDLE_KHR` plus `CL_SVM_ALLOC_ALIGNMENT_KHR` plus `CL_SVM_ALLOC_ACCESS_FLAGS_KHR`.

NOTE: Added by https://github.com/KhronosGroup/OpenCL-CTS/pull/2261.
NOTE: Added by https://github.com/KhronosGroup/OpenCL-CTS/pull/2280.
NOTE: Added by https://github.com/KhronosGroup/OpenCL-CTS/pull/2338.

### Testing Existing SVM APIs

* [ ] `clSetKernelExecInfo(CL_KERNEL_EXEC_INFO_SVM_PTRS)`
    * [X] Follow similar methodology as `CL_SVM_CAPABILITY_INDIRECT_ACCESS_KHR`, except set the indirectly accessed allocation explicitly.
    * [X] Test a pointer offset from the base pointer and verify that the entire allocation may be accessed indirectly.
    * [ ] TODO: Pass an empty set and verify that this is not an error?
    * [ ] TODO: Pass a `NULL` pointer verify that this is not an error?
    * [ ] TODO: Pass a bogus pointer verify that this is not an error?
* [ ] `clSetKernelArgSVMPointer`
    * Generally do not need a targeted test, because this API will be exercised by any tests using SVM pointers in kernels.
    * [ ] Ensure at least one test for each SVM type passes a pointer offset from the base pointer and accesses it in a kernel.
    * [ ] TODO: Pass a `NULL` pointer, execute the kernel, and verify that no error occurs as long as the pointer is not dereferenced?
    * [ ] TODO: Pass a bogus pointer, execute the kernel, and verify that no error occurs as long as the pointer is not dereferenced?
* [ ] `clEnqueueSVMFree` (see: `test_enqueue_api.cpp`)
    * [X] Include an event on the command and verify that the event type is `CL_COMMAND_SVM_FREE`.
    * [X] Allocate memory for each type and verify that it can be freed asynchronously.
    * [ ] TODO: Pass an empty set and verify that this is not an error?
    * [ ] TODO: Pass a `NULL` pointer and verify that this is not an error?
* [ ] `clEnqueueSVMMemcpy` (see: `test_enqueue_api.cpp`)
    * [X] Include an event on the command and verify that the event type is `CL_COMMAND_SVM_MEMCPY`.
    * [X] Test all combinations of SVM pointer and host pointer sources and destinations.
    * [X] Test a pointer offset from the base pointer as a memcpy source and destination.
    * [ ] TODO: Check document issue 10 and consider copying to other devices or contexts.
    * [ ] TODO: Pass a `NULL` pointer and `size` equal to zero and verify that this is not an error, for both memcpy source and destination pointers?
    * [ ] TODO: Pass a bogus pointer and `size` equal to zero and verify that this is not an error, for both memcpy source and destination pointers?
    * [ ] TODO: For all SVM types, pass a valid pointer and `size` equal to zero and verify that this is not an error, for both memcpy source and destination pointers?
* [ ] `clEnqueueSVMMemFill` (see: `test_enqueue_api.cpp`)
    * [X] Include an event on the command and verify that the event type is `CL_COMMAND_SVM_MEMCPY`.
    * [X] Test multiple fill pattern sizes - all powers of two from 1 to 128 inclusive?
    * [X] Test a pointer offset from the base pointer as a fill destination.
    * [ ] TODO: Check document issue 9 and consider filling allocations for other devices or contexts.
    * [ ] TODO: Pass a `NULL` pointer and `size` equal to zero and verify that this is not an error?
    * [ ] TODO: Pass a bogus pointer and `size` equal to zero and verify that this is not an error?
    * [ ] TODO: For all SVM types, pass a valid pointer and `size` equal to zero and verify that this is not an error?
* [X] `clEnqueueSVMMap` / `clEnqueueSVMUnmap`
    * [X] Include an event on the command and verify that the event type is `CL_COMMAND_SVM_MAP` / `CL_COMMAND_SVM_UNMAP`.
    * [X] Test all combinations of map flags: `CL_MAP_READ`, `CL_MAP_WRITE`, `CL_MAP_WRITE_INVALIDATE_REGION`, `CL_MAP_READ | CL_MAP_WRITE`.
* [ ] `clEnqueueSVMMigrateMem`
    * [X] Include an event on the command and verify that the event type is `CL_COMMAND_SVM_MIGRATE_MEM`.
    * [X] Test all combinations of migration flags: `0`, `CL_MIGRATE_MEM_OBJECT_HOST`, `CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED`, `CL_MIGRATE_MEM_OBJECT_HOST | CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED`.
    * [X] Migrate an entire SVM allocation.
    * [X] Migrate a subset of an SVM allocation, starting from the base pointer.
    * [X] Migrate a subset of an SVM allocation, starting from a pointer offset from the base pointer.
    * [ ] TODO: Migrate a `NULL` pointer with `size` equal to zero and verify that this is not an error?
    * [ ] TODO: For all SVM types, migrate a valid pointer with `size` equal to zero and verify that this is not an error?

NOTE: Added by https://github.com/KhronosGroup/OpenCL-CTS/pull/2441.

### Non-Conventional Uses

Depending how these are resolved, they may create additional test items:

* [ ] TODO: `clCreateBuffer(CL_MEM_USE_HOST_PTR)`: Check document issue 20.
* [ ] TODO: `clCreateBuffer(CL_MEM_COPY_HOST_PTR)`: Check document issue 21.
* [ ] TODO: `clEnqueueReadBuffer` and `clEnqueueWriteBuffer` sources and destinations: Check document issue 22.
* [ ] TODO: `clEnqueueSVMMemFill`, etc. patterns: Check document issue 23.
