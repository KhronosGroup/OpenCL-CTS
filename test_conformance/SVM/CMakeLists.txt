set(MODULE_NAME SVM)

set(${MODULE_NAME}_SOURCES
    main.cpp
    test_allocate_shared_buffer.cpp
    test_byte_granularity.cpp
    test_cross_buffer_pointers.cpp
    test_enqueue_api.cpp
    test_fine_grain_memory_consistency.cpp
    test_fine_grain_sync_buffers.cpp
    test_pointer_passing.cpp
    test_set_kernel_exec_info_svm_ptrs.cpp
    test_shared_address_space_coarse_grain.cpp
    test_shared_address_space_fine_grain.cpp
    test_shared_address_space_fine_grain_buffers.cpp
    test_shared_sub_buffers.cpp
    test_migrate.cpp
)

set(${MODULE_NAME}_KERNEL_LANG clc)
set(${MODULE_NAME}_TESTS
svm_byte_granularity
svm_set_kernel_exec_info_svm_ptrs
svm_fine_grain_memory_consistency
svm_fine_grain_sync_buffers
svm_shared_address_space_fine_grain
svm_shared_sub_buffers
svm_shared_address_space_fine_grain_buffers
svm_allocate_shared_buffer
svm_shared_address_space_coarse_grain_old_api
svm_shared_address_space_coarse_grain_new_api
svm_cross_buffer_pointers_coarse_grain
svm_pointer_passing
svm_enqueue_api
svm_migrate
)

include(../CMakeCommon.cmake)
