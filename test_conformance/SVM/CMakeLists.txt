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

include(../CMakeCommon.txt)
