set(MODULE_NAME C11_ATOMICS)

set(${MODULE_NAME}_SOURCES
    common.cpp
    host_atomics.cpp
    main.cpp
    test_atomics.cpp
)

set(${MODULE_NAME}_KERNEL_LANG clc)
set(${MODULE_NAME}_TESTS
atomic_init
atomic_store
atomic_load
atomic_exchange
atomic_compare_exchange_weak
atomic_compare_exchange_strong
atomic_fetch_add
atomic_fetch_sub
atomic_fetch_and
atomic_fetch_or
atomic_fetch_orand
atomic_fetch_xor
atomic_fetch_xor2
atomic_fetch_min
atomic_fetch_max
atomic_flag
atomic_fence
svm_atomic_init
svm_atomic_store
svm_atomic_load
svm_atomic_exchange
svm_atomic_compare_exchange_weak
svm_atomic_compare_exchange_strong
svm_atomic_fetch_add
svm_atomic_fetch_sub
svm_atomic_fetch_and
svm_atomic_fetch_or
svm_atomic_fetch_orand
svm_atomic_fetch_xor
svm_atomic_fetch_xor2
svm_atomic_fetch_min
svm_atomic_fetch_max
svm_atomic_flag
svm_atomic_fence
)

include(../CMakeCommon.cmake)
