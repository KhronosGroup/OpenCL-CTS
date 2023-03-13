set(MODULE_NAME ATOMICS)

set(${MODULE_NAME}_SOURCES
        main.cpp
        test_atomics.cpp
        test_indexed_cases.cpp
)

set(${MODULE_NAME}_KERNEL_LANG clc)
set(${MODULE_NAME}_TESTS
atomic_add
atomic_sub
atomic_xchg
atomic_min
atomic_max
atomic_inc
atomic_dec
atomic_cmpxchg
atomic_and
atomic_or
atomic_xor
atomic_add_index
atomic_add_index_bin
)

include(../CMakeCommon.cmake)


