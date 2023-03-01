set(MODULE_NAME ALLOCATIONS)

set(${MODULE_NAME}_SOURCES
        main.cpp
        allocation_execute.cpp
        allocation_fill.cpp
        allocation_functions.cpp
        allocation_utils.cpp
)

set(${MODULE_NAME}_KERNEL_LANG clc)
set(${MODULE_NAME}_TESTS
        buffer
        image2d_read
        image2d_write
        buffer_non_blocking
        image2d_read_non_blocking
        image2d_write_non_blocking
)

include(../CMakeCommon.cmake)
