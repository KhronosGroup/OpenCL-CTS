set(MODULE_NAME DEVICE_EXECUTION)

set(DEVICE_EXECUTION_SOURCES
    device_info.cpp
    device_queue.cpp
    enqueue_block.cpp
    enqueue_flags.cpp
    enqueue_multi_queue.cpp
    enqueue_ndrange.cpp
    enqueue_wg_size.cpp
    enqueue_profiling.cpp
    execute_block.cpp
    host_multi_queue.cpp
    host_queue_order.cpp
    main.cpp
    nested_blocks.cpp
    utils.cpp
)

set(${MODULE_NAME}_KERNEL_LANG clc)
set(${MODULE_NAME}_TESTS
device_info
device_queue
execute_block
enqueue_block
enqueue_nested_blocks
enqueue_wg_size
enqueue_flags
enqueue_multi_queue
host_multi_queue
enqueue_ndrange
host_queue_order
enqueue_profiling
)

include(../CMakeCommon.cmake)

# end of file #
