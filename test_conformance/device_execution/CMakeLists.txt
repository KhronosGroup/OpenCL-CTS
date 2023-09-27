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

set_gnulike_module_compile_flags("-Wno-sign-compare")

include(../CMakeCommon.txt)
