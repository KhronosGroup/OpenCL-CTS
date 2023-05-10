set(MODULE_NAME CL_KHR_COMMAND_BUFFER)

set(${MODULE_NAME}_SOURCES
    main.cpp
    basic_command_buffer.cpp
    command_buffer_printf.cpp
    command_buffer_get_command_buffer_info.cpp
    command_buffer_set_kernel_arg.cpp
    command_buffer_event_sync.cpp
    command_buffer_out_of_order.cpp
    command_buffer_profiling.cpp
    command_buffer_queue_substitution.cpp
    command_buffer_test_fill.cpp
    command_buffer_test_copy.cpp
    command_buffer_test_barrier.cpp
    command_buffer_test_event_info.cpp
)

include(../../CMakeCommon.txt)

add_subdirectory( cl_khr_command_buffer_mutable_dispatch )
