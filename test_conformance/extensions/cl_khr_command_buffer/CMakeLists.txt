set(MODULE_NAME CL_KHR_COMMAND_BUFFER)

set(${MODULE_NAME}_SOURCES
    main.cpp
    basic_command_buffer.cpp
    svm_command_basic.cpp
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
    command_buffer_finalize.cpp
    negative_command_buffer_finalize.cpp
    negative_command_buffer_svm_mem.cpp
    negative_command_buffer_copy_image.cpp
    negative_command_buffer_retain_release.cpp
    negative_command_buffer_create.cpp
    negative_command_buffer_copy.cpp
    negative_command_nd_range_kernel.cpp
    negative_command_buffer_get_info.cpp
    negative_command_buffer_barrier.cpp
    negative_command_buffer_fill.cpp
    negative_command_buffer_enqueue.cpp
)

set_gnulike_module_compile_flags("-Wno-sign-compare")

include(../../CMakeCommon.txt)

add_subdirectory( cl_khr_command_buffer_mutable_dispatch )
