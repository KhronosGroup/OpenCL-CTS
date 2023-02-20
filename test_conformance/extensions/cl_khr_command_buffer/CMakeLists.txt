set(MODULE_NAME CL_KHR_COMMAND_BUFFER)

set(${MODULE_NAME}_SOURCES
    main.cpp
    basic_command_buffer.cpp
    command_buffer_get_command_buffer_info.cpp
    command_buffer_queue_substitution.cpp
    command_buffer_test_fill.cpp
    command_buffer_test_copy.cpp
    command_buffer_test_barrier.cpp
)

include(../../CMakeCommon.txt)
