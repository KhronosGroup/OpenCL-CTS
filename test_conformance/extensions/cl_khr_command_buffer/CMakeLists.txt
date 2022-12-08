set(MODULE_NAME CL_KHR_COMMAND_BUFFER)

set(${MODULE_NAME}_SOURCES
    main.cpp
    basic_command_buffer.cpp
    command_buffer_test_fill.cpp
    command_buffer_test_copy.cpp
    command_buffer_test_kernel.cpp
)

include(../../CMakeCommon.txt)
