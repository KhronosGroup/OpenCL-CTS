set(MODULE_NAME CL_KHR_MUTABLE_DISPATCH)

set(${MODULE_NAME}_SOURCES
    main.cpp
    command_buffer_test_mutable_dispatch.cpp
    ../basic_command_buffer.cpp
)

include(../../../CMakeCommon.txt)
