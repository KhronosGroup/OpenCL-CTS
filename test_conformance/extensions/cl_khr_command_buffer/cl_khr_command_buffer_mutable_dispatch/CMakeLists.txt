set(MODULE_NAME CL_KHR_MUTABLE_DISPATCH)

set(${MODULE_NAME}_SOURCES
    main.cpp
    mutable_command_info.cpp
    mutable_command_simple.cpp
    mutable_command_out_of_order.cpp
    mutable_command_kernel.cpp
    ../basic_command_buffer.cpp
)

include(../../../CMakeCommon.txt)
