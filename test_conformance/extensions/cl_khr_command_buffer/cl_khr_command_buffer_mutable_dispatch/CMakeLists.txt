set(MODULE_NAME CL_KHR_MUTABLE_DISPATCH)

set(${MODULE_NAME}_SOURCES
    main.cpp
    mutable_command_info.cpp
    mutable_command_local_size.cpp
    ../basic_command_buffer.cpp
)

include(../../../CMakeCommon.txt)
